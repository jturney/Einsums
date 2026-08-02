//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <string>
#include <vector>

namespace einsums::compute_graph::passes {

/**
 * @brief Multi-objective contraction planning pass.
 *
 * Detects chains of GEMM-pattern einsum nodes and computes the optimal
 * contraction order using a cost model that considers:
 *   - FLOPs (compute time scaled by shape-dependent GEMM efficiency)
 *   - Memory traffic (read/write bandwidth to main memory)
 *   - Kernel overhead (BLAS call and allocation costs)
 *   - GEMM shape efficiency (small/skinny GEMMs are slower per FLOP)
 *
 * This pass actually
 * restructures the graph: it declares deferred intermediate tensors and
 * rewrites the contraction sequence (via a standard matrix-chain DP over the
 * chain's leaf matrices) to minimize estimated wall-clock time.
 *
 * The cost model uses a CostModel that can be auto-detected,
 * loaded from a JSON calibration file, or provided programmatically.
 *
 * In the default pipeline (planning phase, before GEMMBatching / Reorder and
 * before DistributionPlanning / Materialization, which size and allocate the
 * deferred intermediates it introduces; `create_default` passes the shared
 * CostModel).
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("contraction_planning");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     // A chain whose left-to-right order is not the cheapest parenthesization.
 *     // T is a graph-owned intermediate consumed only by the second GEMM.
 *     cg::einsum("il <- ik ; kl", 0.0, &T, 1.0, A, B);   // T = A * B
 *     cg::einsum("in <- il ; ln", 0.0, &D, 1.0, T, C);   // D = (A*B) * C
 * }
 * graph.apply(cg::PassManager::create_default());        // ContractionPlanning runs here
 * // If (A*(B*C)) is cheaper, the chain is re-parenthesized and T's write is
 * // replaced by a fresh deferred intermediate.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("contraction_planning")
 * with cg.capture(g):
 *     einsums.einsum("il <- ik ; kl", T, A, B)   # T = A * B
 *     einsums.einsum("in <- il ; ln", D, T, C)   # D = (A*B) * C
 * g.apply(cg.default_pass_manager())             # ContractionPlanning runs in the default pipeline
 * # ContractionPlanning is not a standalone Python-constructible pass: it takes a
 * # CostModel and is applied only as part of the default manager.
 * @endcode
 *
 * @par Limitations
 * - Restructures only chains of **pure** contractions: each member must have
 *   `c_prefactor == 0`, at least one link index, exactly one output and two
 *   inputs, and each output must feed exactly one operand of the next member.
 * - A chain must have >= 2 members and an estimated speedup > 1.05x; otherwise
 *   it gets a cost report only (analysis, no rewrite).
 * - Only **rank-2 shaped** chains are restructured; a higher-rank operand would
 *   need its outer dimensions flattened, which this pass does not do. Rank-2
 *   *runtime-rank* tensors are fine: the emitted executor reaches each operand
 *   through its impl and the dynamic-rank `gemm` overload, so there is no
 *   `Tensor<T,2>*` cast to be confused by.
 * - Leaf orientation is carried as a transpose flag on the emitted GEMM, so a
 *   chain captured with transposed operands (`ik;jk->ij`) folds like any other
 *   and no permute is inserted. What still cannot be expressed: a member whose
 *   OUTPUT is permuted (`ji <- ik ; kj`) - there is no flag that writes an M x N
 *   result as N x M - an operand whose link indices are interleaved with its
 *   targets (no flat `(M,K)` reading at all), and a chain where the running
 *   product enters a later member as `input_b` (the matrix-chain DP assumes
 *   left-to-right leaf order, and GEMM does not commute).
 * - A squaring node (`OUT = T*T`) breaks the chain (`T` would be both link and
 *   leaf), and unknown dtypes are analysis-only.
 * - Interior outputs of the chain must be graph-owned, unaliased intermediates
 *   read by no node outside the chain; a user-visible or externally-read interior
 *   value makes the eliminated write observable, so the chain is declined.
 *
 * @par Future improvements
 * - Flatten higher-rank operands into a rank-2 `(M,K)` reading so those chains
 *   restructure too.
 * - Insert a permute for an operand whose link indices are interleaved with its
 *   targets, priced against the saving with
 *   @ref DeviceProfile::estimate_permute_time_us. Only reachable above rank 2,
 *   so it is blocked behind the flattening above.
 * - Reorder the leaf list when the running product enters a member as `input_b`,
 *   instead of declining the chain.
 */
class EINSUMS_EXPORT ContractionPlanning : public OptimizerPass {
  public:
    /// Construct with auto-detected hardware cost_model.
    ContractionPlanning();

    /// Construct with a specific hardware cost_model.
    explicit ContractionPlanning(CostModel cost_model);

    [[nodiscard]] std::string              name() const override { return "ContractionPlanning"; }
    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// Recurse into loop bodies / conditional branches.
    ///
    /// Safe: restructuring a GEMM chain to its optimal parenthesization is numerically equivalent
    /// (matrix-chain associativity), per-graph, and the intermediates it
    /// creates via create_tensor_dynamic are *eager* (allocated at pass time,
    /// not deferred) so they don't depend on the Materialization pass that
    /// runs earlier in the pipeline.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    // ── Report ─────────────────────────────────────────────────────────────

    /// Per-chain report from the last run.
    struct ChainReport {
        size_t              chain_length{0};
        std::vector<size_t> dimensions; ///< p[0..n] dimension array
        double              original_time_us{0};
        double              optimal_time_us{0};
        double              speedup{1.0};
        size_t              intermediates_created{0};
        double              comm_cost_us{0};        ///< Estimated communication cost (allreduce etc.)
        bool                has_distributed{false}; ///< True if chain involves distributed tensors
    };

    /// Chain reports accumulated over the whole apply(), including every
    /// subgraph. ``_reports`` alone would hold only the last fixpoint iteration
    /// of the last subgraph visited; ``graph.explain()`` reads this getter.
    [[nodiscard]] std::vector<ChainReport> const &chain_reports() const { return _apply_reports; }
    [[nodiscard]] size_t                          chains_restructured() const { return _chains_restructured; }
    [[nodiscard]] size_t                          intermediates_created() const { return _intermediates_created; }

  private:
    CostModel _cost_model;
    /// Reports for the CURRENT graph's last fixpoint iteration; cleared per
    /// iteration because restructuring invalidates the chains it describes.
    std::vector<ChainReport> _reports;
    /// Per-apply accumulation of the above, appended at the end of each run().
    std::vector<ChainReport> _apply_reports;
    size_t                   _chains_restructured{0};
    size_t                   _intermediates_created{0};
};

} // namespace einsums::compute_graph::passes
