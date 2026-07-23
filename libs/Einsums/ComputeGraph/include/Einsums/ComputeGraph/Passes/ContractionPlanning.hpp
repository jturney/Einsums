//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
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
 * The cost model uses a HardwareProfile that can be auto-detected,
 * loaded from a JSON calibration file, or provided programmatically.
 *
 * In the default pipeline (planning phase, before GEMMBatching / Reorder and
 * before DistributionPlanning / Materialization, which size and allocate the
 * deferred intermediates it introduces; `create_default` passes the shared
 * HardwareProfile).
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
 * # HardwareProfile and is applied only as part of the default manager.
 * @endcode
 *
 * @par Limitations
 * - Restructures only chains of **pure** contractions: each member must have
 *   `c_prefactor == 0`, at least one link index, exactly one output and two
 *   inputs, and each output must feed exactly one operand of the next member.
 * - A chain must have >= 2 members and an estimated speedup > 1.05x; otherwise
 *   it gets a cost report only (analysis, no rewrite).
 * - Only **rank-2, non-runtime** chains are restructured. Higher-rank or
 *   runtime-rank chains are analysis-only: the emitted Gemm executor casts
 *   operands to `Tensor<T,2>*`, so a runtime tensor would be type confusion.
 * - Leaf-orientation gate: the folded GEMMs run `gemm<false,false>` on each
 *   leaf's physical layout, so a chain with a transposed operand
 *   (e.g. `ik;jk->ij`, link not a contiguous prefix/suffix) is declined - it
 *   would silently corrupt the result. The running product must enter each later
 *   member as `input_a` (matrix-chain DP assumes left-to-right leaf order).
 * - A squaring node (`OUT = T*T`) breaks the chain (`T` would be both link and
 *   leaf), and unknown dtypes are analysis-only.
 * - Interior outputs of the chain must be graph-owned, unaliased intermediates
 *   read by no node outside the chain; a user-visible or externally-read interior
 *   value makes the eliminated write observable, so the chain is declined.
 *
 * @par Future improvements
 * - Derive per-operand transpose flags from each leaf's captured index order and
 *   emit `gemm<transA,transB>`, lifting the canonical-orientation gate so
 *   transposed and higher-rank chains can also be restructured (needs folding).
 * - Restructure runtime-rank chains once the Gemm executor no longer assumes a
 *   static `Tensor<T,2>` layout.
 */
class EINSUMS_EXPORT ContractionPlanning : public OptimizerPass {
  public:
    /// Construct with auto-detected hardware profile.
    ContractionPlanning();

    /// Construct with a specific hardware profile.
    explicit ContractionPlanning(HardwareProfile profile);

    [[nodiscard]] std::string name() const override { return "ContractionPlanning"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Safe: restructuring a
    /// GEMM chain to its optimal parenthesization is numerically equivalent
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

    [[nodiscard]] std::vector<ChainReport> const &chain_reports() const { return _reports; }
    [[nodiscard]] size_t                          chains_restructured() const { return _chains_restructured; }
    [[nodiscard]] size_t                          intermediates_created() const { return _intermediates_created; }

  private:
    HardwareProfile          _profile;
    std::vector<ChainReport> _reports;
    size_t                   _chains_restructured{0};
    size_t                   _intermediates_created{0};
};

} // namespace einsums::compute_graph::passes
