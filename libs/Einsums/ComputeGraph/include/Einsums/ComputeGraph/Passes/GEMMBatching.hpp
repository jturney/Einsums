//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Collapse groups of independent GEMMs into a single
 * `blas::gemm_batch` call.
 *
 * Groups Einsum nodes that match all of:
 *   - Same dependency level (no data dependencies between members).
 *   - 2D×2D→2D contraction shape with exactly one link index (the
 *     `gemm_hint` populated at capture time).
 *   - Identical `m`, `n`, `k`, `trans_a`, `trans_b`, element type
 *     (@ref BlasScalar), and `alpha`/`beta` prefactors.
 *
 * Each such group is replaced with a single @ref OpKind::BatchedGemm
 * node whose executor extracts data pointers + leading dimensions
 * from all members at call time, packs them into pointer arrays, and
 * dispatches the type-appropriate `blas::gemm_batch<T>`. The batched
 * node's `inputs` vector holds 2N tensor ids (A_0, B_0, A_1, B_1, …);
 * its `outputs` holds N tensor ids (C_0, C_1, …).
 *
 * @note Today, `blas::gemm_batch` is an OpenMP-parallel loop over
 * per-matrix `dgemm` calls (see BLASVendor/src/gemm_batch.cpp). The
 * primary speedup from this pass therefore comes from batch-level
 * parallelism, not vendor-native batched dispatch. When a vendor
 * override (e.g. MKL's `cblas_dgemm_batch`) is wired in, this pass
 * picks up the additional benefit without any code change here.
 *
 * @par Leading-dimension handling:
 * `blas::gemm_batch` takes a single scalar `lda`/`ldb`/`ldc` shared
 * across the batch, different per-matrix strides can't be batched.
 * Verifies uniformity at pass time using the hint's extractors on the
 * current tensor state; if strides differ across the group, the pass
 * falls through without rewriting (a future optimisation could split
 * the group into stride-compatible sub-batches).
 *
 * In the default pipeline (planning phase, after ContractionPlanning and before
 * Reorder; `create_default` constructs it with the shared HardwareProfile so
 * the profitability gate below is active).
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("gemm_batching");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     // Two independent, identically-shaped GEMMs at the same dependency level
 *     // (no path between them). Same m,n,k, trans flags, dtype and prefactors:
 *     cg::einsum("ij <- ik ; kj", 0.0, &C1, 1.0, A1, B1);   // C1 = A1 * B1
 *     cg::einsum("ij <- ik ; kj", 0.0, &C2, 1.0, A2, B2);   // C2 = A2 * B2
 * }
 * graph.apply(cg::PassManager::create_default());           // GEMMBatching runs here
 * // The two einsums are replaced by one BatchedGemm node backed by blas::gemm_batch.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("gemm_batching")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C1, A1, B1)
 *     einsums.einsum("ij <- ik ; kj", C2, A2, B2)
 * g.apply(cg.default_pass_manager())                        # GEMMBatching runs in the default pipeline
 * # GEMMBatching is not a standalone Python-constructible pass: it takes a
 * # HardwareProfile and is applied only as part of the default manager.
 * @endcode
 *
 * @par Limitations
 * - Only groups **Einsum** nodes carrying a `gemm_hint` (2D x 2D -> 2D with
 *   exactly one link index, populated at capture); other einsums are ignored.
 * - Members must share dependency level and have bit-identical `m`, `n`, `k`,
 *   `trans_a`, `trans_b`, `BlasScalar` dtype, and bit-equal `alpha`/`beta`
 *   (no precision-drift matching: 1.0 never batches with 0.9999...).
 * - Conjugated einsums (`conj_a`/`conj_b`) are skipped; conjugation is not
 *   threaded through the batch rewrite.
 * - `blas::gemm_batch` takes one scalar `lda`/`ldb`/`ldc`; a group with
 *   non-uniform leading dimensions declines wholesale (no sub-batch split).
 * - Placement/interference gate: the batch occupies the first member's slot, so
 *   no outside node between the first and last member may read/write anything the
 *   batch reads or writes, and any `Loop`/`Conditional` in that span (I/O hidden
 *   in a sub-graph) disqualifies the group.
 * - Profitability gate applies only when a profile is supplied: a group whose
 *   estimated per-GEMM time exceeds `max_gemm_us` (default 100us) is left as
 *   independent nodes so the Dataflow executor can spread it across workers. The
 *   default-constructed pass has no gate and always batches.
 * - Today `blas::gemm_batch` is an OpenMP loop over per-matrix `dgemm`, so the
 *   win is batch-level parallelism amortizing per-node scheduling, not
 *   vendor-native batched dispatch.
 *
 * @par Future improvements
 * - Split a stride-mismatched group into stride-compatible sub-batches instead
 *   of declining it.
 * - Thread `conj_a`/`conj_b` through the rewrite so conjugated einsums batch.
 * - Gate batching on the absence of a distribution requirement so einsums that
 *   still need the distribution/GPU passes are not batched first (they do not
 *   inspect BatchedGemm nodes).
 * - Pick up a vendor-native batched GEMM (e.g. MKL's `cblas_dgemm_batch`) with
 *   no change here once `blas::gemm_batch` dispatches to it.
 */
class EINSUMS_EXPORT GEMMBatching : public OptimizerPass {
  public:
    GEMMBatching() = default;

    /// Profitability-gated batching. gemm_batch runs as ONE node - and on
    /// TaskPool workers BLAS is single-threaded - so collapsing large GEMMs
    /// that the Dataflow executor would otherwise spread across workers is a
    /// pessimization. With a profile, a group is only batched when the
    /// estimated single-GEMM time is at most @p max_gemm_us (default 100us,
    /// i.e. batching amortizes per-node scheduling for small GEMMs and stays
    /// away from work that deserves its own node). The default-constructed
    /// pass keeps the ungated always-batch behavior.
    explicit GEMMBatching(HardwareProfile profile, double max_gemm_us = 100.0)
        : _profile(std::move(profile)), _has_profile(true), _max_gemm_us(max_gemm_us) {}

    [[nodiscard]] std::string name() const override { return "GEMMBatching"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: batches sibling GEMMs
    /// within the single graph it's handed (one iteration's worth of
    /// nodes).
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of batches created this run (one per collapsed group).
    [[nodiscard]] size_t num_batches() const { return _num_batches; }

    /// Total einsum nodes absorbed into BatchedGemm nodes this run
    /// (sum of group sizes across all batches).
    [[nodiscard]] size_t total_batched() const { return _total_batched; }

    /// Groups skipped by the profitability gate this run.
    [[nodiscard]] size_t num_gate_skipped() const { return _num_gate_skipped; }

  private:
    HardwareProfile _profile{};
    bool            _has_profile{false};
    double          _max_gemm_us{100.0};
    size_t          _num_batches{0};
    size_t          _total_batched{0};
    size_t          _num_gate_skipped{0};
};

} // namespace einsums::compute_graph::passes
