//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

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
 * across the batch — different per-matrix strides can't be batched.
 * Verifies uniformity at pass time using the hint's extractors on the
 * current tensor state; if strides differ across the group, the pass
 * falls through without rewriting (a future optimisation could split
 * the group into stride-compatible sub-batches).
 */
class EINSUMS_EXPORT GEMMBatching : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "GEMMBatching"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: batches sibling GEMMs
    /// within the single graph it's handed (one iteration's worth of
    /// nodes). See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of batches created this run (one per collapsed group).
    [[nodiscard]] size_t num_batches() const { return _num_batches; }

    /// Total einsum nodes absorbed into BatchedGemm nodes this run
    /// (sum of group sizes across all batches).
    [[nodiscard]] size_t total_batched() const { return _total_batched; }

  private:
    size_t _num_batches{0};
    size_t _total_batched{0};
};

} // namespace einsums::compute_graph::passes
