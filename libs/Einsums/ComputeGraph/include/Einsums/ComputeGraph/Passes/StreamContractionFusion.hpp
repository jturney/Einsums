//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <string>

namespace einsums::compute_graph::passes {

/**
 * @brief Fuse contractions that stream the same large tensor into one pass
 *        over it (the SCF "J and K from one TEI read" idiom).
 *
 * Detects groups of einsum nodes that each contract the SAME large tensor
 * @f$S@f$ against a small operand, where every output index and every
 * small-operand index is drawn from @f$S@f$'s index pattern (the
 * "GEMV-shaped" class: each element of @f$S@f$ is read exactly once and
 * contributes @f$\alpha_k\,S[\mathrm{idx}]\,W_k[\pi_k(\mathrm{idx})]@f$ to
 * @f$C_k[\rho_k(\mathrm{idx})]@f$). Such contractions are memory-bandwidth
 * bound - their cost is one stream of @f$S@f$ - so @f$N@f$ of them cost
 * @f$N@f$ streams executed separately, but only ONE stream when fused: the
 * replacement node walks @f$S@f$ once in storage order and feeds every
 * member's accumulator from the same element.
 *
 * @par Example (Fock build)
 * Before (two 800 MB streams of the TEI at n = 100, plus the scrambled K
 * pattern running below stream speed):
 * @code
 * // J[mu,nu] += 2 * TEI[mu,nu,lam,sig] * D[lam,sig]
 * // K[mu,nu] -= 1 * TEI[mu,lam,nu,sig] * D[lam,sig]
 * @endcode
 * After: one storage-order stream of TEI updating both J and K, with
 * thread-private accumulators for the (small) outputs.
 *
 * Members may write different outputs (J and K) or accumulate into a shared
 * one; for a shared output every member after the first must purely
 * accumulate (c prefactor 1), mirroring
 * @ref LinearCombinationContractionFolding's tail rule. Conjugated members,
 * repeated indices, and complex prefactors are left unfused. The streamed
 * tensor must dominate the member operands and outputs in size, and outputs
 * are capped so per-thread privatization stays cheap (both thresholds below).
 *
 * @par Relationship to LinearCombinationContractionFolding
 * LCCF serves the same 2J-K algebra by materializing a linear combination
 * @f$L@f$ of the streamed tensor and contracting once - measured 2.7x over
 * unfused for the Fock idiom, but it still makes ~4 passes over
 * @f$S@f$-sized data. This pass replaces those with a single pass. When both
 * passes are registered, run this one FIRST: it consumes the pattern LCCF
 * would otherwise fold.
 *
 * @par Opt-in
 * Not registered in @ref PassManager::create_default; workload-dependent.
 * @code
 * cg::PassManager pm; pm.add(std::make_shared<StreamContractionFusion>());
 * graph.apply(pm);
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT StreamContractionFusion : public OptimizerPass {
  public:
    APIARY_EXPOSE StreamContractionFusion() = default;

    [[nodiscard]] std::string name() const override { return "StreamContractionFusion"; }

    bool run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local rewrite within the
    /// graph it is handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of fused stream groups created.
    [[nodiscard]] size_t num_groups() const { return _num_groups; }

    /// Total number of einsum nodes eliminated (fused away).
    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    /// The streamed tensor must have at least this many elements. Measured
    /// (Apple M4, Fock J/K pair): the fused kernel beats the unfused graph at
    /// every size down to 4096 elements by 1.5-2.9x, because fusing removes a
    /// whole node dispatch (and any permute the member would need) while the
    /// fused kernel costs no more than one node. This floor only excludes
    /// trivial streams where either path is measurement noise.
    static constexpr size_t kMinStreamElems = size_t{1} << 12;

    /// Outputs above this element count are not privatized per thread;
    /// members writing them stay unfused.
    static constexpr size_t kMaxOutputElems = size_t{1} << 22;

    /// The streamed tensor must be at least this many times larger than each
    /// member's output and small operand for the "one stream dominates"
    /// argument to hold.
    static constexpr size_t kMinSizeRatio = 8;

    size_t _num_groups{0};
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
