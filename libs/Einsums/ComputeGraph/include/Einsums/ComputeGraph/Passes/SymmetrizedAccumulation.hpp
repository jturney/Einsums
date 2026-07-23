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
 * @brief Detect the CCSD permutational-symmetrization idiom
 *        @f$r2 \mathrel{+}= s\,(t + P(t))@f$ (MATCHER-ONLY SLICE).
 *
 * Chemistry residuals accumulate a tensor and its index-swapped transpose into
 * the same output. In the graph this appears as four nodes per site (the
 * @c symacc helper in @c ccsd_rhf_graph_numpy_style.py):
 * @code
 * einsum(spec, A, B)            -> tmp    // tmp = pf*(A(x)B)   (kept)
 * axpby(s, tmp,  1.0, r2)                 // r2 += s*tmp
 * permute("jiba <- ijab", tmp) -> tmpP    // tmpP = P(tmp)
 * axpby(s, tmpP, 1.0, r2)                 // r2 += s*tmp^{jiba}
 * @endcode
 * The last three nodes fold to one node that makes a single sweep writing both
 * the element and its transposed partner, eliminating the @c tmpP buffer and
 * two of three O(o^2 v^2) sweeps per site.
 *
 * DETECTION: structural match + involutive permutation + accumulate gate +
 * interference guard. REWRITE (Level 1, permute reuse): for each safely-foldable
 * site over runtime tensors, rewrite the permute node to accumulate directly
 * into the output - @f$r2 = r2 + s_2\,P(t)@f$ via the existing string_permute
 * kernel with beta = 1 - and delete the second axpby and the @c tmpP buffer.
 * The first axpby (@f$r2 \mathrel{+}= s_1 t@f$) is left untouched. This drops the
 * O(o^2 v^2) @c tmpP storage and one full sweep per site with no new kernel or
 * OpKind. A later slice may fuse further to one sweep. See
 * @c docs/symmetrized_accumulation_design.md.
 *
 * Like LinearCombinationContractionFolding the rewrite fires only when the
 * operands are runtime tensors of one dtype (typed captures are left folded-out,
 * avoiding the typed-cast segfault class of bug-1015). Not registered in
 * @ref PassManager::create_default (workload-dependent). Recurses into loop
 * bodies - the CCSD residual is captured as a loop body.
 */
class EINSUMS_EXPORT SymmetrizedAccumulation : public OptimizerPass {
  public:
    SymmetrizedAccumulation() = default;

    [[nodiscard]] std::string name() const override { return "SymmetrizedAccumulation"; }

    bool run(Graph &graph) override;

    /// The CCSD residual is a loop body; the sites live in the subgraph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Structural matches: permute (involutive, overwrite) whose output is the
    /// sole-produced/sole-consumed source of an accumulating axpby into the same
    /// output that a sibling accumulating axpby also feeds from the un-permuted
    /// source.
    [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Candidates that also pass the interference guard (no other node observes
    /// the half-symmetrized output or rewrites the source mid-fold) - the sites
    /// that are safe to fold.
    [[nodiscard]] size_t num_matched() const { return _num_matched; }

    /// Sites actually rewritten (num_matched that also passed the runtime-tensor
    /// / uniform-dtype gate). Each eliminates one axpby and one tmpP buffer.
    [[nodiscard]] size_t num_rewritten() const { return _num_rewritten; }

  private:
    size_t _num_candidates{0};
    size_t _num_matched{0};
    size_t _num_rewritten{0};
};

} // namespace einsums::compute_graph::passes
