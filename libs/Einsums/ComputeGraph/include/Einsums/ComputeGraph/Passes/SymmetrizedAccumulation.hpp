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
 * This slice DETECTS and counts foldable sites (structural match + involutive
 * permutation + accumulate gate + interference guard); it performs NO rewrite.
 * It validates the pattern detection against real captured graphs before the
 * node model (@c OpKind::SymmetrizedAxpby) and the fused kernel are built. See
 * @c docs/symmetrized_accumulation_design.md.
 *
 * Not registered in @ref PassManager::create_default (workload-dependent, and
 * inert until the rewrite lands). Recurses into loop bodies - the CCSD residual
 * is captured as a loop body, so the sites live in a subgraph.
 */
class EINSUMS_EXPORT SymmetrizedAccumulation : public OptimizerPass {
  public:
    SymmetrizedAccumulation() = default;

    [[nodiscard]] std::string name() const override { return "SymmetrizedAccumulation"; }

    /// Matcher-only: never modifies the graph, always returns false.
    bool run(Graph &graph) override;

    /// The CCSD residual is a loop body; the sites live in the subgraph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Structural matches: permute (involutive, overwrite) whose output is the
    /// sole-produced/sole-consumed source of an accumulating axpby into the same
    /// output that a sibling accumulating axpby also feeds from the un-permuted
    /// source.
    [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Candidates that also pass the interference guard (no other node observes
    /// the half-symmetrized output or rewrites the source mid-fold) - i.e. the
    /// sites the rewrite slice will be able to fold safely.
    [[nodiscard]] size_t num_matched() const { return _num_matched; }

  private:
    size_t _num_candidates{0};
    size_t _num_matched{0};
};

} // namespace einsums::compute_graph::passes
