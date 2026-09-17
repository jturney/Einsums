//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Tag the output of a permutation operator with the antisymmetry it
 *        unconditionally has.
 *
 * A node whose spec names a permutation operator (@ref PermutationOperator)
 * produces a tensor with structure, and this records what that structure is so a
 * later pass can rewrite arithmetic against it. Today the consumer is
 * @ref AntisymmetrizerFolding.
 *
 * @par The rule, and the narrow case it covers
 * For an operator whose groups are all SINGLETONS, the expansion is
 * @f$P = \sum_\sigma \mathrm{sign}(\sigma)\,\sigma@f$ over the whole symmetric
 * group on its letters, so @f$\tau P = \mathrm{sign}(\tau) P@f$ for every
 * transposition and the output is antisymmetric in those axes whatever the
 * operand was.
 *
 * For an operator with a group of two or more this is FALSE, and measurably so.
 * `P(i/jk)` applied to arbitrary rank-3 data is antisymmetric under none of
 * `(ij)`, `(ik)` or `(jk)`: a coset form carries the antisymmetry only when its
 * operand already has it within each group, which is the same precondition the
 * operator's well-definedness rests on. Those operators are DECLINED here rather
 * than tagged, and reaching them needs a fact about the operand that this pass
 * has no way to establish. See `DESIGN-permutation-operator-folding.md`.
 *
 * @par Soundness
 * Three gates, all load-bearing:
 * - the destination prefactor must be ZERO, since with a nonzero one the result
 *   is the previous contents plus an antisymmetric part, which is not
 *   antisymmetric;
 * - the tensor must have exactly one writer in this graph and no reference from
 *   a child sub-graph, which is @ref EscapeAnalysis, shared with
 *   @ref SymmetryPropagation rather than re-derived;
 * - only graph-owned intermediates are tagged, never a user's tensor.
 *
 * @par Ordering
 * Before @ref AntisymmetrizerExpansion, which strips the operator and so destroys
 * the evidence this rule reads. That is also why this is a separate pass from
 * @ref SymmetryPropagation, which runs after Materialization for a different
 * purpose and would find nothing left to infer from.
 *
 * Writes only @ref TensorHandle::symmetry_hint, which is graph metadata, so it
 * needs no backing storage and can run before anything is materialized.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AntisymmetryInference : public OptimizerPass {
  public:
    APIARY_EXPOSE AntisymmetryInference() = default;

    [[nodiscard]] std::string name() const override { return "AntisymmetryInference"; }

    /// @copydoc OptimizerPass::phase
    /// Annotation only; never rewrites the node set.
    [[nodiscard]] PassPhase phase() const override { return PassPhase::Analysis; }

    /// @copydoc OptimizerPass::tier
    /// Changes no arithmetic at all.
    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// A CCSD residual is a loop body, and the guard above is what makes
    /// recursing into one safe.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Operator-bearing nodes examined.
    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] std::size_t num_candidates() const { return _num_candidates; }

    /// Outputs tagged antisymmetric.
    APIARY_EXPOSE APIARY_GETTER("num_tagged") [[nodiscard]] std::size_t num_tagged() const { return _num_tagged; }

  private:
    std::size_t _num_candidates{0};
    std::size_t _num_tagged{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
