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
 * @brief Pull a sum of antisymmetrized quantities inside one antisymmetrizer.
 *
 * A permutation operator is LINEAR, so
 * @f[
 *   a\,P[G](B) \;+\; s\,\bigl(c\,P[G](A)\bigr) \;=\; P[G]\bigl(a B + s c A\bigr)
 * @f]
 * and a graph that antisymmetrizes two things and adds them can antisymmetrize
 * once instead.
 *
 * @par Why it matters more than the node it saves
 * On its own the rewrite trades one permuted-accumulation sweep for an add, which
 * is worth having and is not the point. The point is that the sum's producer
 * becomes an OPERATOR again. @ref AntisymmetrizerFolding matches a contraction
 * whose operand was written by an operator, and a residual writes
 * `V := W + P(Xd)`, so the operand's producer is a linear combination and the
 * fold cannot see the operator standing behind it. This puts it back where the
 * fold can reach, which is what turns an N-term antisymmetrizer into a scalar
 * multiply.
 *
 * It is the same move the hand-written folded spelling of
 * `examples/toy/ccsd_t_spinorbital_toy.py` makes when it accumulates
 * `Xc := Xc + Xd` and antisymmetrizes once.
 *
 * @par What it matches
 * - a tensor written exactly twice: an operator with a ZERO destination
 *   prefactor, then an `Axpby` accumulating another tensor into it;
 * - that other tensor written exactly once, by an operator with the SAME groups
 *   and a zero destination prefactor;
 * - both tensors graph-owned intermediates untouched by a child sub-graph.
 *
 * The first operator's source is replaced by a freshly built sum, its prefactor
 * folded into that sum, and the accumulation deleted. The second operator is left
 * standing, because a residual reads it separately (the toy divides it by the
 * energy denominator) and @ref DeadNodeElimination removes it when nothing does.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AntisymmetrizerLinearity : public OptimizerPass {
  public:
    APIARY_EXPOSE AntisymmetrizerLinearity() = default;

    [[nodiscard]] std::string name() const override { return "AntisymmetrizerLinearity"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }
    /// @copydoc OptimizerPass::understood_features
    /// Not views or complex prefactors: the merged sum is sized from the root and its alpha is composed as a real number.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override {
        return NodeFeatures{} | NodeFeature::PermutationOperators | NodeFeature::Conjugation | NodeFeature::Grouped |
               NodeFeature::ControlFlow | NodeFeature::Tiled | NodeFeature::RawScalar | NodeFeature::RedirectedSlot;
    }

    /// @copydoc OptimizerPass::tier
    /// Linearity is exact in the algebra, and the sum is formed before the
    /// antisymmetrizer rather than after, which reorders the additions.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Sites where one antisymmetrized quantity accumulates into another.
    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] std::size_t num_candidates() const { return _num_candidates; }

    /// Sites rewritten.
    APIARY_EXPOSE APIARY_GETTER("num_merged") [[nodiscard]] std::size_t num_merged() const { return _num_merged; }

  private:
    std::size_t _num_candidates{0};
    std::size_t _num_merged{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
