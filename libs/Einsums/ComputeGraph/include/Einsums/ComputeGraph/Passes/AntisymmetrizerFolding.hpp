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
 * @brief Collapse a permutation operator contracted against an antisymmetric
 *        operand into a scalar multiple.
 *
 * When @f$W@f$ is antisymmetric under every permutation @f$P[G]@f$ sums over,
 * @f[
 *   \sum_x W(x)\,P[G](V)(x) \;=\; N \sum_x W(x)\,V(x)
 * @f]
 * with @f$N@f$ the operator's term count. Substituting @f$y = \sigma_t(x)@f$
 * turns each term into @f$\mathrm{sign}_t \sum_y W(\sigma_t^{-1}(y))V(y)@f$, and
 * @f$W@f$'s antisymmetry makes that @f$\mathrm{sign}_t^2 \sum_y W V@f$, so every
 * term contributes the same thing.
 *
 * The saving is the whole antisymmetrizer: @f$N-1@f$ permuted accumulations over
 * a tensor the size of the contraction's operands, replaced by one scalar
 * multiply. It is the transformation
 * `examples/toy/ccsd_t_spinorbital_toy.py` performs by hand between its "naive"
 * and "folded" (T) spellings, where its own table puts it at about a third of the
 * whole-tensor runtime.
 *
 * @par What it matches
 * - a @ref OpKind::Dot whose operands are @c A and @c B;
 * - @c B written by exactly one node, which carries an operator with a ZERO
 *   destination prefactor;
 * - @c A carrying a @ref TensorHandle::symmetry_hint that states antisymmetry on
 *   every axis the operator permutes;
 * - nothing between the two nodes writing either operand.
 *
 * The rewrite repoints the contraction at the operator's SOURCE and inserts a
 * scale by @f$N@f$ on the result. The operator node is left standing;
 * @ref DeadNodeElimination removes it when nothing else reads it, which in the
 * toy's naive (T) is true of @c V and false of @c W, matching by construction
 * what the hand-written folded spelling does.
 *
 * @par Where the premise comes from
 * @ref AntisymmetryInference, which must run first. Today that pass proves the
 * unconditional case only: an operator whose groups are all singletons. The
 * coset forms the triples correction uses need a fact about the operand that
 * nothing yet establishes, so this pass does not fire on them. See
 * `DESIGN-permutation-operator-folding.md`.
 *
 * @par Ordering
 * Before @ref AntisymmetrizerExpansion, which strips the operator and so removes
 * the thing this matches on.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AntisymmetrizerFolding : public OptimizerPass {
  public:
    APIARY_EXPOSE AntisymmetrizerFolding() = default;

    [[nodiscard]] std::string name() const override { return "AntisymmetrizerFolding"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }

    /// @copydoc OptimizerPass::tier
    ///
    /// The folded form sums ONE term where the original summed @f$N@f$ and then
    /// multiplies, so it is exact in its algebra and not to the last bit: the
    /// discarded terms were equal in exact arithmetic and merely close in
    /// floating point.
    [[nodiscard]] PassTier tier() const override { return PassTier::ReAssociating; }

    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// A residual is captured as a loop body.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Contractions examined whose operand came from an operator.
    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] std::size_t num_candidates() const { return _num_candidates; }

    /// Contractions rewritten.
    APIARY_EXPOSE APIARY_GETTER("num_folded") [[nodiscard]] std::size_t num_folded() const { return _num_folded; }

  private:
    std::size_t _num_candidates{0};
    std::size_t _num_folded{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
