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
 * @brief Lower a contraction's permutation operator into an explicit
 *        contraction plus one permuted accumulation per term.
 *
 * A spec may name a permutation operator (@ref PermutationOperator), as in
 * @code
 * cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,e ; m,b,e,j", 1.0, &r2, 1.0, t2, W);
 * @endcode
 * and a node carrying one already computes the right answer without this pass:
 * @ref dispatch::string_einsum contracts into a temporary and accumulates the
 * terms out of it. What it cannot do is own that temporary. It is a
 * @c std::vector sized from C on EVERY call, which for an @f$o^2v^2@f$ residual
 * term is an allocation per replay and, at production dimensions, an allocation
 * far too large to make per replay.
 *
 * This pass rewrites the site into nodes the graph owns:
 * @code
 * einsum(spec without the operator)  -> tmp     // tmp = ab_pf * (A (x) B)
 * permute(term_0 <- c_indices)       -> C       // C = c_pf * C + sign_0 * tmp
 * permute(term_t <- c_indices)       -> C       // C = C + sign_t * P_t(tmp)
 * @endcode
 * The temporary becomes an ordinary intermediate, so Materialization,
 * MemoryPlanning and the lifetime passes treat it as they treat every other one,
 * and the allocation happens once instead of per call.
 *
 * The contraction cannot write C directly: the permuted accumulations would then
 * read what they are writing. C's own prefactor rides on the FIRST term, which
 * is what keeps it applied exactly once rather than once per term, and the
 * identity term is always first so that term is the one that may overwrite.
 *
 * @par What this does not do
 * A @ref OpKind::Permute node carrying an operator is LEFT ALONE. It needs no
 * temporary, because every term reads a source the operation never writes, so
 * lowering it would replace one node with N that do exactly the same work in the
 * same order. That is a node-count regression with nothing bought: the N
 * permutes share one source and one destination, so there is nothing for CSE to
 * merge and nothing for PermuteFusion to fold. The whole-tensor (T)
 * antisymmetrizer in @c examples/toy/ccsd_t_spinorbital_toy.py is the case,
 * where it would turn two nodes back into eighteen.
 *
 * @par Ordering
 * Early in the graph-transforming group, ahead of ConstantFolding, so everything
 * downstream sees the lowered form: CSE and DeadNodeElimination over the emitted
 * nodes, LoopInvariantHoisting over the temporary's construction, and
 * MemoryPlanning over its lifetime. Recurses into subgraphs, because a CCSD
 * residual is captured as a loop body.
 *
 * @par Example
 * @code
 * cg::Graph graph("ring");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,e ; m,b,e,j", 1.0, &r2, 1.0, t2, W);
 * }
 * graph.apply(cg::PassManager::create_default());
 * // one einsum node becomes an einsum into scratch plus four accumulating permutes
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AntisymmetrizerExpansion : public OptimizerPass {
  public:
    APIARY_EXPOSE AntisymmetrizerExpansion() = default;

    [[nodiscard]] std::string name() const override { return "AntisymmetrizerExpansion"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }
    /// @copydoc OptimizerPass::understood_features
    /// Every feature today: it expands the operators, carries conjugation and live prefactors onto the rebuilt nodes, and declines a
    /// mixed-precision contraction or a scalar destination.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override {
        return NodeFeatures{} | NodeFeature::PermutationOperators | NodeFeature::Views | NodeFeature::Conjugation |
               NodeFeature::ComplexPrefactor | NodeFeature::MixedPrecision | NodeFeature::Grouped | NodeFeature::ControlFlow |
               NodeFeature::Tiled | NodeFeature::RawScalar | NodeFeature::RedirectedSlot;
    }

    /// @copydoc OptimizerPass::tier
    ///
    /// The lowered form performs the SAME operations in the same order as the
    /// node it replaces, because the un-lowered executor already contracts into
    /// a temporary and applies the same terms out of it. A permute is data
    /// movement plus one multiply-add per element and reassociates nothing, and
    /// the emitted nodes all write C, so the hazard edges hold them in order
    /// under any scheduler.
    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// A CCSD residual is a loop body, so the sites live in the subgraph.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Einsum nodes found carrying at least one permutation operator.
    APIARY_EXPOSE APIARY_GETTER("num_sites") [[nodiscard]] std::size_t num_sites() const { return _num_sites; }

    /// Sites actually rewritten: those that also passed the operand gate (one
    /// dtype, a rank-erased impl on every operand, known extents for the
    /// temporary).
    APIARY_EXPOSE APIARY_GETTER("num_expanded") [[nodiscard]] std::size_t num_expanded() const { return _num_expanded; }

    /// Permuted accumulations emitted across every rewritten site, which is the
    /// sum of the sites' term counts.
    APIARY_EXPOSE APIARY_GETTER("num_terms") [[nodiscard]] std::size_t num_terms() const { return _num_terms; }

  private:
    std::size_t _num_sites{0};
    std::size_t _num_expanded{0};
    std::size_t _num_terms{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
