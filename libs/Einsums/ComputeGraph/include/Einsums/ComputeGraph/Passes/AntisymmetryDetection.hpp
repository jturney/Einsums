//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Establish, by reading the data, which of a permutation operator's
 *        symmetries a bound input actually has.
 *
 * @ref AntisymmetryInference proves what an operator's output has STRUCTURALLY,
 * which reaches only the singleton-partition case. The facts a coupled-cluster
 * residual needs are about the problem rather than the graph: that the
 * amplitudes are antisymmetric in a pair of slots, that an energy denominator is
 * invariant under the permutations the residual applies. Nothing derives those,
 * and this pass does not ask a user to assert them either. It looks.
 *
 * For each operator in the graph it forms two candidate descriptors over the
 * operator's output axes, and tests them against the bound inputs of matching
 * shape:
 * - **within-group antisymmetry**, which is the precondition a coset operator's
 *   output needs before it carries any antisymmetry of its own;
 * - **invariance** under each permutation the operator sums over, which is what
 *   lets an elementwise division by that tensor preserve a numerator's
 *   antisymmetry.
 *
 * Whatever holds is recorded on @ref TensorHandle::symmetry_hint, which is graph
 * metadata. The backing tensor is never written, so a user's declared symmetry,
 * or absence of one, is left exactly as it was.
 *
 * @par Only inputs, and why that is not a limitation but a correctness gate
 * A tensor written by some node in this graph holds, at optimize time, whatever
 * it was initialized to. Graph-owned scratch is typically ZERO, and a zero tensor
 * satisfies every symmetry there is. Tagging one would record a fact that is true
 * of the buffer now and false of the value the graph will put in it, which is the
 * most direct route to a wrong answer this design has. So a candidate must have
 * NO writer in the graph: it is a bound input, and what is in it at optimize time
 * is what the contraction will read.
 *
 * @par Cost
 * One pass over a candidate per generator that HOLDS, and effectively nothing per
 * generator that does not, since @ref check_symmetry stops at the first
 * violation: measured at 1.47 ns an element against 0.4 microseconds for a whole
 * rank-6 tensor whose generator fails. Probing many candidates is therefore cheap
 * and only a real symmetry costs a sweep.
 *
 * @par The premise is DATA, which constrains what may be built on it
 * A fact established here describes the tensors bound right now. A rebind
 * replaces them, and `ContractionPlanning` states the house rule this runs into:
 * a measurement may found a hint, not a premise. Anything that REWRITES
 * arithmetic on the strength of one of these facts therefore has to be re-checked
 * per bound problem, which is what @ref OpKind::Setup exists for. See
 * `DESIGN-permutation-operator-folding.md`.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT AntisymmetryDetection : public OptimizerPass {
  public:
    APIARY_EXPOSE AntisymmetryDetection() = default;

    [[nodiscard]] std::string name() const override { return "AntisymmetryDetection"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::Analysis; }
    /// Every feature: an analysis pass annotates and never rewrites a node, which the pass manager enforces.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override { return NodeFeatures::all(); }

    /// @copydoc OptimizerPass::tier
    [[nodiscard]] PassTier tier() const override { return PassTier::BitwiseExact; }

    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// (tensor, generator) pairs tested against the data.
    APIARY_EXPOSE APIARY_GETTER("num_probed") [[nodiscard]] std::size_t num_probed() const { return _num_probed; }

    /// Generators found to hold.
    APIARY_EXPOSE APIARY_GETTER("num_found") [[nodiscard]] std::size_t num_found() const { return _num_found; }

    /// Tensors that gained at least one generator.
    APIARY_EXPOSE APIARY_GETTER("num_tensors") [[nodiscard]] std::size_t num_tensors() const { return _num_tensors; }

  private:
    std::size_t _num_probed{0};
    std::size_t _num_found{0};
    std::size_t _num_tensors{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
