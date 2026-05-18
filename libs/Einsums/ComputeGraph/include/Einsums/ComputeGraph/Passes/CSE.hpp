//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Common Subexpression Elimination (CSE) pass.
 *
 * Detects pairs of nodes that compute the same expression — identical OpKind,
 * identical input TensorIds, and identical operation metadata (EinsumDescriptor,
 * ScaleDescriptor, etc.). The duplicate node is eliminated and all downstream
 * consumers of its outputs are redirected to the original node's outputs.
 *
 * **Equivalence criteria:**
 * - Same OpKind
 * - Same input TensorIds (in order)
 * - Same OpData (prefactors, index patterns, flags must all match)
 * - Same number of outputs
 *
 * @par Example
 * @code
 * // Before CSE: two identical einsums
 * cg::einsum("ik;kj->ij", &C, A, B);
 * cg::einsum("ik;kj->ij", &D, A, B);
 *
 * // After CSE: second einsum removed, D's consumers use C's result
 * auto [modified, cse] = graph.apply<cg::passes::CSE>();
 * @endcode
 *
 * @note Tensor ID redirections are propagated through all remaining nodes'
 *       input lists, so multi-level CSE works in a single pass.
 */
class EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph") EINSUMS_PYBIND_HOLDER(std::shared_ptr) EINSUMS_EXPORT CSE
    : public OptimizerPass {
  public:
    EINSUMS_PYBIND_EXPOSE CSE() = default;

    [[nodiscard]] std::string name() const override { return "CSE"; }

    /**
     * @brief Run the CSE pass on the graph.
     * @param[in,out] graph The graph to optimize.
     * @return True if at least one duplicate node was eliminated.
     */
    bool run(Graph &graph) override;
};

} // namespace einsums::compute_graph::passes
