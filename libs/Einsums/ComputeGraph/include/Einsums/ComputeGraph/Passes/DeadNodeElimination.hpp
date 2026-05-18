//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Dead node elimination pass.
 *
 * Removes nodes whose outputs are never consumed by any subsequent node.
 * A tensor is "live" if it's read by at least one node that is itself live.
 * Nodes that only write to dead tensors are eliminated.
 *
 * This is useful after CSE or other passes that redirect consumers away
 * from a node — the original producer may become dead.
 *
 * The pass considers ALL output tensors as potentially live (they may be
 * read outside the graph by user code). Only tensors that are graph-owned
 * intermediates (is_intermediate=true) with no readers are candidates
 * for dead elimination.
 *
 * @note Control flow nodes (Conditional, Loop) and Alloc/Free nodes are
 *       never eliminated.
 */
class EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph") EINSUMS_PYBIND_HOLDER(std::shared_ptr) EINSUMS_EXPORT DeadNodeElimination
    : public OptimizerPass {
  public:
    EINSUMS_PYBIND_EXPOSE DeadNodeElimination() = default;

    [[nodiscard]] std::string name() const override { return "DeadNodeElimination"; }
    bool                      run(Graph &graph) override;

    /// Number of nodes eliminated in the last run.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_GETTER("num_eliminated") [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
