//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Memory-aware topological sort pass.
 *
 * Re-sorts the graph's nodes using a priority-based Kahn's algorithm that
 * preferentially schedules nodes whose completion frees the most memory.
 * This reduces peak memory usage by releasing large intermediate tensors
 * as early as possible.
 *
 * **Algorithm:**
 * 1. Build the dependency adjacency list (same as standard topological sort)
 * 2. For each node, compute "bytes freed" = total bytes of tensors for which
 *    this node is the last consumer
 * 3. Use a max-heap priority queue: among ready nodes, pick the one that
 *    frees the most memory
 *
 * Only reorders independent nodes — data dependencies are always respected.
 *
 * @par Example
 * @code
 * auto [modified, reorder] = graph.apply<cg::passes::Reorder>();
 * // Nodes are now ordered to minimize peak memory
 * @endcode
 */
class EINSUMS_EXPORT Reorder : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "Reorder"; }

    /**
     * @brief Run the memory-aware reorder pass.
     * @param[in,out] graph The graph to reorder.
     * @return True if the node order changed.
     */
    bool run(Graph &graph) override;
};

} // namespace einsums::compute_graph::passes
