//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Memory-aware topological sort: reschedules independent nodes so the ones that free the most memory run earliest.
 *
 * Re-sorts the node list with a priority-based Kahn's algorithm. The dependency graph encodes all three hazard classes -- RAW
 * (writer->reader), WAW (writer->writer), and WAR (reader->writer) -- over each node's *effective* I/O (so a Loop/Conditional
 * carries edges for the tensors its body touches) with view aliases resolved to the owning buffer, so only genuinely independent
 * nodes may swap. Among the ready set, a max-heap picks the node whose completion frees the most bytes (it is the last consumer
 * of the largest total tensor storage), releasing big intermediates as early as data dependencies allow and lowering peak
 * memory. The pass moves nodes only; it never adds, removes, or rewrites them.
 *
 * In the default pipeline (populate_default, after GEMMBatching so the scheduler sees a batched node as one unit). Not in the
 * O1 cleanup cluster of PassManager::create_for.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("reorder");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &Big, 1.0, A, B);   // large intermediate
 *     cg::einsum("pq <- pr ; rq", 0.0, &Small, 1.0, X, Y); // independent, small
 *     cg::einsum("i <- ij", 0.0, &r, 1.0, Big);            // last consumer of Big -> frees it
 * }
 * cg::PassManager pm; pm.add<cg::passes::Reorder>();
 * graph.apply(pm);                                          // or PassManager::create_default()
 * // Among ready nodes the reducer over Big is scheduled early so Big is freed before Small's chain grows the footprint.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("reorder")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", Big, A, B)
 *     einsums.einsum("pq <- pr ; rq", Small, X, Y)
 *     einsums.einsum("i <- ij", r, Big)
 * pm = cg.PassManager(); pm.add(cg.Reorder())
 * g.apply(pm)                                              # or cg.default_pass_manager()
 * # Reorder exposes no result counter; it only permutes the node order to lower peak memory.
 * @endcode
 *
 * @par Limitations
 * - It **reschedules only**: it never fuses, splits, or eliminates nodes, and it cannot lower peak memory below what the fixed
 *   node set and its true dependencies allow.
 * - The "bytes freed" ranking is a **greedy local heuristic** (largest last-consumer tensor first), not a globally optimal
 *   peak-memory schedule.
 * - If the dependency graph is cyclic or otherwise leaves nodes unscheduled (`order.size() != n`), the pass bails and leaves the
 *   node order untouched.
 * - Tensor sizes come from `TensorHandle::total_bytes()`; the pass runs before Materialization, so sizes are taken from declared
 *   dims rather than realized allocations.
 *
 * @par Future improvements
 * - Replace the greedy last-consumer heuristic with a cost-model-driven schedule that minimizes true peak footprint (e.g.
 *   lookahead or a liveness-interval formulation) instead of a per-step local choice.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT Reorder : public OptimizerPass {
  public:
    APIARY_EXPOSE Reorder() = default;

    [[nodiscard]] std::string name() const override { return "Reorder"; }

    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /**
     * @brief Run the memory-aware reorder pass.
     * @param[in,out] graph The graph to reorder.
     * @return True if the node order changed.
     */
    bool run(Graph &graph) override;
};

} // namespace einsums::compute_graph::passes
