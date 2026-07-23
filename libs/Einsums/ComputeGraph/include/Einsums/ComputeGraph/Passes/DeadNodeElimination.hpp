//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <unordered_set>

namespace einsums::compute_graph::passes {

/**
 * @brief Dead node elimination (DNE): removes producers whose every output is an unread graph-owned intermediate.
 *
 * A node is dead when ALL of its outputs are graph-owned intermediates (`is_intermediate == true`) that are not read by any node
 * in this graph, not referenced by any descendant sub-graph (a nested loop body / conditional branch), and not referenced by any
 * enclosing or sibling graph. Output tensors owned by the user are always considered live -- user code may read them directly --
 * so only unread intermediates are candidates. Outputs are resolved through view aliases to their owning buffer, so a write
 * through a view of a live tensor keeps its producer.
 *
 * Especially useful after CSE (or any pass that redirects consumers away from a node), which can strand a now-unread producer.
 *
 * In the default pipeline (populate_default; also the O1 cleanup cluster in PassManager::create_for). Runs right after CSE.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("dne");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     auto &Tmp = graph.create_tensor<double, 2>("Tmp", m, n);  // graph-owned intermediate
 *     cg::einsum("ij <- ik ; kj", 0.0, &Tmp, 1.0, A, B);       // writes Tmp -- but nothing reads it
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);         // C is the real (user-visible) result
 * }
 * cg::PassManager pm; pm.add<cg::passes::DeadNodeElimination>();
 * graph.apply(pm);                                             // or PassManager::create_default()
 * // The Tmp-producing einsum is removed; the C-producing einsum stays.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("dne")
 * tmp = g.create_tensor("Tmp", [m, n], dtype="float64")       # graph-owned intermediate, never read
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", tmp, A, B)              # dead: nothing consumes tmp
 *     einsums.einsum("ij <- ik ; kj", C, A, B)               # live
 * dne = cg.DeadNodeElimination()
 * pm = cg.PassManager(); pm.add(dne)
 * g.apply(pm)                                                 # or cg.default_pass_manager()
 * # dne.num_eliminated -> 1   (getter is a property, not a method)
 * @endcode
 *
 * @par Limitations
 * - Only **graph-owned intermediates** are ever removed. A dead write into a user-owned tensor is kept, because the user may
 *   read that storage after `execute()`.
 * - Control-flow nodes (Conditional, Loop) and memory nodes (Alloc, Free) are never eliminated, nor is a side-effect-only node
 *   that has no outputs (e.g. an in-place `scale`).
 * - Liveness is a **reachability** analysis: it removes a producer whose result nothing reads, but does not prune a live-but-
 *   redundant computation, nor perform partial-output DCE (a node is kept whole if any one output is live).
 * - Cross-graph liveness is threaded manually: `recurse_into_subgraphs() == false` and `run()` descends the sub-graph tree
 *   itself, passing each child the set of buffers referenced by its parent and every sibling sub-tree (a per-graph recursion
 *   would wrongly kill a body producer whose only reader sits in the parent after the loop or in a sibling body).
 *
 * @par Future improvements
 * - Partial-output elimination: rewrite a surviving node to stop producing its individually-dead outputs, rather than keeping
 *   the whole node whenever a single output is live.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT DeadNodeElimination : public OptimizerPass {
  public:
    APIARY_EXPOSE DeadNodeElimination() = default;

    [[nodiscard]] std::string name() const override { return "DeadNodeElimination"; }
    bool                      run(Graph &graph) override;

    /// Opt out of PassManager auto-recursion: run() descends the sub-graph
    /// tree itself. A body tensor is live if ANY enclosing or sibling graph
    /// reads it, a cross-graph fact the per-graph recursion cannot see (it
    /// would eliminate a body producer whose only reader sits in the parent
    /// after the loop, or in a sibling loop body). run() threads the external
    /// references down into each child instead.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /// Number of nodes eliminated in the last run.
    APIARY_EXPOSE APIARY_GETTER("num_eliminated") [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    /// Eliminate dead nodes in @p g, then descend into its sub-graphs. A
    /// node's output tensor is kept alive if its buffer pointer appears in
    /// @p external_refs (referenced by an enclosing or sibling graph).
    bool run_one(Graph &g, std::unordered_set<void const *> const &external_refs);

    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
