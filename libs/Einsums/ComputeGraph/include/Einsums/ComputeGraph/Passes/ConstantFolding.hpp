//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Constant folding: evaluates nodes whose inputs never change once, at pass time, and replaces them with no-ops.
 *
 * A node is foldable when every input tensor is "constant" -- no node in the graph writes it -- and the node is not a control-
 * flow / memory / I/O / communication / user-defined kind. Folding is transitive: once a node is folded its outputs become
 * constant too, so a chain of setup ops (building transformation, orthogonalization, or denominator matrices from fixed data)
 * collapses in one topological sweep. The folded node is executed immediately and its executor replaced with a no-op, so the
 * result is baked and skipped on every `graph.execute()`.
 *
 * Only graph-owned intermediates (`create_tensor()`) never written by any node are seeded as constant. User-owned tensors are
 * NOT assumed constant -- they may change between loop iterations or successive `execute()` calls -- which keeps the pass safe
 * for both one-shot graphs and loop bodies.
 *
 * In the default pipeline (populate_default, first pass; also the head of the O1 cleanup cluster in PassManager::create_for).
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("constant_folding");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     // A and B are graph-owned intermediates filled before capture and never written here.
 *     cg::einsum("ij <- ik ; kj", 0.0, &S, 1.0, A, B);   // constant -> evaluated once, becomes a no-op
 *     cg::scale(2.0, &S);                                 // reads S; after folding S is constant, so this folds too
 * }
 * cg::PassManager pm; pm.add<cg::passes::ConstantFolding>();
 * graph.apply(pm);                                        // or PassManager::create_default()
 * // Both nodes are no-ops after the pass; S holds its final value and replays for free.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("constant_folding")
 * # A and B are graph-owned intermediates, populated then never written in capture.
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", S, A, B)           # constant -> folded
 *     einsums.linalg.scale(2.0, S)                        # folds transitively
 * cf = cg.ConstantFolding()
 * pm = cg.PassManager(); pm.add(cf)
 * g.apply(pm)                                            # or cg.default_pass_manager()
 * # cf.num_folded -> 2   (getter is a property, not a method)
 * @endcode
 *
 * @par Limitations
 * - Constants are seeded only from graph-owned intermediates never written by a node; user-owned inputs are never treated as
 *   constant, so a graph fed entirely from user tensors folds nothing.
 * - A node is folded only when **all** of its tensors are already `Materialized` at pass time. ConstantFolding runs before the
 *   Materialization pass, so deferred (shell) tensors -- notably loop-body workspace -- are skipped rather than executed against
 *   unallocated storage.
 * - Never folds control-flow (Conditional, Loop), memory (Alloc, Free, Materialize, Initialize), I/O (DiskRead, DiskWrite),
 *   GPU transfer (HostToDevice, DeviceToHost), collective (Allreduce, Broadcast, Allgather, Scatter, Barrier), or Custom nodes.
 * - Folding has a **side effect**: it mutates tensor data during the pass, not at `execute()` time.
 *
 * @par Future improvements
 * - A second constant-folding phase after Materialization could fold chains whose intermediates are deferred until then and are
 *   currently skipped by the materialization guard.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ConstantFolding : public OptimizerPass {
  public:
    APIARY_EXPOSE ConstantFolding() = default;

    [[nodiscard]] std::string name() const override { return "ConstantFolding"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Safe now that
    /// run() only folds nodes whose tensors are materialized at pass time
    /// (see the materialization guard in run()): a loop body's deferred
    /// workspace tensors are simply skipped rather than executed against
    /// unallocated storage. Loop-carried tensors are never folded because
    /// they appear in ``written_tensors``.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of nodes folded in the last run.
    APIARY_EXPOSE APIARY_GETTER("num_folded") [[nodiscard]] size_t num_folded() const { return _num_folded; }

  private:
    size_t _num_folded{0};
};

} // namespace einsums::compute_graph::passes
