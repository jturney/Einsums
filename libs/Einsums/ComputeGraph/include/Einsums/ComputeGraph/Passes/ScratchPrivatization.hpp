//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Break false dependencies from reused scratch buffers by renaming each reuse onto its own clone.
 *
 * Captured code that recycles one scratch tensor through many unrelated write→read episodes -- the CCSD idiom
 * ``for each term: tmp = contract(...); r2 += tmp`` -- serializes the whole graph: every overwrite of ``tmp``
 * carries a WAR edge from the previous episode's readers and a WAW edge from its writer, so operations that
 * share no data still execute strictly one after another and a parallel executor finds no width.
 *
 * The pass splits each such tensor's accesses into *generations* (a generation starts at every pure overwrite:
 * a node that writes the whole tensor without reading it) and renames the interior generations onto a bounded
 * pool of clones, round-robin. The LAST generation keeps the original tensor, so every value observable after
 * the graph (or one loop-body replay) completes is unchanged -- which is what makes the rewrite safe without
 * knowing who else holds the tensor: outside observers can only look between executions, and between
 * executions the original buffer holds exactly the value it always did.
 *
 * Clones are declared as graph-owned deferred intermediates (Materialization allocates them, FreeInsertion
 * and MemoryPlanning manage them like any other scratch), and the pool is capped -- more clones than the
 * machine has cores adds memory without adding parallelism.
 *
 * A tensor is left alone whenever renaming can't be proven safe locally:
 * - any access through a view (a partial write is not a generation boundary),
 * - any read before the first pure overwrite (the value is carried in from outside, e.g. across iterations),
 * - any touch by a control-flow or lifecycle node, or by a node kind the pass cannot rebuild,
 * - fewer than two generations (nothing to split).
 *
 * Renaming a node means rebuilding its executor: captured executors resolve operands through TensorSlot
 * pointers baked at capture time, so editing ``Node::inputs`` alone would change the schedule but not the
 * computation (the CSE baked-lambda lesson). Einsum nodes are rebuilt through ``Graph::make_einsum_node``;
 * Permute and Axpby nodes get executors that resolve their operands by TensorId at call time.
 *
 * In the default pipeline (after LoopInvariantHoisting, before ContractionPlanning): late enough that the
 * structural rewrites (SymmetrizedAccumulation, LCCF, hoisting) have settled, early enough that the planning
 * and memory passes see the renamed graph. Recursion into loop bodies is innermost-first and self-driven.
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("sp")
 * body = g.add_loop("iter", n, cont)
 * with cg.capture(body):
 *     einsums.einsum("ij <- ik ; kj", tmp, A1, B1); einsums.linalg.axpby(1.0, tmp, 1.0, acc)
 *     einsums.einsum("ij <- ik ; kj", tmp, A2, B2); einsums.linalg.axpby(1.0, tmp, 1.0, acc)
 * body.set_executor(cg.DataflowExecutor())   # BEFORE apply: the pass rewrites only executor-carrying graphs
 * g.apply(cg.default_pass_manager())         # the two contractions can now overlap
 * @endcode
 *
 * @par Limitations
 * - Only Einsum, Permute (single-input) and Axpby nodes are rebuilt; a generation touched by any other node
 *   kind disqualifies the tensor. Notably a SymmetrizedAccumulation-rewritten permute (two inputs) is skipped
 *   deliberately: its descriptor no longer matches its baked executor.
 * - Trusts declared node I/O, like every other pass; an executor touching a tensor it does not declare was
 *   already unsound under Reorder.
 * - No profitability model beyond the pool cap: with the sequential executor the rewrite only spends memory.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ScratchPrivatization : public OptimizerPass {
  public:
    APIARY_EXPOSE ScratchPrivatization() = default;

    /// @param max_copies Clone-pool cap per tensor; 0 means "number of hardware threads".
    APIARY_EXPOSE explicit ScratchPrivatization(size_t max_copies) : _max_copies(max_copies) {}

    /// By default the pass rewrites only graphs that carry an installed
    /// executor (Graph::set_executor): under the built-in sequential replay
    /// the clones cost cache locality and buy nothing, so a graph that will
    /// never run wide is left alone.
    ///
    /// Install the executor BEFORE apply().
    /// Setting this false privatizes unconditionally (tests, custom drivers).
    APIARY_EXPOSE void set_require_executor(bool require) { _require_executor = require; }

    [[nodiscard]] std::string name() const override { return "ScratchPrivatization"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase                phase() const override { return PassPhase::StructuralResource; }
    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    APIARY_EXPOSE APIARY_GETTER("num_tensors_privatized") [[nodiscard]] size_t num_tensors_privatized() const {
        return _num_tensors_privatized;
    }
    APIARY_EXPOSE APIARY_GETTER("num_copies_created") [[nodiscard]] size_t num_copies_created() const { return _num_copies_created; }
    APIARY_EXPOSE APIARY_GETTER("num_nodes_rebuilt") [[nodiscard]] size_t  num_nodes_rebuilt() const { return _num_nodes_rebuilt; }

  private:
    /// Innermost-first: privatize inside nested bodies/branches before this level.
    void run_recursive(Graph &graph);

    /// Privatize within one graph's node list. Appends to the counters and
    /// calls mark_sorted() when it rewrote anything (order unchanged, deps stale).
    void privatize_one_graph(Graph &graph);

    size_t _max_copies{0};          ///< 0 = hardware concurrency.
    bool   _require_executor{true}; ///< Rewrite only graphs with an installed executor.

    size_t _num_tensors_privatized{0};
    size_t _num_copies_created{0};
    size_t _num_nodes_rebuilt{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
