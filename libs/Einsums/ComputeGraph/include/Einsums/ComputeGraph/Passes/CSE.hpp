//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Common Subexpression Elimination (CSE): merges two nodes that compute the same value -- or one that is a fixed multiple
 *        of the other -- and redirects readers of the duplicate onto the survivor's output.
 *
 * Detects pairs of nodes with identical OpKind, identical input TensorIds (in order), the same number of outputs, and OpData
 * (EinsumDescriptor / AxpbyDescriptor / ScaleDescriptor / PermuteDescriptor / BatchedGemmDescriptor) that agrees on index
 * patterns, conj flags, and the destination prefactor. The later duplicate is deleted; every downstream reader of its output is
 * redirected -- in both the TensorId metadata (so liveness passes see the survivor's buffer as live) and via
 * `Graph::redirect_slot` (so any already-baked executor lambda reads the survivor's result at execute time). Redirections
 * propagate through all remaining nodes' input lists, so multi-level CSE resolves in a single pass.
 *
 * @par Proportional duplicates
 * The two nodes need not agree on the SOURCE prefactor, the one their result is linear in (einsum's `ab_prefactor`, axpby's
 * `alpha`, permute's and batched-gemm's `alpha`). When they differ by a factor `r` that is an exact power of two -- so `(r*x)*y`
 * and `x*(r*y)` agree bit for bit and no arithmetic changes -- the duplicate is eliminated and `r` is multiplied into every
 * reader's own prefactor instead. CCSD's tau and tau-tilde are the motivating shape: the same `t1 x t1` outer product differing
 * only by a 0.5. Ratios that are exactly representable but not powers of two are declined, so the result never depends on which
 * of two proportional nodes the pass happened to keep.
 *
 * @par Example (proportional)
 * @code
 * cg::einsum("ij <- ik ; kj", 0.0, &P, 1.0, A, B);      // survivor
 * cg::einsum("ij <- ik ; kj", 0.0, &Q, 0.5, A, B);      // Q == 0.5 * P
 * cg::einsum("ij <- ik ; kj", 0.0, &out, 1.0, Q, F);    // reader of the duplicate
 * // After CSE: Q's producer is gone and the reader's ab_prefactor is 0.5.
 * @endcode
 *
 * In the default pipeline (populate_default; also the O1 cleanup cluster in PassManager::create_for). Runs after PermuteFusion so
 * duplicate permute->einsum patterns collapse into the same fused node first, and before DeadNodeElimination so the now-dead
 * survivors of a redirect are swept.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("cse");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);   // C = A·B
 *     cg::einsum("ij <- ik ; kj", 0.0, &D, 1.0, A, B);   // D = A·B  -- identical subexpression
 *     cg::einsum("ij <- ij ; ij", 0.0, &E, 1.0, C, D);   // reads both
 * }
 * cg::PassManager pm; pm.add<cg::passes::CSE>();
 * graph.apply(pm);                                        // or PassManager::create_default()
 * // The second einsum is gone; E now reads C for both operands.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("cse")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C, A, B)           # C = A·B
 *     einsums.einsum("ij <- ik ; kj", D, A, B)           # D = A·B  -- duplicate
 * pm = cg.PassManager(); pm.add(cg.CSE())
 * g.apply(pm)                                            # or cg.default_pass_manager()
 * # CSE exposes no result counter; the duplicate node is simply removed from g.
 * @endcode
 *
 * @par Limitations
 * - Only **pure-overwrite** producers are eligible: einsum with `c_prefactor == 0`, axpby / permute / batched-gemm with
 *   `beta == 0`. Accumulating ops (nonzero destination prefactor) and scale/axpy/element-transform (whose scalar coefficients
 *   are not carried in `op_data`, so equality can't be decided) are never merged.
 * - The duplicate's outputs must be graph-owned intermediates that are **not** views (`is_intermediate && aliases == 0`); a
 *   user-visible output is left in place because the user reads that tensor directly, not through an executor slot.
 * - Every merged output buffer must have **exactly one writer** in the whole graph (Guard B), and the shared inputs must not be
 *   overwritten by any node between the two candidates (Guard A); either condition would make the reused value stale.
 * - Neither output may be a buffer a control-flow node's sub-graph reads or writes (Guard D). A Loop/Conditional node's own
 *   `inputs`/`outputs` do not list what its body touches (that is @ref Graph::effective_io), and `redirect_slot` repoints only
 *   this graph's slot table, so a body reading the eliminated duplicate's output would read a never-written buffer.
 * - Neither output may be visible outside the graph being rewritten (Guard F). `Graph::redirect_slot` repoints only that
 *   graph's slot table, so a reader in the parent, a sibling branch, or a nested body would keep reading the eliminated
 *   duplicate's now never-written buffer. Inside a sub-graph the SURVIVOR must additionally be graph-owned: a loop predicate or
 *   DIIS callback runs between iterations and can write user tensors from Python, which no node list describes.
 * - A proportional (`r != 1`) merge additionally needs EVERY reader of the duplicate's output to absorb `r` (Guard E): an einsum
 *   or axpby reading it as exactly one operand through live shared params. permute and batched-gemm bake their scalars into the
 *   executor closure, so a reader of either blocks the merge.
 * - Matching walks the node list once, bucketing candidates by `(kind, redirected inputs, output count)` and comparing each node
 *   only against earlier survivors in its own bucket. The bucket key deliberately excludes `op_data`, so two nodes differing
 *   only in a prefactor still meet.
 *
 * @par Future improvements
 * - Fold user-visible duplicates, and proportional duplicates whose readers cannot absorb the factor, behind an inserted scaled
 *   copy of the survivor. A copy is two sweeps over the result while an outer product is barely more than one, so this needs a
 *   cost model to decide (CostModel's permute/bandwidth estimate, roadmap 3.3) rather than always being a win.
 * - Recursing into sub-graphs is DONE: `run()` walks the tree itself. The SCF-body case that originally motivated the opt-out
 *   (`axpby(1,H,0,F)` and `axpby(1,H,0,sum_HF)` diverging afterwards) turns out to be ruled out by Guard B already, since both
 *   destinations are written twice.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT CSE : public OptimizerPass {
  public:
    APIARY_EXPOSE CSE() = default;

    [[nodiscard]] std::string name() const override { return "CSE"; }

    /// Opts out of the PassManager's recursion and descends ITSELF, which is
    /// not the same as not running on bodies - it does, since `run()` walks the
    /// tree.
    ///
    /// The driver hands a pass nothing but the sub-graph, and two guards
    /// cannot be decided from inside one: whether a buffer escapes the graph
    /// being rewritten (``Graph::redirect_slot`` repoints only that graph's
    /// slots, so an outside reader would be stranded), and whether a body's
    /// handle for a parent-created tensor describes a user tensor or graph
    /// scratch (the body's own handle says `is_intermediate == false` either
    /// way). Both are answerable from the root, so `run()` collects the tree
    /// once and passes it down.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /**
     * @brief Run the CSE pass on @p graph and every loop body / conditional
     *        branch beneath it.
     * @param[in,out] graph The graph to optimize.
     * @return True if at least one duplicate node was eliminated anywhere.
     */
    bool run(Graph &graph) override;

  private:
    /// One graph of the tree. @p tree_context is an opaque handle to the
    /// root-level facts collected by @ref run (type-erased to keep the tree
    /// bookkeeping out of this header). @p is_subgraph tightens Guard C: inside
    /// a body the survivor must be graph-owned too, because a loop predicate or
    /// DIIS callback runs between iterations and can write user tensors.
    bool run_on_graph(Graph &graph, void const *tree_context, bool is_subgraph);
};

} // namespace einsums::compute_graph::passes
