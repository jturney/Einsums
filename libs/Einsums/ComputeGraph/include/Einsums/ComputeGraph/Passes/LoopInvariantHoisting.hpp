//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Loop invariant hoisting: moves body operations whose inputs never change across iterations to run once before the loop.
 *
 * For each Loop node, identifies body operations whose every input is loop-invariant -- not written by any node anywhere in the
 * loop subtree (including nested conditional branches and inner loops), or produced by an already-hoisted node -- and lifts them
 * into the enclosing graph, wiring their outputs as inputs of the Loop so the scheduler keeps the writer->reader edge. Invariance
 * is transitive, so a producer chain (`T = XᵀH`, then `F = TX`) hoists together. Descent is innermost-first: an invariant lifted
 * from an inner loop into its enclosing body may be lifted again by the outer sweep, one level per composed single-level hoist.
 *
 * In the default pipeline (populate_default, after ElementWiseFusion). Not in the O1 cleanup cluster of PassManager::create_for.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("lih");
 * {
 *     auto &body = graph.add_loop("loop", niter, [](size_t it) { return it + 1 < niter; });
 *     cg::CaptureGuard const guard(body);
 *     cg::einsum("ij <- ki ; kj", 0.0, &T, 1.0, X, H);   // X, H never change -> invariant, sole writer of T
 *     cg::einsum("ij <- ik ; kj", 0.0, &F, 1.0, T, X);   // depends only on invariants -> invariant
 *     cg::axpy(1.0, F, &acc);                            // acc += F changes each iteration -> stays in loop
 * }
 * cg::PassManager pm; pm.add<cg::passes::LoopInvariantHoisting>();
 * graph.apply(pm);                                        // or PassManager::create_default()
 * // The T- and F-producing einsums now sit before the loop; only the acc update remains in the body.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("lih")
 * body = g.add_loop("loop", niter, lambda it: it + 1 < niter)
 * with cg.capture(body):
 *     einsums.einsum("ij <- ki ; kj", T, X, H)           # invariant, sole writer of T
 *     einsums.einsum("ij <- ik ; kj", F, T, X)           # invariant (transitive)
 *     einsums.linalg.axpy(1.0, F, acc)                    # acc += F varies -> stays in loop
 * lih = cg.LoopInvariantHoisting()
 * pm = cg.PassManager(); pm.add(lih)
 * g.apply(pm)                                                # or cg.default_pass_manager()
 * # lih.num_hoisted -> 2   (getter is a property, not a method)
 * @endcode
 *
 * @par Limitations
 * - Hoists out of **Loop bodies only**. Conditional branches are never descended and no node is moved across a branch boundary
 *   in either direction (lifting a branch-guarded op would run it unconditionally and change semantics).
 * - Never hoists a **self-modifying** node: the always-accumulating ops (scale/axpy/axpby/element-transform), any einsum/
 *   permute/batched-gemm with a nonzero destination prefactor, or any node that lists the same tensor as both input and output.
 * - A candidate's output must have **exactly one** value-writer in the loop subtree; if another node also writes it, or its
 *   single-writer status can't be proven, the hoist is refused (removing the per-iteration write would change which write wins).
 * - A producer whose output is read by an *earlier* body node is loop-carried through it and is not hoisted; reads by *later*
 *   body nodes are fine.
 * - **No cost model**: hoisting a cheap producer of a huge tensor extends that tensor's live range across the whole loop, which
 *   can cost more memory than the recomputation it saves. The pass hoists whenever it is legal, not whenever it is profitable.
 *
 * @par Future improvements
 * - Add a cost/benefit test (recomputation cost vs. the extended live range) so a large-tensor producer is hoisted only when it
 *   actually pays off, per the caveat in `hoist_one_level()`.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LoopInvariantHoisting : public OptimizerPass {
  public:
    APIARY_EXPOSE LoopInvariantHoisting() = default;

    [[nodiscard]] std::string name() const override { return "LoopInvariantHoisting"; }
    bool                      run(Graph &graph) override;

    APIARY_EXPOSE APIARY_GETTER("num_hoisted") [[nodiscard]] size_t num_hoisted() const { return _num_hoisted; }

  private:
    /// Innermost-first descent. Recurse into every Loop body that is a direct
    /// child of @p graph BEFORE hoisting at this level (Conditional branches are
    /// deliberately NOT descended, see the .cpp). A producer invariant across
    /// several nested loops is lifted one level per composed single-level hoist.
    void run_recursive(Graph &graph);

    /// Single-level driver: hoist invariant producers out of each Loop body that
    /// is a direct child of @p graph, up one level into @p graph. Accumulates
    /// into @ref _num_hoisted and re-sorts @p graph if it hoisted anything.
    void hoist_one_level(Graph &graph);

    size_t _num_hoisted{0};
};

} // namespace einsums::compute_graph::passes
