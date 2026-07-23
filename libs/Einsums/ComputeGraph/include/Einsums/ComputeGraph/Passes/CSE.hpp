//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Common Subexpression Elimination (CSE): merges two nodes that compute the same value and redirects readers of the
 *        duplicate onto the survivor's output.
 *
 * Detects pairs of nodes with identical OpKind, identical input TensorIds (in order), identical OpData
 * (EinsumDescriptor / ScaleDescriptor / PermuteDescriptor / BatchedGemmDescriptor prefactors, index patterns, conj flags), and
 * the same number of outputs. The later duplicate is deleted; every downstream reader of its output is redirected -- in both the
 * TensorId metadata (so liveness passes see the survivor's buffer as live) and via `Graph::redirect_slot` (so any already-baked
 * executor lambda reads the survivor's result at execute time). Redirections propagate through all remaining nodes' input lists,
 * so multi-level CSE resolves in a single pass.
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
 * - Only **pure-overwrite** producers are eligible: einsum with `c_prefactor == 0`, permute with `beta == 0`, or batched-gemm
 *   with `beta == 0`. Accumulating ops (nonzero destination prefactor) and scale/axpy/axpby/element-transform (whose scalar
 *   coefficients are not carried in `op_data`, so equality can't be decided) are never merged.
 * - The duplicate's outputs must be graph-owned intermediates that are **not** views (`is_intermediate && aliases == 0`); a
 *   user-visible output is left in place because the user reads that tensor directly, not through an executor slot.
 * - Every merged output buffer must have **exactly one writer** in the whole graph (Guard B), and the shared inputs must not be
 *   overwritten by any node between the two candidates (Guard A); either condition would make the reused value stale.
 * - Opt-out of PassManager auto-recursion (`recurse_into_subgraphs() == false`): never runs on loop bodies / conditional
 *   branches, because redirecting a duplicate whose output is later written independently is unsound (the SCF-body case).
 * - Matching is an O(n^2) pairwise scan over the node list.
 *
 * @par Future improvements
 * - Fold user-visible duplicates behind an inserted copy node instead of skipping them (see Guard C).
 * - Add a write-once precondition on the merged output so CSE can safely recurse into loop bodies / conditional branches.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT CSE : public OptimizerPass {
  public:
    APIARY_EXPOSE CSE() = default;

    [[nodiscard]] std::string name() const override { return "CSE"; }

    /// NOTE: stays opt-out. CSE has a latent soundness gap with
    /// mutable-tensor reuse: it merges two nodes that compute the same
    /// value into different output tensors and redirects readers of the
    /// duplicate's output to the original's, which is wrong when that
    /// output is subsequently written independently (e.g. the SCF body's
    /// ``axpby(1,H,0,F)`` and ``axpby(1,H,0,sum_HF)``, where F and sum_HF
    /// then diverge). Recursing exposed this on the SCF example. Until CSE
    /// gains a write-once precondition on the merged output it must not run
    /// on bodies.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /**
     * @brief Run the CSE pass on the graph.
     * @param[in,out] graph The graph to optimize.
     * @return True if at least one duplicate node was eliminated.
     */
    bool run(Graph &graph) override;
};

} // namespace einsums::compute_graph::passes
