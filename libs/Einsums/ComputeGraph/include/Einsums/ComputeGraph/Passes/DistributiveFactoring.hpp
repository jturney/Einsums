//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributive factoring pass: factors out shared operands from accumulating einsums.
 *
 * Detects groups of einsum nodes that:
 * 1. Accumulate into the same output tensor (c_prefactor != 0)
 * 2. Share one input operand with identical index pattern
 * 3. Have different other operands with identical index pattern and shape
 *
 * Rewrites by summing the non-shared operands into an intermediate, then
 * performing a single einsum with the shared operand and the sum.
 *
 * The lowering is ordinary nodes, not one fused node: a Scale that zeros the
 * intermediate, one Axpy per summed operand carrying that member's product
 * prefactor, and one Einsum with ``ab_pf = 1``. An earlier version fused all of
 * it into an ``OpKind::Custom`` node that swapped a slot pointer so the first
 * member's baked executor read the intermediate. That ran correctly but hid the
 * intermediate from every other pass, so nothing could share or hoist it.
 *
 * Groups that sum the same operands, up to an overall factor, share one
 * intermediate, so a quantity several terms consume is built once. This is the
 * shape of CCSD's tau, which feeds W_mnij and W_abef with a quarter and the T2
 * equation with a half: written out term by term, with no hint that tau exists,
 * the pass recovers it, builds it once, and puts each consumer's factor on that
 * consumer's contraction. Reuse lives here rather than in CSE because an
 * accumulation buffer has several writers, and CSE's single-writer guard is what
 * stops readers being redirected onto a buffer that is mutated again.
 *
 * @par Example (CCSD-like pattern)
 * Before:
 * @code
 * // R[i,a] += A[i,k] * B1[k,a]     (einsum node 1)
 * // R[i,a] += A[i,k] * B2[k,a]     (einsum node 2)
 * // R[i,a] += A[i,k] * B3[k,a]     (einsum node 3)
 * @endcode
 *
 * After:
 * @code
 * // T[k,a]  = 0                            (Scale by zero)
 * // T[k,a] += B1[k,a]                       (Axpy)
 * // T[k,a] += B2[k,a]                       (Axpy)
 * // T[k,a] += B3[k,a]                       (Axpy)
 * // R[i,a] += A[i,k] * T[k,a]               (single Einsum)
 * @endcode
 *
 * This reduces N contractions to 1 contraction + (N-1) additions, saving
 * significant FLOPs when the contraction is expensive relative to the addition.
 *
 * @par Profitability
 * Only applied when the FLOP savings exceed the cost of the intermediate
 * (memory + copy + additions). Currently uses a simple heuristic: factor
 * when there are 2+ factorable terms.
 *
 * @par Opt-in
 * This pass is **not** registered in ``PassManager::create_default()``.
 * Factoring is a workload-dependent tradeoff, it saves FLOPs only when
 * the shared operand's contraction is expensive relative to the extra
 * axpy chain, and it adds a temporary allocation. Opt in per graph when
 * you know the pattern helps:
 * @code
 * cg::PassManager pm; pm.add(std::make_shared<DistributiveFactoring>());
 * graph.apply(pm);
 * // or:  auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("distributive_factoring")
 * with cg.capture(g):
 *     einsums.einsum("ia <- ik ; ka", R, A, B1, c_pf=1.0)   # R += A * B1
 *     einsums.einsum("ia <- ik ; ka", R, A, B2, c_pf=1.0)   # R += A * B2
 * df = cg.DistributiveFactoring()
 * pm = cg.PassManager(); pm.add(df)
 * g.apply(pm)
 * # df.num_groups -> 1, df.num_eliminated -> 1  (getters are properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - Members must be **einsum** nodes with exactly two inputs and one output,
 *   accumulating (`c_prefactor != 0`) into the same output, sharing one operand
 *   (same tensor id and index pattern) with the non-shared operands sharing the
 *   other index pattern.
 * - Every member after the first must accumulate with `c_prefactor == 1`. The
 *   combined contraction applies the output prefactor once, so a member that
 *   rescales the partial sum its predecessors wrote cannot be folded:
 *   `R = 2*(R + A*B1) + A*B2` is not `R = 2*R + A*(B1 + B2)`. Such a group
 *   declines rather than silently computing the second form.
 * - Sharing covers sums that are equal or PROPORTIONAL: a consumer wanting the
 *   same sum at a different strength reuses the build and carries the ratio on
 *   its own ``ab_pf``, which is free. The ratio must be an exact power of two,
 *   because only then does scaling the assembled sum agree with scaling each term
 *   (``r*(a+b) == r*a + r*b``). Other ratios decline; lifting that is safe up to
 *   rounding but would make the answer depend on what the pass chose to share.
 * - Sums that are not proportional stay separate even when related. CCSD's tau and
 *   tau-tilde are ``T2 + P`` and ``T2 + P/2`` for the same ``P``, a ratio of 1 on
 *   one term and 1/2 on the others, so they are two tensors here. That matches
 *   practice: deriving one from the other saves a single pass over an o^2v^2
 *   buffer against contractions costing o^2v^4, and serialises two builds that are
 *   otherwise independent. Seeing them as one quantity at two coefficients is
 *   multi-term optimisation over a coefficient matrix, which nothing here does.
 * - Reuse only looks within one graph level. A sum built in a loop body is not
 *   offered to the enclosing graph or to a sibling branch.
 * - Non-shared operands must all be **distinct** tensors of identical shape and
 *   dtype; the shared operand may not alias any non-shared operand (the
 *   slot-redirect trick cannot separate two reads of one tensor).
 * - The factoring math is real-valued: `conj_a`/`conj_b` einsums are skipped, and
 *   a prefactor with a nonzero imaginary part declines the node.
 * - Every summed operand must be the same tensor kind (all runtime, or all
 *   compile-time) so the accumulator dispatches correctly - a mismatch rank-errors
 *   at execute.
 * - Placement/interference gate: the combined node takes the first member's slot,
 *   so no node between the first and last member may read/write the output or
 *   write a factor operand, and any `Loop`/`Conditional` in the span disqualifies
 *   the group.
 * - Profitability is a simple heuristic: factor whenever there are 2+ factorable
 *   terms, without weighing the contraction cost against the extra axpy chain.
 *
 * @par Future improvements
 * - Replace the "2+ terms" heuristic with a cost-model decision (compare the FLOPs
 *   saved against the axpy chain plus the intermediate allocation).
 * - Thread `conj_a`/`conj_b` and complex prefactors through the rewrite so
 *   conjugated / complex-scaled contractions can also be factored.
 * - Fold the zeroing Scale into the first Axpy once an Axpby node factory exists,
 *   saving a node per group.
 * - Offer sums across graph levels, so a loop-invariant one built in a CC
 *   iteration can be shared with, or hoisted above, the enclosing graph.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT DistributiveFactoring : public OptimizerPass {
  public:
    APIARY_EXPOSE DistributiveFactoring() = default;

    [[nodiscard]] std::string name() const override { return "DistributiveFactoring"; }

    bool run(Graph &graph) override;
    void reset_stats() override;

    /// Manages its own descent (like LoopInvariantHoisting): run() resets the
    /// counters once at the root and recurses into loop bodies / conditional
    /// branches itself. Opting into PassManager auto-recursion would re-invoke
    /// run() per body and reset (clobber) the top-level tally each time.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /// Number of factoring groups found.
    APIARY_EXPOSE APIARY_GETTER("num_groups") [[nodiscard]] size_t num_groups() const { return _num_groups; }

    /// Total number of einsum nodes eliminated.
    APIARY_EXPOSE APIARY_GETTER("num_eliminated") [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

    /// Description of each factoring group found.
    struct FactoringGroup {
        std::string              shared_tensor;  ///< Name of the shared input tensor
        std::string              output_tensor;  ///< Name of the accumulation output
        size_t                   num_terms{0};   ///< Number of terms factored together
        std::vector<std::string> summed_tensors; ///< Names of the non-shared tensors that were summed
    };

    [[nodiscard]] std::vector<FactoringGroup> const &groups() const { return _groups; }

  private:
    /// Recurse into loop bodies / conditional branches after factoring the
    /// current level. Counters accumulate across the whole tree (no reset).
    bool run_recursive(Graph &graph);

    /// Factor one graph in isolation (no descent). Returns true if it rewrote
    /// anything. Called by run_recursive per graph in the subgraph tree.
    bool factor_one_level(Graph &graph);

    size_t                      _num_groups{0};
    size_t                      _num_eliminated{0};
    std::vector<FactoringGroup> _groups;
};

} // namespace einsums::compute_graph::passes
