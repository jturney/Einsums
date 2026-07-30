//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace einsums::compute_graph::passes {

/// Whether to let the cost model decide that a group is worth factoring.
enum class APIARY_EXPOSE APIARY_MODULE("graph") Factor : std::uint8_t{
    Auto,   ///< Price the trade against the @ref CostModel. The default.
    Always, ///< Factor whenever the rewrite is structurally possible, whatever it
            ///< costs. For tests that exercise the rewrite on shapes too small to
            ///< be worth it, and for callers who know their workload better than
            ///< the model does.
};

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
 * Priced, not guessed. The trade is (N-1) contractions saved against one axpy
 * chain over the summed operands plus a buffer, so it pays when the contraction
 * is flop-bound and loses when it is bandwidth-bound: a cheap contraction over
 * large operands spends more assembling the sum than it saves, and pays memory
 * for the privilege. Both sides are estimated in microseconds from the shared
 * @ref CostModel, the same way TiledExpansion chooses between its lowerings, and
 * @ref num_unprofitable counts the groups declined on those grounds.
 *
 * Note the direction of the memory effect: factoring ADDS a buffer that the
 * unfactored form does not need. Sharing reduces how many it adds when several
 * consumers want one sum; it does not make the pass a memory saving.
 *
 * @par In the default pipeline
 * Registered by ``PassManager::populate_default()``, next to
 * @ref LinearCombinationContractionFolding and before
 * @ref LoopInvariantHoisting, so a loop-invariant sum is built once rather than
 * every replay. Because it self-gates on cost, it is a no-op on graphs it cannot
 * help. To exercise it alone, or to price it against a chosen profile:
 * @code
 * auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
 * auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(my_cost_model);
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
 *
 * @par Future improvements
 * - Thread `conj_a`/`conj_b` and complex prefactors through the rewrite so
 *   conjugated / complex-scaled contractions can also be factored.
 * - Fold the zeroing Scale into the first Axpy once an Axpby node factory exists,
 *   saving a node per group.
 * - Offer sums across graph levels, so a loop-invariant one built in a CC
 *   iteration can be shared with, or hoisted above, the enclosing graph.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT DistributiveFactoring : public OptimizerPass {
  public:
    APIARY_EXPOSE DistributiveFactoring();

    /// Override the profitability decision. @ref Factor::Always skips it entirely.
    APIARY_EXPOSE explicit DistributiveFactoring(Factor factor);

    /// Price the trade against a supplied profile instead of the detected one.
    /// populate_default() passes the profile it shares with the other cost-model
    /// passes, so every planning decision is made against one machine.
    explicit DistributiveFactoring(CostModel cost_model, Factor factor = Factor::Auto);

    [[nodiscard]] std::string name() const override { return "DistributiveFactoring"; }

    bool run(Graph &graph) override;

    [[nodiscard]] std::vector<std::string> explain() const override;

    void reset_stats() override;

    /// Manages its own descent (like LoopInvariantHoisting): run() resets the
    /// counters once at the root and recurses into loop bodies / conditional
    /// branches itself. Opting into PassManager auto-recursion would re-invoke
    /// run() per body and reset (clobber) the top-level tally each time.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    /// Groups declined because the axpy chain would cost more than the
    /// contractions it saves. A bandwidth-bound contraction over large operands
    /// is the shape that lands here.
    APIARY_EXPOSE APIARY_GETTER("num_unprofitable") [[nodiscard]] size_t num_unprofitable() const { return _num_unprofitable; }

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
    size_t                      _num_unprofitable{0};
    size_t                      _num_eliminated{0};
    std::vector<FactoringGroup> _groups;
    CostModel                   _cost_model;
    Factor                      _factor{Factor::Auto};
};

} // namespace einsums::compute_graph::passes
