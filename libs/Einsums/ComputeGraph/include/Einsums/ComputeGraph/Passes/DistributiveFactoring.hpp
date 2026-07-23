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
 * // T[k,a] = B1[k,a] + B2[k,a] + B3[k,a]  (axpy chain)
 * // R[i,a] += A[i,k] * T[k,a]               (single einsum)
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
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT DistributiveFactoring : public OptimizerPass {
  public:
    APIARY_EXPOSE DistributiveFactoring() = default;

    [[nodiscard]] std::string name() const override { return "DistributiveFactoring"; }

    bool run(Graph &graph) override;

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
