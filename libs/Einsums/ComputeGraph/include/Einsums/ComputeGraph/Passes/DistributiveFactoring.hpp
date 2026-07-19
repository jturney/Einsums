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
 * auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
 * @endcode
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
