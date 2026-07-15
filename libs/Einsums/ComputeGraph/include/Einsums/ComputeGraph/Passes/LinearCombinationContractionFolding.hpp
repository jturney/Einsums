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
 * @brief Fold transpose-paired contractions into one (the CCSD "2J−K" idiom).
 *
 * Sibling of @ref DistributiveFactoring. Where DistributiveFactoring sums
 * *different* operands sharing the *same* index pattern, this pass folds
 * contractions that reuse the *same* operand tensor read with *permuted*
 * index patterns. Detects groups of einsum nodes that:
 *  1. accumulate into the same output (c_prefactor != 0),
 *  2. share one operand with an identical index pattern,
 *  3. read the *other* operand from the **same tensor** but with index
 *     patterns that are permutations of one another.
 *
 * Because einsum is linear in each operand,
 * @f$\sum_k \alpha_k\,\mathrm{einsum}(\text{spec}_k, A, B)
 *    = \mathrm{einsum}(\text{spec}_0, A,\; \sum_k \alpha_k\,P_k(B))@f$,
 * where @f$P_k@f$ permutes @f$B@f$ into operand-0's (canonical) layout. The
 * rewrite builds @f$L = \sum_k \alpha_k P_k(B)@f$ once and runs a single
 * contraction: trading @f$N@f$ flop-bound contractions for one contraction
 * plus @f$N-1@f$ cheap memory-bound permute/axpy steps.
 *
 * @par Example (CCSD spin-adaptation)
 * Before:
 * @code
 * // Fae[a,e] += 2 * t1[m,f] * ovvv[m,a,f,e]
 * // Fae[a,e] -= 1 * t1[m,f] * ovvv[m,a,e,f]
 * @endcode
 * After:
 * @code
 * // L[m,a,f,e] = 2*ovvv[m,a,f,e] - ovvv[m,a,e,f]   (permute + axpy)
 * // Fae[a,e]  += t1[m,f] * L[m,a,f,e]              (single einsum)
 * @endcode
 *
 * @par Opt-in
 * Not registered in @ref PassManager::create_default, like DistributiveFactoring
 * it is a workload-dependent rewrite. Apply it
 * explicitly:
 * @code
 * cg::PassManager pm; pm.add(std::make_shared<LinearCombinationContractionFolding>());
 * graph.apply(pm);
 * @endcode
 * The fold's @f$L@f$ build is loop-invariant when @f$B@f$ is a constant integral,
 * so a subsequent LoopInvariantHoisting can lift it out of an iteration loop.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LinearCombinationContractionFolding
    : public OptimizerPass {
  public:
    APIARY_EXPOSE LinearCombinationContractionFolding() = default;

    [[nodiscard]] std::string name() const override { return "LinearCombinationContractionFolding"; }

    bool run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local rewrite within the
    /// graph it is handed. See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of fold groups rewritten.
    [[nodiscard]] size_t num_groups() const { return _num_groups; }

    /// Total number of einsum nodes eliminated (folded away).
    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_groups{0};
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
