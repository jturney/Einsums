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
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("lccf")
 * with cg.capture(g):
 *     # Same operand tensor B read with permuted index patterns, accumulating into Fae.
 *     einsums.einsum("ae <- mf ; mafe", Fae, t1, ovvv, c_pf=0.0, ab_pf=2.0)   # seed
 *     einsums.einsum("ae <- mf ; maef", Fae, t1, ovvv, c_pf=1.0, ab_pf=-1.0)  # accumulate
 * p = cg.LinearCombinationContractionFolding()
 * pm = cg.PassManager(); pm.add(p)
 * g.apply(pm)
 * # p.num_groups -> 1, p.num_eliminated -> 1  (getters are properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - Members must be **einsum** nodes with two inputs and one output, sharing one
 *   operand (same tensor id and index pattern) and reading the *other* operand
 *   from the **same** tensor with index patterns that are genuine permutations of
 *   one another (at least one member permuted, else it is duplicate-contraction
 *   territory, not this pass).
 * - Every non-first member must purely accumulate (`c_prefactor == 1`) so the
 *   reassociation is exact; the first member may seed (overwrite or any prefactor).
 *   A stray overwrite in the tail rejects the group.
 * - Conjugated einsums (`conj_a`/`conj_b`) are skipped; conjugation is not
 *   threaded through the rewrite (they still execute correctly, just unfolded).
 * - Output, shared and non-shared operands must all be **runtime** tensors of one
 *   dtype; typed `Tensor<T,Rank>` captures or a mixed-dtype triple decline (a blind
 *   cast in the fused kernel would be type confusion / a segfault).
 * - On a real dtype, every member's `ab_prefactor` and node-0's `c_prefactor` must
 *   be real-valued; complex prefactors fold only on complex dtypes.
 * - Interference guard: between the first and last member, no other node may
 *   read/write the output or write the shared / non-shared operand; the fused node
 *   occupies node-0's slot (never appended) to stay scan-before its consumer.
 *
 * @par Future improvements
 * - Make the fold conj-aware (thread `conj_a`/`conj_b` through the @f$L@f$ build) if
 *   the conjugated variant shows up hot (see the `TODO` in the source).
 * - Extend to statically-typed `Tensor<T,Rank>` captures, not only runtime tensors.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT LinearCombinationContractionFolding
    : public OptimizerPass {
  public:
    APIARY_EXPOSE LinearCombinationContractionFolding() = default;

    [[nodiscard]] std::string name() const override { return "LinearCombinationContractionFolding"; }

    bool run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local rewrite within the
    /// graph it is handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of fold groups rewritten.
    APIARY_EXPOSE APIARY_GETTER("num_groups") [[nodiscard]] size_t num_groups() const { return _num_groups; }

    /// Total number of einsum nodes eliminated (folded away).
    APIARY_EXPOSE APIARY_GETTER("num_eliminated") [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_groups{0};
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
