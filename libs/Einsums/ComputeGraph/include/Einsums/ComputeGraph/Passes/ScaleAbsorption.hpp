//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <utility>
#include <vector>

namespace einsums::compute_graph::passes {

/**
 * @brief Scale-elimination pass: removes a Scale by deleting it (dead) or
 *        folding its factor into a consumer's prefactor (live).
 *
 * A `scale` is in-place (`C *= α`), so its live range ends at the tensor's next
 * writer. It is eliminated in two ways:
 *
 * **Dead-scale removal** — the next node writing the tensor overwrites it
 * without reading its prior contents (c_prefactor / beta == 0) and nothing reads
 * it in between, so the scale's result is discarded wholesale:
 * - **Einsum**: Scale(α) + Einsum(c_pf=0) → Einsum(c_pf=0)
 * - **BatchedGemm**: Scale(α) + BatchedGemm(beta=0) → BatchedGemm(beta=0)
 * - **Permute**: Scale(α) + Permute(beta=0) → Permute(beta=0)
 *
 * **Operand fold (live scale)** — the scaled tensor is read by exactly one
 * einsum as a single operand and is then overwritten (so its scaled value is
 * dead everywhere else). einsum is linear in each operand, so `α` folds into
 * that einsum's `ab_prefactor` and the Scale is dropped:
 * - Scale(α, A) + Einsum(ab_pf, …, A, B) + overwrite(A) → Einsum(ab_pf·α, …, A, B)
 *
 * The fold is written to BOTH the live EinsumParams (read by the CPU executor)
 * and the descriptor snapshot (read by GPU dispatch / later passes); a
 * descriptor-only edit would desync them — the reason only ops with live shared
 * params are folded into. Because the fold removes a writer that the einsum
 * read, it deliberately flips that read to the tensor's initial contents; the
 * pass declares the `(einsum, tensor)` pair via @ref compensated_reads so the
 * program-order validator waives its structural guard for exactly that read.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("scale_absorption");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::scale(3.0, &A);                                 // A *= 3
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);    // C = 3·(A·B)  -- A is the sole reader
 *     cg::einsum("ik <- i ; k", 0.0, &A, 1.0, x, y);      // A overwritten (its scaled value is dead)
 * }
 * graph.apply(cg::PassManager::create_default());         // ScaleAbsorption runs here
 * // The Scale is gone and the first einsum's ab_prefactor is now 3.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("scale_absorption")
 * with cg.capture(g):
 *     einsums.linalg.scale(3.0, A)                        # A *= 3
 *     einsums.einsum("ij <- ik ; kj", C, A, B)            # C = 3·(A·B)
 *     einsums.einsum("ik <- i ; k", A, x, y)              # A overwritten
 * sa = cg.ScaleAbsorption()
 * pm = cg.PassManager(); pm.add(sa)
 * g.apply(pm)                                             # or cg.default_pass_manager()
 * # sa.num_absorbed  -> 1   (getters are exposed as properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - Operand folding targets only **einsum** (folds into `ab_prefactor`). A scale
 *   feeding a live gemm/axpby/permute/BatchedGemm operand is kept — `permute`
 *   and `BatchedGemm` bake their prefactors into the executor closure, so
 *   folding into them is not desync-safe (they lack live shared params).
 * - Operand folding requires the scaled tensor to be read by exactly ONE op as
 *   ONE operand AND overwritten afterward: two readers, the tensor used as both
 *   einsum operands (α² not α), or a scaled value that stays live (never
 *   overwritten, e.g. a graph output) are left un-folded.
 * - **Accumulator** scales — `scale(α, C)` before an *accumulating* write
 *   (c_pf/beta != 0) — are not yet folded into the accumulate prefactor.
 *
 * @par Future improvements
 * - Extend operand folding to axpby (fold into `alpha`, via AxpbyDescriptor's
 *   live params) and gemm; and to `permute`/`BatchedGemm` once they carry live
 *   prefactor params (CG audit §5.1-3).
 * - Accumulator folding: `scale(α, C)` + accumulating consumer → fold α into the
 *   consumer's c_prefactor / beta.
 * - Fold into ALL readers when a scaled tensor has several foldable consumers,
 *   instead of bailing on the multi-reader case.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ScaleAbsorption : public OptimizerPass {
  public:
    APIARY_EXPOSE ScaleAbsorption() = default;

    [[nodiscard]] std::string name() const override { return "ScaleAbsorption"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local rewrite within the
    /// graph it is handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// The einsum reads whose writer (a folded-away scale) was removed while the
    /// einsum's ab_prefactor was compensated. See OptimizerPass::compensated_reads.
    [[nodiscard]] std::vector<std::pair<NodeId, TensorId>> compensated_reads() const override { return _compensated; }

    /// Number of scale nodes eliminated in the last run (dead-removed or folded).
    APIARY_EXPOSE APIARY_GETTER("num_absorbed") [[nodiscard]] size_t num_absorbed() const { return _num_absorbed; }

  private:
    size_t                                   _num_absorbed{0};
    std::vector<std::pair<NodeId, TensorId>> _compensated;
};

} // namespace einsums::compute_graph::passes
