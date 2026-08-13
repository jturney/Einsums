//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

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
 * **Fold (live scale)** — the scaled tensor is overwritten later, so its scaled
 * value is observable only to the nodes in between. Every one of those nodes
 * takes the factor and the Scale is dropped. Two ways a node can take it:
 * - **Operand**: it reads the tensor as one operand. einsum and axpby are
 *   linear in their source, so `α` folds into `ab_prefactor` / `alpha`:
 *   Scale(α, A) + Einsum(ab_pf, …, A, B) + overwrite(A) → Einsum(ab_pf·α, …, A, B)
 * - **Accumulator**: it accumulates into the tensor and reads it only as that
 *   destination. `scale(α, C)` then `C = c_pf·C + …` is exactly
 *   `C = (c_pf·α)·C + …`, so `α` folds into `c_prefactor` / `beta`:
 *   Scale(α, C) + Einsum(c_pf≠0, C, …) → Einsum(c_pf·α, C, …)
 *
 * The fold is all-or-nothing across the window: if any observer cannot take the
 * factor the whole Scale is kept, because a partial fold would be wrong rather
 * than merely a missed optimization.
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
 * - Only **einsum** and **axpby** can take the factor. `gemm`, `permute` and
 *   `BatchedGemm` bake their prefactors into the executor closure, so folding
 *   into them is not desync-safe (they lack live shared params); a scale whose
 *   window contains one of them is kept.
 * - The tensor must be read as exactly ONE operand by each observer: the tensor
 *   used as both einsum operands, or accumulated into AND used as an operand by
 *   the same node, would need α² and is left un-folded.
 * - A scaled value that is never overwritten stays observable to the caller
 *   after execute, so its Scale is always kept.
 * - A scale whose buffer is also accessed through a VIEW while the scaled value
 *   is live is kept. A slice consumer's prefactor scales only its own span,
 *   where the scale covers the whole buffer, so the scale's effect on the
 *   regions no slice touches (padding, unused slots) is observable and nothing
 *   can stand in for it. This holds for the dead-scale route too: a later
 *   whole-tensor overwrite does not make the scale dead when a view write in
 *   between read the scaled bytes.
 * - Only ops with live shared params can take the factor at all, so the grouped
 *   ops (GroupedBatchedGemm, GroupedDot, GroupedAxpby) and every kind this pass
 *   does not enumerate veto rather than fold.
 * - Reads from inside a control-flow node's sub-graphs count (via
 *   @ref Graph::effective_io) but cannot be folded into, so a loop body reading
 *   the scaled tensor keeps the Scale.
 *
 * @par Future improvements
 * - Fold into `permute` / `BatchedGemm` / `gemm` once they carry live prefactor
 *   params the way EinsumParams and AxpbyParams do (CG audit §5.1-3).
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ScaleAbsorption : public OptimizerPass {
  public:
    APIARY_EXPOSE ScaleAbsorption() = default;

    [[nodiscard]] std::string              name() const override { return "ScaleAbsorption"; }
    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

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

EINSUMS_NAMESPACE_END(compute_graph::passes)
