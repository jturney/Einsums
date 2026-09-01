//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Permute-into-einsum fusion: absorbs an axis-reordering Permute into the subscript of the Einsum that reads it.
 *
 * Eliminates one tensor-shaped data copy (and, on GPU, one kernel launch + device round-trip) per match by rewriting the
 * consumer einsum to index the pre-permute tensor directly. The rewrite mutates @ref EinsumIndices in place (captured by the
 * executor via `shared_ptr`, so it takes effect on the next `graph.execute()` without rebuilding the lambda), updates the
 * @ref EinsumDescriptor::spec snapshot so analysis passes stay consistent, redirects the permute-output slot at the source
 * tensor's storage, and removes the Permute node.
 *
 * In the default pipeline (populate_default; also the O1 cleanup cluster in PassManager::create_for). Ordered BEFORE CSE/DNE so
 * duplicate permute->einsum patterns collapse into the same fused node, and before Materialization / GPU placement so those
 * passes never allocate or place a tensor that is about to be removed.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("permute_fusion");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::permute("ji <- ij", &A_T, A);                  // physical transpose
 *     cg::einsum("ik <- ji ; jk", 0.0, &C, 1.0, A_T, B); // GEMM on the transposed copy
 * }
 * cg::PassManager pm; pm.add<cg::passes::PermuteFusion>();
 * graph.apply(pm);                                        // or PassManager::create_default()
 * // The permute is gone; the einsum now reads A directly with subscript "ij" on the A slot.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * import einsums._core.graph as _G
 * g = cg.Graph("permute_fusion")
 * with cg.capture(g):
 *     einsums.permute("ji <- ij", A_T, A)                # physical transpose
 *     einsums.einsum("ik <- ji ; jk", C, A_T, B)         # GEMM on the transposed copy
 * pf = _G.PermuteFusion()
 * pm = _G.PassManager(); pm.add(pf); pm.run(g)
 * # pf.num_rewrites -> 1   (getter is a property, not a method)
 * @endcode
 * It also runs as part of ``cg.default_pass_manager()``.
 *
 * @par Limitations
 * - The consumer must be an **Einsum** node and the permuted tensor must feed one of its first two operand slots (A or B); a
 *   permute feeding a third-or-later operand, or any non-einsum consumer, is not fused.
 * - The Permute must be a **pure axis reordering**: `alpha == 1 && beta == 0`, with `c_indices` a duplicate-free permutation of
 *   `a_indices`. Scaling, accumulation, or diagonal/sum index patterns disqualify it.
 * - The Permute's output must have **exactly one** consumer; a shared transposed temporary is left in place.
 * - The consumer einsum must carry populated `EinsumDescriptor::indices` shared state (the mutable-indices infrastructure); an
 *   einsum without it, or a slot whose rank disagrees with the permute output rank, is skipped defensively.
 * - The producer kind must be `Permute` or `Transpose`; the fusion never introduces a permute, only removes one.
 *
 * @par Future improvements
 * - Extend fusion to non-einsum consumers (e.g. gemm/BatchedGemm slots) and to permutes feeding operand positions beyond A/B.
 * - Handle a permute with multiple consumers by fusing into each reader instead of bailing on the shared-temporary case.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT PermuteFusion : public OptimizerPass {
  public:
    APIARY_EXPOSE PermuteFusion() = default;

    [[nodiscard]] std::string name() const override { return "PermuteFusion"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralAlgebraic; }

    /// @copydoc OptimizerPass::tier
    /// Exact in its reasoning and NOT bit-identical in practice: the fused form hands the vendor a transa flag where the
    /// unfused form hands it a transposed copy, and those are different kernels. Bit-identical on five CI legs and one ULP
    /// apart on Accelerate, which is why this is not the bitwise tier with a caveat.
    [[nodiscard]] PassTier                 tier() const override { return PassTier::ReAssociating; }
    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// Safe on loop bodies / conditional branches: a local fold of
    /// Permute→Einsum/Gemm pairs within the graph it's handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of Permute→Einsum/Gemm pairs detected this run (before safety filtering).
    APIARY_EXPOSE APIARY_GETTER("num_candidates") [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

    /// Number of candidates that passed safety checks and were actually rewritten.
    APIARY_EXPOSE APIARY_GETTER("num_rewrites") [[nodiscard]] size_t num_rewrites() const { return _num_rewrites; }

  private:
    size_t _num_candidates{0};
    size_t _num_rewrites{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
