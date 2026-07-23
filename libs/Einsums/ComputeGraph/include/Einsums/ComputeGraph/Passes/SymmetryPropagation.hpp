//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Phase 3 of the symmetry plan: infer symmetry on graph intermediates.
 *
 * Walks the graph in topological order. For each node that produces a
 * tensor, checks whether the output can be proven symmetric / antisymmetric
 * / Hermitian from the inputs' declared descriptors, and if so tags the
 * output ``TensorHandle`` and pushes the descriptor back to the backing
 * tensor (so subsequent ``graph.execute()`` benefits from the rank-2 BLAS
 * dispatch in LinearAlgebra/SymmetryDispatch.hpp).
 *
 * @par Current rules
 * - **Scale**: ``C = α·A`` inherits A's descriptor (scaling preserves symmetry).
 * - **Axpy / Axpby / sum of same-descriptor inputs**: ``C = α·A + β·B`` where A and B share a descriptor inherits that descriptor.
 * - **Permute / Transpose** of a rank-2 symmetric tensor with the natural swap-permutation remains symmetric.
 * - **Einsum ``ki,kj->ij`` (pattern ``AᵀA``)**: the product of A with its own transpose is always symmetric (for real) / Hermitian (for
 * complex) regardless of A.
 *
 * Rules are deliberately conservative, only cases that hold unconditionally
 * get tagged. Everything else is left untagged, which is the safe default.
 *
 * @par Scope
 * Only tags graph-owned intermediates (``is_intermediate == true``).
 * User-owned tensors are never mutated by this pass.
 *
 * In the default pipeline (after Materialization, so the backing tensors exist,
 * and before GPU placement, so downstream passes and executions see the inferred
 * symmetry).
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("symmetry_propagation");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     // T = AᵀA is provably symmetric; when T is a graph-owned intermediate
 *     // (e.g. produced by ContractionPlanning or declare_tensor) the pass tags it.
 *     cg::einsum("ij <- ki ; kj", 0.0, &T, 1.0, A, A);   // T symmetric
 *     cg::einsum("il <- ij ; jl", 0.0, &C, 1.0, T, B);   // consumer sees the tag
 * }
 * graph.apply(cg::PassManager::create_default());        // SymmetryPropagation runs here
 * // T's handle is tagged symmetric, enabling the rank-2 BLAS dispatch at execute().
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("symmetry_propagation")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ki ; kj", T, A, A)   # T = AᵀA (symmetric intermediate)
 *     einsums.einsum("il <- ij ; jl", C, T, B)
 * sp = cg.SymmetryPropagation()
 * pm = cg.PassManager(); pm.add(sp)
 * g.apply(pm)                                    # or cg.default_pass_manager()
 * # sp.num_inferred -> number of tensors tagged (getter is a property, not a method)
 * @endcode
 *
 * @par Limitations
 * - Analysis-only: the pass never changes the node list. It only tags
 *   the @c TensorHandle objects and pushes descriptors to the backing tensors.
 * - Tags only graph-owned intermediates (``is_intermediate``); user-owned tensors
 *   are never mutated.
 * - Soundness guard: a tensor is tagged only when it has exactly one writer in this
 *   graph and is not referenced by a child sub-graph (a nested loop/conditional
 *   body could rewrite it invisibly), so nothing can invalidate the tag later.
 * - Only four rules fire, all unconditional: Scale (inherit), Axpy/Axpby whose two
 *   inputs carry **identical** descriptors, rank-2 Permute/Transpose with the swap
 *   permutation and a zero destination prefactor on a symmetric/Hermitian input, and the
 *   self-contraction einsum (same tensor in both operand slots, rank-2 x rank-2 ->
 *   rank-2 with exactly one link index).
 * - The permute rule is rank-2 only and propagates only symmetric/Hermitian pair
 *   descriptors (higher-rank descriptors can encode cross-axis invariants that do
 *   not carry through every permute).
 * - Everything not provably structured is left untagged (the safe default).
 *
 * @par Future improvements
 * - Add rules for more op kinds: general einsum symmetry inference, antisymmetric
 *   outputs, propagation through gemm / BatchedGemm nodes.
 * - Relax the rank-2-only permute rule to higher-rank symmetry descriptors once the
 *   cross-axis invariants are tracked.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT SymmetryPropagation : public OptimizerPass {
  public:
    APIARY_EXPOSE SymmetryPropagation() = default;

    [[nodiscard]] std::string name() const override { return "SymmetryPropagation"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Safe: the pass only
    /// reads op structure (no execution, no node changes) and tags a tensor
    /// only when its symmetry is guaranteed: single-writer in this graph
    /// and not written by a child sub-graph (see the InferGuard in run()).
    /// So a body intermediate proven symmetric by a self-contraction gets
    /// tagged per iteration; anything that could be invalidated is left
    /// untagged.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of tensor handles this pass tagged with an inferred descriptor.
    APIARY_EXPOSE APIARY_GETTER("num_inferred") [[nodiscard]] std::size_t num_inferred() const { return _num_inferred; }

  private:
    std::size_t _num_inferred{0};
};

} // namespace einsums::compute_graph::passes
