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
 * dispatch added in Phase 2).
 *
 * @par Current rules
 * - **Scale**: ``C = α·A`` inherits A's descriptor (scaling preserves symmetry).
 * - **Axpy / Axpby / sum of same-descriptor inputs**: ``C = α·A + β·B`` where A and B share a descriptor inherits that descriptor.
 * - **Permute / Transpose** of a rank-2 symmetric tensor with the natural swap-permutation remains symmetric.
 * - **Einsum ``ki,kj->ij`` (pattern ``AᵀA``)**: the product of A with its own transpose is always symmetric (for real) / Hermitian (for
 * complex) regardless of A.
 *
 * Rules are deliberately conservative — only cases that hold unconditionally
 * get tagged. Everything else is left untagged, which is the safe default.
 *
 * @par Scope
 * Only tags graph-owned intermediates (``is_intermediate == true``).
 * User-owned tensors are never mutated by this pass.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT SymmetryPropagation : public OptimizerPass {
  public:
    APIARY_EXPOSE SymmetryPropagation() = default;

    [[nodiscard]] std::string name() const override { return "SymmetryPropagation"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. Safe: the pass only
    /// reads op structure (no execution, no node changes) and tags a tensor
    /// only when its symmetry is *guaranteed* — single-writer in this graph
    /// and not written by a child sub-graph (see the InferGuard in run()).
    /// So a body intermediate proven symmetric by a self-contraction gets
    /// tagged per iteration; anything that could be invalidated is left
    /// untagged. See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of tensor handles this pass tagged with an inferred descriptor.
    APIARY_EXPOSE APIARY_GETTER("num_inferred") [[nodiscard]] std::size_t num_inferred() const { return _num_inferred; }

  private:
    std::size_t _num_inferred{0};
};

} // namespace einsums::compute_graph::passes
