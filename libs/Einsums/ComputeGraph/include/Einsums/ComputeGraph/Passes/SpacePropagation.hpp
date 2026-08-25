//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Infer per-slot index spaces on graph intermediates.
 *
 * Walks the graph in topological order and, for each node that produces a tensor, works out what
 * each output slot ranges over from the annotations its operands carry. When every slot resolves,
 * the output's @ref TensorHandle::spaces is filled in and marked
 * @ref TensorHandle::spaces_inferred. That is what keeps annotation effort proportional to a
 * program's INPUTS rather than to its node count: a method annotates its handful of persistent
 * tensors once, and every intermediate derived from them gets its spaces for free.
 *
 * @par Current rules
 * - **Einsum**: the output slot takes the space of the letter that produced it, from the letters
 *   bound by the two input operands' slots.
 * - **Scale**: ``C = α·A`` inherits A's spaces (a scale changes values, never what an axis ranges
 *   over).
 * - **Permute / Transpose**: the output slots take the input's spaces reordered by the node's own
 *   index letters, so ``ji <- ij`` on an ``(occ, virt)`` tensor yields ``(virt, occ)``.
 * - **Axpby**: the output takes the spaces every annotated input agrees on, slot by slot.
 *
 * @par Scope
 * Only annotates graph-owned intermediates (``is_intermediate == true``). A user-owned tensor is
 * never written to, and an unannotated INPUT never has a space pushed back onto it: inheritance
 * across operands is deliberately off, because it is precisely how one wrong annotation spreads
 * silently through the region a validation pass exists to police. Propagation flows from producer
 * to output and nowhere else.
 *
 * An annotation the user DECLARED is never overwritten. One this pass or capture inferred may be
 * refined, and re-inferring the same value is a no-op, so repeated runs converge.
 *
 * In the default pipeline it sits beside SymmetryPropagation, after Materialization and before GPU
 * placement. The two are independent analyses of the same shape, and neither reads the other's
 * output.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("space_propagation");
 * auto     &T = graph.create_zero_tensor<double, 2>("T", nocc, naux);
 * auto     &U = graph.create_zero_tensor<double, 2>("U", nocc, nocc);
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ix <- ia ; ax", &T, A, B);   // T is (occ, aux)
 *     cg::einsum("ij <- ix ; jx", &U, T, T);   // U is (occ, occ), through T
 * }
 * graph.annotate_spaces(A, {occ, virt});       // declarations may arrive after capture
 * graph.annotate_spaces(B, {virt, aux});
 * auto [modified, pass] = graph.apply<cg::passes::SpacePropagation>();
 * // pass.num_inferred() == 2: T from A and B, then U from T, in one topological sweep.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * cg.annotate(A, ("occ", "virt"), graph=g)
 * cg.annotate(B, ("virt", "aux"), graph=g)
 * prop = cg.SpacePropagation()
 * pm = cg.PassManager()
 * pm.add(prop)
 * pm.run(g)
 * assert prop.num_inferred == 2
 * @endcode
 *
 * @par Limitations
 * - Analysis-only: the pass never changes the node list, it only annotates @ref TensorHandle
 *   objects. It always reports "not modified".
 * - Annotates only graph-owned intermediates; user-owned tensors are never mutated.
 * - Soundness guard: a tensor is annotated only when it has exactly one writer in this graph and
 *   is not referenced by a child sub-graph, the same rule SymmetryPropagation uses. A second
 *   writer could bind the slot to something else, and a nested loop body's writes are invisible in
 *   this graph's node list.
 * - All-or-nothing per tensor: a rule that cannot resolve EVERY output slot writes nothing, since
 *   a half-filled annotation is indistinguishable from a deliberately partial one.
 * - Operands that disagree about a letter are declined and counted in @ref skip_reasons rather
 *   than reported. A cross-space conflict is a diagnosis for a validation pass to deliver, and
 *   this pass has no business failing a pipeline over it.
 * - Only the four rules above fire. Every other op kind, including Gemm, BatchedGemm and the
 *   grouped ops, leaves its output unannotated.
 *
 * @par Future improvements
 * - Cover the remaining structure-preserving kinds: Gemm and Gemv, the grouped and batched forms,
 *   element-wise products and quotients, and View, whose slots are its parent's.
 * - Optional inheritance onto unannotated inputs, off by default, for a partially annotated
 *   program whose author accepts the weaker verdict that then follows.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT SpacePropagation : public OptimizerPass {
  public:
    APIARY_EXPOSE SpacePropagation() = default;

    /// @copydoc OptimizerPass::name
    [[nodiscard]] std::string name() const override { return "SpacePropagation"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::Analysis; }

    /// @copydoc OptimizerPass::run
    bool run(Graph &graph) override;

    /// @copydoc OptimizerPass::reset_stats
    void reset_stats() override;

    /// @copydoc OptimizerPass::explain
    [[nodiscard]] std::vector<std::string> explain() const override;

    /// Recurse into loop bodies / conditional branches.
    ///
    /// Safe: the pass only reads op structure (no execution, no node changes) and annotates a
    /// tensor only when nothing can rebind its slots afterwards, single-writer in this graph and
    /// not referenced by a child sub-graph (see the InferGuard in run()). A body intermediate
    /// whose spaces follow from the body's own operands is therefore annotated per iteration, and
    /// anything a hidden write could contradict is left alone.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of tensor handles this pass annotated with inferred spaces.
    APIARY_EXPOSE APIARY_GETTER("num_inferred") [[nodiscard]] std::size_t num_inferred() const { return _num_inferred; }

  private:
    std::size_t _num_inferred{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
