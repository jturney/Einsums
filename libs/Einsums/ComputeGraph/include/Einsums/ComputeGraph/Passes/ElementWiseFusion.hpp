//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Element-wise operation fusion: collapses consecutive in-place Scale ops on the same tensor into a single node.
 *
 * Detects a run of adjacent `Scale` nodes that all write the same tensor and merges them into the first: their factors are
 * multiplied and their executors composed into one closure. Each merge removes a node (and its per-replay executor call). The
 * forward scan stops at the first node that is not a same-tensor scale, so only a textually consecutive scale chain fuses.
 *
 * In the default pipeline (populate_default; also the O1 cleanup cluster in PassManager::create_for). Runs after
 * SymmetrizedAccumulation, which must fold its `r2 += s*(tmp + P(tmp))` idiom before this pass composes the two axpby into one
 * executor and hides the pattern.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("elementwise_fusion");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::scale(2.0, &C);   // C *= 2
 *     cg::scale(3.0, &C);   // C *= 3   -- adjacent scale on the same tensor
 * }
 * cg::PassManager pm; pm.add<cg::passes::ElementWiseFusion>();
 * graph.apply(pm);          // or PassManager::create_default()
 * // One scale node remains with factor 6.
 * @endcode
 *
 * @par Example (Python)
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("elementwise_fusion")
 * with cg.capture(g):
 *     einsums.linalg.scale(2.0, C)   # C *= 2
 *     einsums.linalg.scale(3.0, C)   # C *= 3
 * ewf = cg.ElementWiseFusion()
 * pm = cg.PassManager(); pm.add(ewf)
 * g.apply(pm)                        # or cg.default_pass_manager()
 * # ewf.num_fused -> 1   (getter is a property, not a method)
 * @endcode
 *
 * @par What it fuses
 * - **axpby chains**: `Y = a1·X + b1·Y` directly followed by `Y = a2·X + b2·Y` on the same pair becomes
 *   `Y = (a2 + b2·a1)·X + (b2·b1)·Y`. axpby reads its scalars from live shared params, so this is a real fusion - the result is
 *   ONE sweep over Y. If the composed `beta` is zero the node stops listing Y as an input, since it no longer reads it.
 * - **Scale-into-Scale** on the same tensor: the factors multiply.
 *
 * @par Limitations
 * - The Scale fusion removes a node but not a sweep: `ScaleDescriptor` carries a plain `double` with no live params, so the
 *   merged node composes the two executors instead of applying one combined factor. Giving Scale live params (as
 *   EinsumParams/AxpbyParams do) would make it a real fusion, and would let ScaleAbsorption fold into it as well.
 * - Both fusions need the participants to be DIRECTLY consecutive; the scan breaks on any intervening node, so ops separated by
 *   another node do not fuse even when nothing in between touches the tensor.
 * - axpby fusion requires the same source X and the same destination Y. Two different sources are a three-operand update, which
 *   no single axpby expresses. It also requires all four prefactors to share a PrefactorScalar alternative.
 * - `ElementTransform` is not composed: it holds an opaque callable, so two of them can only be chained, not merged.
 * - Merged scalars are plain floating-point arithmetic; no reassociation or overflow handling.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ElementWiseFusion : public OptimizerPass {
  public:
    APIARY_EXPOSE ElementWiseFusion() = default;

    [[nodiscard]] std::string              name() const override { return "ElementWiseFusion"; }
    bool                                   run(Graph &graph) override;
    [[nodiscard]] std::vector<std::string> explain() const override;
    void                                   reset_stats() override;

    /// Safe on loop bodies / conditional branches: a local fusion of
    /// adjacent element-wise ops within the graph it's handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    APIARY_EXPOSE APIARY_GETTER("num_fused") [[nodiscard]] size_t num_fused() const { return _num_fused; }

  private:
    size_t _num_fused{0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
