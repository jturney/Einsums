//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

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
 * @par Limitations
 * - Fuses **only consecutive Scale-into-Scale** pairs on the same output tensor, despite the general "element-wise" name; other
 *   element-wise kinds (ElementTransform, axpy/axpby chains) are not yet composed here.
 * - Each participating scale must have exactly one output and a `ScaleDescriptor`; the scan breaks on any intervening node that
 *   is not a same-target scale, so scales separated by another op do not fuse.
 * - The merged factor is a plain product; no numeric reassociation or overflow handling beyond floating-point multiplication.
 *
 * @par Future improvements
 * - Extend fusion to the other element-wise ops the class name promises (ElementTransform composition, axpy/axpby chains) into a
 *   single composite executor.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ElementWiseFusion : public OptimizerPass {
  public:
    APIARY_EXPOSE ElementWiseFusion() = default;

    [[nodiscard]] std::string name() const override { return "ElementWiseFusion"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local fusion of
    /// adjacent element-wise ops within the graph it's handed.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    APIARY_EXPOSE APIARY_GETTER("num_fused") [[nodiscard]] size_t num_fused() const { return _num_fused; }

  private:
    size_t _num_fused{0};
};

} // namespace einsums::compute_graph::passes
