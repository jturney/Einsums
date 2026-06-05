//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Element-wise operation fusion pass.
 *
 * Detects chains of element-wise operations on the same tensor
 * (Scale + Scale, Scale + ElementTransform, etc.) and merges them
 * into a single node with a composite executor.
 *
 * @par Example
 * @code
 * // Before:
 * scale(2.0, &C);
 * scale(3.0, &C);
 *
 * // After fusion:
 * scale(6.0, &C);  // Single operation, factor = 2.0 * 3.0
 * @endcode
 *
 * Currently fuses consecutive Scale operations on the same tensor
 * by multiplying their factors.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT ElementWiseFusion : public OptimizerPass {
  public:
    APIARY_EXPOSE ElementWiseFusion() = default;

    [[nodiscard]] std::string name() const override { return "ElementWiseFusion"; }
    bool                      run(Graph &graph) override;

    /// Safe on loop bodies / conditional branches: a local fusion of
    /// adjacent element-wise ops within the graph it's handed.
    /// See docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    APIARY_EXPOSE APIARY_GETTER("num_fused") [[nodiscard]] size_t num_fused() const { return _num_fused; }

  private:
    size_t _num_fused{0};
};

} // namespace einsums::compute_graph::passes
