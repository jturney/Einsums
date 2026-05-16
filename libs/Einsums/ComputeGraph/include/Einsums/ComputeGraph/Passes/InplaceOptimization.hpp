//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief In-place optimization pass (analysis only).
 *
 * Detects when an intermediate tensor is produced by one node and consumed
 * by exactly one subsequent node. These are candidates for in-place
 * optimization — the consumer could overwrite the intermediate's buffer
 * instead of using a separate output tensor.
 *
 * This pass reports candidates but does NOT modify the graph — actual
 * in-place execution requires changes to the BLAS dispatch layer.
 *
 * @par Criteria for in-place eligibility:
 * - Tensor is graph-owned intermediate (is_intermediate=true)
 * - Exactly one node writes to it (producer)
 * - Exactly one node reads from it (consumer)
 * - Tensor dimensions match between producer output and consumer input
 */
class EINSUMS_EXPORT InplaceOptimization : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "InplaceOptimization"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_candidates() const { return _num_candidates; }

  private:
    size_t _num_candidates{0};
};

} // namespace einsums::compute_graph::passes
