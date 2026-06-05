//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Automatically slice pre-allocated inputs for distributed einsums.
 *
 * When an einsum has a distributed output (e.g., C distributed along dim 0),
 * any pre-allocated input that shares the same index dimension needs to be
 * temporarily "viewed" as a local slice for each rank.
 *
 * This pass inserts BeginSlice / EndSlice nodes around the einsum that
 * modify the input tensor's data pointer and dimensions to present only
 * the local partition. After the einsum, the original state is restored.
 *
 * Uses the EinsumDescriptor's ContractionSpec to determine which input
 * dimensions correspond to the distributed output dimension.
 *
 * Runs AFTER DistributionPlanning and Materialization.
 */
class EINSUMS_EXPORT InputSlicing : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "InputSlicing"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_sliced() const { return _num_sliced; }

  private:
    size_t _num_sliced{0};
};

} // namespace einsums::compute_graph::passes
