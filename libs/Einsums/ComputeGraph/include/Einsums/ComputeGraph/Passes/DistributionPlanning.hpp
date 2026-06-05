//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Decide distribution strategy for deferred tensors.
 *
 * For each tensor with AllocState::Deferred, decides:
 * - **Replicate** if the tensor is below the size threshold or only 1 rank
 * - **Block-distribute** along the largest dimension if the tensor is large
 *
 * Annotates TensorHandle with is_distributed and is_replicated fields.
 * The MaterializationPass (which runs after) reads these to decide whether
 * to allocate the full tensor or just the local partition.
 *
 * On single-rank (mock or MPI with 1 process), this pass is a no-op:
 * all tensors remain replicated at full size.
 *
 * Runs BEFORE MaterializationPass in create_default().
 */
class EINSUMS_EXPORT DistributionPlanning : public OptimizerPass {
  public:
    /// @param threshold Tensors smaller than this (bytes) are replicated.
    /// @param enable_summa If true (default), distribute link indices for SUMMA on square grids.
    explicit DistributionPlanning(size_t threshold = static_cast<size_t>(64 * 1024 * 1024), bool enable_summa = true);

    [[nodiscard]] std::string name() const override { return "DistributionPlanning"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_distributed() const { return _num_distributed; }
    [[nodiscard]] size_t num_replicated() const { return _num_replicated; }

  private:
    size_t _threshold;
    bool   _enable_summa;
    size_t _num_distributed{0};
    size_t _num_replicated{0};
};

} // namespace einsums::compute_graph::passes
