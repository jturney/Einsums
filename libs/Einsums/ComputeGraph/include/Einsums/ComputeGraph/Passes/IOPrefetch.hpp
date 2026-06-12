//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Move DiskRead nodes as early as legally possible in the schedule.
 *
 * For each DiskRead node, computes its earliest legal position (immediately
 * after all its predecessors) and moves it there. This maximizes the window
 * between a read's async_start and its first consumer, enabling maximum
 * I/O-compute overlap when used with the DataflowExecutor.
 *
 * Should run after Reorder (which optimizes for memory) so that IOPrefetch
 * can pull reads earlier without violating dependency constraints.
 *
 * DiskWrite nodes are NOT moved, writes should happen as late as possible
 * to avoid blocking compute on I/O completion.
 */
class EINSUMS_EXPORT IOPrefetch : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "IOPrefetch"; }
    bool                      run(Graph &graph) override;

    /// Number of DiskRead nodes moved in the last run.
    [[nodiscard]] size_t num_prefetched() const { return _num_prefetched; }

  private:
    size_t _num_prefetched{0};
};

} // namespace einsums::compute_graph::passes
