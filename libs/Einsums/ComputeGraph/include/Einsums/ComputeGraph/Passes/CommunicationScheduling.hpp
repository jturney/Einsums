//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Overlap communication with computation.
 *
 * Analogous to IOPrefetch for disk I/O. For Allreduce/Allgather nodes:
 * 1. Splits into iallreduce (async_start) + wait (async_finish)
 * 2. Moves independent computation between start and finish
 *
 * Uses the existing Node::async_start / async_finish mechanism that the
 * DataflowExecutor supports.
 */
class EINSUMS_EXPORT CommunicationScheduling : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CommunicationScheduling"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_scheduled() const { return _num_scheduled; }

  private:
    size_t _num_scheduled{0};
};

} // namespace einsums::compute_graph::passes
