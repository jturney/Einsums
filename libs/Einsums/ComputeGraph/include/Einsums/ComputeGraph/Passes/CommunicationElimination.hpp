//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Remove redundant communication nodes.
 *
 * Analogous to TransferElimination for GPU passes. Removes:
 * - Back-to-back allreduce of the same tensor
 * - Broadcast of a tensor that's already replicated
 * - Allgather when all ranks already have the full data
 */
class EINSUMS_EXPORT CommunicationElimination : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CommunicationElimination"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
