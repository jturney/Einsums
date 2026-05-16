//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Remove redundant GPU memory transfer nodes.
 *
 * Simulates execution in topological order, tracking per-tensor residency.
 * A transfer node is redundant if the tensor already has the required
 * residency:
 * - HostToDevice is redundant if residency is Device or Both
 * - DeviceToHost is redundant if residency is Host or Both
 *
 * Additionally, consecutive GPU nodes sharing a tensor don't need
 * intermediate D2H + H2D round-trips — the tensor stays on device.
 *
 * Uses Belady's optimal strategy: when the full schedule is known,
 * tensors whose next GPU use is furthest away are the best eviction
 * candidates. (Eviction insertion is a future extension when GPU memory
 * budget tracking is added.)
 *
 * This pass must run AFTER TransferInsertion.
 */
class EINSUMS_EXPORT TransferElimination : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "TransferElimination"; }
    bool                      run(Graph &graph) override;

    /// Number of transfer nodes removed in the last run.
    [[nodiscard]] size_t num_eliminated() const { return _num_eliminated; }

  private:
    size_t _num_eliminated{0};
};

} // namespace einsums::compute_graph::passes
