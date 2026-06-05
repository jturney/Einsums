//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Insert HostToDevice and DeviceToHost transfer nodes around GPU-placed operations.
 *
 * For each GPU-targeted node:
 * - For each input tensor: if not already on the device, insert an HostToDevice node.
 *   Skips H2D for "dead" inputs (e.g., C in `C = A*B` when c_prefactor == 0).
 * - For each output tensor: if a later CPU node reads it, insert a DeviceToHost node.
 *
 * After the main loop, inserts final D2H nodes for any user-visible (non-intermediate)
 * tensor that is still Device-resident. This makes the graph self-contained: on
 * discrete GPUs (CUDA/HIP), explicit D2H ensures results are available to the user
 * after execute() without relying on an implicit flush. On unified memory (MPS),
 * these D2H nodes exist in the graph structure but execution skips the actual copy.
 *
 * Tensor residency in TensorHandle is updated as transfers are inserted:
 * - After HostToDevice: residency = Both
 * - After DeviceToHost: residency = Both
 * - GPU node outputs: residency = Device
 *
 * This pass must run AFTER GPUPlacement.
 * Redundant transfers are removed by TransferElimination.
 */
class EINSUMS_EXPORT TransferInsertion : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "TransferInsertion"; }
    bool                      run(Graph &graph) override;

    /// Recurse into loop bodies / conditional branches. TransferInsertion
    /// is a per-graph transform: run on a body it inserts that body's H2D
    /// before each GPU op and D2H after, making the body self-contained.
    /// (Re-transfers loop-invariant inputs each iteration — hoisting those
    /// before the loop is a later optimization.) See
    /// docs/loop_handling_audit.md.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return true; }

    /// Number of transfer nodes inserted in the last run.
    [[nodiscard]] size_t num_transfers() const { return _num_transfers; }

  private:
    size_t _num_transfers{0};
};

} // namespace einsums::compute_graph::passes
