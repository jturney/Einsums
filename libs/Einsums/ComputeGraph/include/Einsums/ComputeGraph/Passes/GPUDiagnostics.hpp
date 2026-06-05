//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <iosfwd>

namespace einsums::compute_graph::passes {

/**
 * @brief Non-mutating diagnostic pass that summarizes GPU placement and transfers.
 *
 * Logs (and optionally prints) a report of:
 * - Nodes placed on GPU vs CPU
 * - Total H2D and D2H transfer nodes
 * - Total bytes transferred
 * - Peak estimated device memory usage
 *
 * This pass does NOT modify the graph.
 * Should run after GPUPlacement, TransferInsertion, and TransferElimination.
 */
class EINSUMS_EXPORT GPUDiagnostics : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "GPUDiagnostics"; }
    bool                      run(Graph &graph) override;

    /// Print the diagnostic report to a stream.
    void print_report(std::ostream &os) const;

    [[nodiscard]] size_t gpu_nodes() const { return _gpu_nodes; }
    [[nodiscard]] size_t cpu_nodes() const { return _cpu_nodes; }
    [[nodiscard]] size_t h2d_transfers() const { return _h2d_transfers; }
    [[nodiscard]] size_t d2h_transfers() const { return _d2h_transfers; }
    [[nodiscard]] size_t total_transfer_bytes() const { return _total_transfer_bytes; }
    [[nodiscard]] size_t peak_device_bytes() const { return _peak_device_bytes; }

  private:
    size_t _gpu_nodes{0};
    size_t _cpu_nodes{0};
    size_t _h2d_transfers{0};
    size_t _d2h_transfers{0};
    size_t _total_transfer_bytes{0};
    size_t _peak_device_bytes{0};
};

} // namespace einsums::compute_graph::passes
