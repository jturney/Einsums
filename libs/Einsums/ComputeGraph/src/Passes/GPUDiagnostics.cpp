//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp>
#include <Einsums/Logging.hpp>

#include <ostream>

namespace einsums::compute_graph::passes {

bool GPUDiagnostics::run(Graph &graph) {
    _gpu_nodes            = 0;
    _cpu_nodes            = 0;
    _h2d_transfers        = 0;
    _d2h_transfers        = 0;
    _total_transfer_bytes = 0;
    _peak_device_bytes    = 0;

    // Simulate device memory usage to find peak.
    size_t current_device_bytes = 0;

    for (auto const &node : graph.nodes()) {
        if (node.kind == OpKind::HostToDevice) {
            _h2d_transfers++;
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (desc) {
                _total_transfer_bytes += desc->size_bytes;
                current_device_bytes += desc->size_bytes;
                if (current_device_bytes > _peak_device_bytes) {
                    _peak_device_bytes = current_device_bytes;
                }
            }
        } else if (node.kind == OpKind::DeviceToHost) {
            _d2h_transfers++;
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (desc) {
                _total_transfer_bytes += desc->size_bytes;
                // D2H doesn't free device memory (tensor stays in Both state).
            }
        } else if (node.target == Target::GPU) {
            _gpu_nodes++;
            // GPU outputs go to device.
            for (auto tid : node.outputs) {
                size_t const bytes = graph.tensor(tid).total_bytes();
                current_device_bytes += bytes;
                if (current_device_bytes > _peak_device_bytes) {
                    _peak_device_bytes = current_device_bytes;
                }
            }
        } else {
            _cpu_nodes++;
        }
    }

    EINSUMS_LOG_INFO("GPUDiagnostics: {} GPU nodes, {} CPU nodes, {} H2D + {} D2H transfers ({} bytes total), peak device memory {} bytes",
                     _gpu_nodes, _cpu_nodes, _h2d_transfers, _d2h_transfers, _total_transfer_bytes, _peak_device_bytes);

    // Non-mutating — always returns false.
    return false;
}

void GPUDiagnostics::print_report(std::ostream &os) const {
    os << "=== GPU Diagnostics ===\n";
    os << fmt::format("  GPU nodes:          {}\n", _gpu_nodes);
    os << fmt::format("  CPU nodes:          {}\n", _cpu_nodes);
    os << fmt::format("  H2D transfers:      {}\n", _h2d_transfers);
    os << fmt::format("  D2H transfers:      {}\n", _d2h_transfers);
    os << fmt::format("  Total transfer:     {} bytes ({:.2f} MB)\n", _total_transfer_bytes,
                      static_cast<double>(_total_transfer_bytes) / (1024.0 * 1024.0));
    os << fmt::format("  Peak device memory: {} bytes ({:.2f} MB)\n", _peak_device_bytes,
                      static_cast<double>(_peak_device_bytes) / (1024.0 * 1024.0));
}

} // namespace einsums::compute_graph::passes
