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

namespace {

struct GpuStats {
    size_t gpu_nodes{0};
    size_t cpu_nodes{0};
    size_t h2d{0};
    size_t d2h{0};
    size_t transfer_bytes{0};
    size_t peak_device_bytes{0};
};

// Count one graph in isolation. Node counts and transfer bytes sum across
// the tree; peak device bytes is maxed (worst single graph). The device
// simulation is per-graph because device residency doesn't carry across a
// loop boundary in this model.
GpuStats count_one(Graph &graph) {
    GpuStats stats;
    size_t   current_device_bytes = 0;

    for (auto const &node : graph.nodes()) {
        if (node.kind == OpKind::HostToDevice) {
            stats.h2d++;
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (desc) {
                stats.transfer_bytes += desc->size_bytes;
                current_device_bytes += desc->size_bytes;
                stats.peak_device_bytes = std::max(stats.peak_device_bytes, current_device_bytes);
            }
        } else if (node.kind == OpKind::DeviceToHost) {
            stats.d2h++;
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (desc) {
                stats.transfer_bytes += desc->size_bytes;
                // D2H doesn't free device memory (tensor stays in Both state).
            }
        } else if (node.target == Target::GPU) {
            stats.gpu_nodes++;
            for (auto tid : node.outputs) {
                size_t const bytes = graph.tensor(tid).total_bytes();
                current_device_bytes += bytes;
                stats.peak_device_bytes = std::max(stats.peak_device_bytes, current_device_bytes);
            }
        } else {
            stats.cpu_nodes++;
        }
    }
    return stats;
}

void accumulate(Graph &graph, GpuStats &acc) {
    GpuStats const s = count_one(graph);
    acc.gpu_nodes += s.gpu_nodes;
    acc.cpu_nodes += s.cpu_nodes;
    acc.h2d += s.h2d;
    acc.d2h += s.d2h;
    acc.transfer_bytes += s.transfer_bytes;
    acc.peak_device_bytes = std::max(acc.peak_device_bytes, s.peak_device_bytes);
    graph.for_each_subgraph([&](Graph &sub) { accumulate(sub, acc); });
}

} // namespace

bool GPUDiagnostics::run(Graph &graph) {
    // Aggregate over the whole graph tree so transfers / GPU nodes inside
    // loop bodies and conditional branches are counted too.
    GpuStats acc;
    accumulate(graph, acc);

    _gpu_nodes            = acc.gpu_nodes;
    _cpu_nodes            = acc.cpu_nodes;
    _h2d_transfers        = acc.h2d;
    _d2h_transfers        = acc.d2h;
    _total_transfer_bytes = acc.transfer_bytes;
    _peak_device_bytes    = acc.peak_device_bytes;

    EINSUMS_LOG_INFO("GPUDiagnostics: {} GPU nodes, {} CPU nodes, {} H2D + {} D2H transfers ({} bytes total), peak device memory {} bytes",
                     _gpu_nodes, _cpu_nodes, _h2d_transfers, _d2h_transfers, _total_transfer_bytes, _peak_device_bytes);
    report(1, fmt::format("{} GPU / {} CPU node(s), {} H2D + {} D2H transfer(s), device peak {} bytes", _gpu_nodes, _cpu_nodes,
                          _h2d_transfers, _d2h_transfers, _peak_device_bytes));

    // Non-mutating, always returns false.
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
