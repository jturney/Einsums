//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

bool MemoryPlanning::run(Graph &graph) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    if (nodes.empty() || tensors.empty()) {
        _total_memory        = 0;
        _peak_memory         = 0;
        _device_total_memory = 0;
        _device_peak_memory  = 0;
        return false;
    }

    size_t const n = nodes.size();

    // Compute liveness intervals: [first_use, last_use] for each tensor.
    struct LiveInterval {
        size_t first_use{SIZE_MAX};
        size_t last_use{0};
        size_t bytes{0};
        bool   on_device{false}; ///< True if this tensor is used by a GPU node or is a transfer target.
    };

    std::unordered_map<TensorId, LiveInterval> intervals;

    for (size_t i = 0; i < n; i++) {
        auto const &node = nodes[i];

        auto update = [&](TensorId tid) {
            auto &interval     = intervals[tid];
            interval.first_use = std::min(interval.first_use, i);
            interval.last_use  = std::max(interval.last_use, i);
            auto it            = tensors.find(tid);
            if (it != tensors.end()) {
                interval.bytes = it->second.total_bytes();
            }
        };

        for (auto tid : node.inputs) {
            update(tid);
        }
        for (auto tid : node.outputs) {
            update(tid);
        }

        // Mark tensors as device-resident if used by GPU or transfer nodes.
        if (node.target == Target::GPU) {
            for (auto tid : node.inputs)
                intervals[tid].on_device = true;
            for (auto tid : node.outputs)
                intervals[tid].on_device = true;
        }
        if (node.kind == OpKind::HostToDevice) {
            auto const *desc = std::get_if<TransferDescriptor>(&node.op_data);
            if (desc)
                intervals[desc->tensor_id].on_device = true;
        }
    }

    // ── Host memory analysis ────────────────────────────────────────────────
    _total_memory = 0;
    for (auto const &[tid, interval] : intervals) {
        _total_memory += interval.bytes;
    }

    _peak_memory = 0;
    for (size_t i = 0; i < n; i++) {
        size_t live_bytes = 0;
        for (auto const &[tid, interval] : intervals) {
            if (i >= interval.first_use && i <= interval.last_use) {
                live_bytes += interval.bytes;
            }
        }
        _peak_memory = std::max(_peak_memory, live_bytes);
    }

    // ── Device memory analysis ──────────────────────────────────────────────
    // Only consider tensors that are actually used on the device.
    _device_total_memory = 0;
    for (auto const &[tid, interval] : intervals) {
        if (interval.on_device) {
            _device_total_memory += interval.bytes;
        }
    }

    // Device peak: simulate which device tensors are live at each node.
    // A device tensor is "live" from its first GPU/transfer use to its last.
    _device_peak_memory = 0;
    for (size_t i = 0; i < n; i++) {
        size_t live_bytes = 0;
        for (auto const &[tid, interval] : intervals) {
            if (interval.on_device && i >= interval.first_use && i <= interval.last_use) {
                live_bytes += interval.bytes;
            }
        }
        _device_peak_memory = std::max(_device_peak_memory, live_bytes);
    }

    return false;
}

void MemoryPlanning::print_report(std::ostream &os) const {
    os << fmt::format("MemoryPlanning report:\n");
    os << fmt::format("  Total tensor memory: {} bytes ({:.2f} MB)\n", _total_memory,
                      static_cast<double>(_total_memory) / (1024.0 * 1024.0));
    os << fmt::format("  Peak live memory:    {} bytes ({:.2f} MB)\n", _peak_memory, static_cast<double>(_peak_memory) / (1024.0 * 1024.0));
    if (_total_memory > 0 && _peak_memory < _total_memory) {
        os << fmt::format("  Potential savings:   {} bytes ({:.1f}%)\n", _total_memory - _peak_memory,
                          100.0 * (1.0 - static_cast<double>(_peak_memory) / static_cast<double>(_total_memory)));
    }

    if (_device_total_memory > 0) {
        os << fmt::format("  Device total memory: {} bytes ({:.2f} MB)\n", _device_total_memory,
                          static_cast<double>(_device_total_memory) / (1024.0 * 1024.0));
        os << fmt::format("  Device peak memory:  {} bytes ({:.2f} MB)\n", _device_peak_memory,
                          static_cast<double>(_device_peak_memory) / (1024.0 * 1024.0));
        if (_device_total_memory > _device_peak_memory) {
            os << fmt::format("  Device reuse savings: {} bytes ({:.1f}%)\n", _device_total_memory - _device_peak_memory,
                              100.0 * (1.0 - static_cast<double>(_device_peak_memory) / static_cast<double>(_device_total_memory)));
        }
    }
}

} // namespace einsums::compute_graph::passes
