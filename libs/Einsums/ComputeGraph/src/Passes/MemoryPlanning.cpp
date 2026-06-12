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

namespace {

// Per-graph memory stats. Totals are summed across the graph tree; peaks
// are maxed (the worst single graph's simultaneously-live footprint). A
// precise cross-loop peak would need loop-carried liveness tracking; max
// is a defensible lower bound for a reporting pass and strictly better
// than ignoring loop bodies entirely.
struct MemStats {
    size_t total{0};
    size_t peak{0};
    size_t device_total{0};
    size_t device_peak{0};
};

MemStats analyze_one(Graph &graph) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    MemStats stats;
    if (nodes.empty() || tensors.empty()) {
        return stats;
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
    for (auto const &[tid, interval] : intervals) {
        stats.total += interval.bytes;
    }

    for (size_t i = 0; i < n; i++) {
        size_t live_bytes = 0;
        for (auto const &[tid, interval] : intervals) {
            if (i >= interval.first_use && i <= interval.last_use) {
                live_bytes += interval.bytes;
            }
        }
        stats.peak = std::max(stats.peak, live_bytes);
    }

    // ── Device memory analysis ──────────────────────────────────────────────
    // Only consider tensors that are actually used on the device.
    for (auto const &[tid, interval] : intervals) {
        if (interval.on_device) {
            stats.device_total += interval.bytes;
        }
    }

    // Device peak: simulate which device tensors are live at each node.
    // A device tensor is "live" from its first GPU/transfer use to its last.
    for (size_t i = 0; i < n; i++) {
        size_t live_bytes = 0;
        for (auto const &[tid, interval] : intervals) {
            if (interval.on_device && i >= interval.first_use && i <= interval.last_use) {
                live_bytes += interval.bytes;
            }
        }
        stats.device_peak = std::max(stats.device_peak, live_bytes);
    }

    return stats;
}

// Accumulate stats over @p graph and every descendant (loop bodies,
// conditional branches, nesting). Totals sum; peaks take the max.
void accumulate(Graph &graph, MemStats &acc) {
    MemStats const s = analyze_one(graph);
    acc.total += s.total;
    acc.device_total += s.device_total;
    acc.peak        = std::max(acc.peak, s.peak);
    acc.device_peak = std::max(acc.device_peak, s.device_peak);
    graph.for_each_subgraph([&](Graph &sub) { accumulate(sub, acc); });
}

} // namespace

bool MemoryPlanning::run(Graph &graph) {
    // Walk the whole graph tree so loop bodies / conditional branches
    // contribute to the reported footprint. recurse_into_subgraphs() stays
    // false: this pass aggregates here rather than being re-run per
    // sub-graph (which would clobber the counters with the last subgraph).
    MemStats acc;
    accumulate(graph, acc);

    _total_memory        = acc.total;
    _peak_memory         = acc.peak;
    _device_total_memory = acc.device_total;
    _device_peak_memory  = acc.device_peak;

    report(1, fmt::format("peak host memory {} bytes (of {} total allocated){}", _peak_memory, _total_memory,
                          _device_peak_memory > 0 ? fmt::format(", device peak {} bytes", _device_peak_memory) : std::string{}));

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
