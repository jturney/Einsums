//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <new>
#include <ostream>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
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

// ─── Planned arena ──────────────────────────────────────────────────────────

/// One shared, 64-byte-aligned block per graph, owned by shared_ptr captures
/// in the rewritten Materialize/Free executors, so it lives exactly as long
/// as the graph's nodes and is reused verbatim by every replay.
struct ArenaBlock {
    explicit ArenaBlock(size_t bytes) : size(bytes), data(static_cast<std::byte *>(::operator new[](bytes, std::align_val_t{64}))) {}
    ~ArenaBlock() { ::operator delete[](data, std::align_val_t{64}); }
    ArenaBlock(ArenaBlock const &)            = delete;
    ArenaBlock &operator=(ArenaBlock const &) = delete;

    size_t     size;
    std::byte *data;
};

constexpr size_t kArenaAlign = 64;

size_t align_up(size_t v) {
    return (v + kArenaAlign - 1) & ~(kArenaAlign - 1);
}

struct PlannedTensor {
    TensorId tid;
    size_t   bytes;     ///< raw tensor bytes
    size_t   offset;    ///< assigned arena offset (align_up'd extent)
    size_t   mat_node;  ///< position of the tensor's Materialize node
    size_t   free_node; ///< position of the tensor's Free node
};

struct ArenaPlan {
    std::vector<PlannedTensor> tensors;
    size_t                     arena_bytes{0};
};

/// Plan storage for one flat host-only graph: every graph-owned intermediate
/// whose lifetime is bracketed by exactly one Materialize and one Free node
/// (placed by the Materialization / FreeInsertion passes that run earlier)
/// gets a first-fit-decreasing offset in a shared arena. Intervals are the
/// [materialize, free] node positions - anything outside that window cannot
/// touch the buffer.
ArenaPlan plan_arena(Graph &graph) {
    ArenaPlan plan;
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Same conservatism as InplaceOptimization: control flow at this level
    // references tensors invisibly to plain node lists, and GPU placement
    // swaps buffers behind the slots.
    for (auto const &node : nodes) {
        if (node.kind == OpKind::Loop || node.kind == OpKind::Conditional || node.target == Target::GPU ||
            node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            return plan;
        }
    }

    // Locate each tensor's Materialize/Free bracket.
    std::unordered_map<TensorId, std::pair<size_t, size_t>> bracket; // tid -> (mat, free), SIZE_MAX = unseen/duplicate
    for (size_t i = 0; i < nodes.size(); i++) {
        auto const &node = nodes[i];
        if (node.kind == OpKind::Materialize && node.outputs.size() == 1) {
            auto [it, fresh] = bracket.try_emplace(node.outputs[0], std::pair{i, SIZE_MAX});
            if (!fresh) {
                it->second.first = SIZE_MAX; // duplicate materialize: disqualify
            }
        } else if (node.kind == OpKind::Free && node.inputs.size() == 1) {
            auto it = bracket.find(node.inputs[0]);
            if (it != bracket.end()) {
                it->second.second = it->second.second == SIZE_MAX ? i : SIZE_MAX;
            }
        }
    }

    // Tensors somebody views must keep their own storage.
    std::unordered_set<TensorId> view_targets;
    for (auto const &[tid, handle] : tensors) {
        if (handle.aliases != 0) {
            view_targets.insert(handle.aliases);
        }
    }

    for (auto const &[tid, br] : bracket) {
        auto const [mat, fre] = br;
        if (mat == SIZE_MAX || fre == SIZE_MAX || fre <= mat) {
            continue;
        }
        auto it = tensors.find(tid);
        if (it == tensors.end()) {
            continue;
        }
        auto const &handle = it->second;
        if (!handle.is_intermediate || handle.aliases != 0 || view_targets.contains(tid) || !handle.materialize_into_fn ||
            !handle.release_fn || handle.total_bytes() == 0) {
            continue;
        }
        plan.tensors.push_back({.tid = tid, .bytes = handle.total_bytes(), .offset = 0, .mat_node = mat, .free_node = fre});
    }

    if (plan.tensors.empty()) {
        return plan;
    }

    // First-fit-decreasing over the lifetime intervals: place each tensor at
    // the lowest offset whose byte range is free of every overlapping-lifetime
    // tensor already placed.
    std::ranges::sort(plan.tensors, [](PlannedTensor const &a, PlannedTensor const &b) { return a.bytes > b.bytes; });

    std::vector<PlannedTensor const *> placed;
    for (auto &pt : plan.tensors) {
        size_t const extent = align_up(pt.bytes);
        size_t       offset = 0;
        bool         moved  = true;
        while (moved) {
            moved = false;
            for (auto const *other : placed) {
                bool const lifetimes_overlap = pt.mat_node <= other->free_node && other->mat_node <= pt.free_node;
                bool const bytes_overlap     = offset < other->offset + align_up(other->bytes) && other->offset < offset + extent;
                if (lifetimes_overlap && bytes_overlap) {
                    offset = other->offset + align_up(other->bytes); // skip past, rescan
                    moved  = true;
                }
            }
        }
        pt.offset        = offset;
        plan.arena_bytes = std::max(plan.arena_bytes, offset + extent);
        placed.push_back(&pt);
    }

    return plan;
}

/// Rewrite the planned tensors' Materialize/Free executors to attach/detach
/// arena slices instead of allocating/freeing.
void apply_arena_plan(Graph &graph, ArenaPlan const &plan) {
    if (plan.tensors.empty()) {
        return;
    }

    auto  arena = std::make_shared<ArenaBlock>(plan.arena_bytes);
    auto &nodes = graph.nodes();

    for (auto const &pt : plan.tensors) {
        auto const &handle = graph.tensor(pt.tid);

        // Materialize: attach at the planned offset. Idempotent across
        // replays (materialize_into with the same pointer is a no-op).
        nodes[pt.mat_node].execute = [arena, offset = pt.offset, attach = handle.materialize_into_fn]() {
            attach(static_cast<void *>(arena->data + offset));
        };
        // The tensor no longer allocates; don't charge it to the Dataflow
        // memory-budget gate.
        nodes[pt.mat_node].estimated_bytes = 0;

        // Free: detach only. Tensor::release() on external storage parks the
        // impl on the sentinel without freeing; the arena block persists for
        // the next replay.
        nodes[pt.free_node].execute         = [rel = handle.release_fn]() { rel(); };
        nodes[pt.free_node].estimated_bytes = 0;
    }

    EINSUMS_LOG_INFO("MemoryPlanning: arena of {} bytes hosts {} intermediates ({} bytes of buffers)", plan.arena_bytes,
                     plan.tensors.size(), [&] {
                         size_t s = 0;
                         for (auto const &t : plan.tensors)
                             s += t.bytes;
                         return s;
                     }());
}

void plan_tree(Graph &graph, bool apply, size_t &arena_bytes, size_t &num_planned, size_t &tensor_bytes) {
    ArenaPlan const plan = plan_arena(graph);
    arena_bytes += plan.arena_bytes;
    num_planned += plan.tensors.size();
    for (auto const &t : plan.tensors) {
        tensor_bytes += t.bytes;
    }
    if (apply) {
        apply_arena_plan(graph, plan);
    }
    graph.for_each_subgraph([&](Graph &sub) { plan_tree(sub, apply, arena_bytes, num_planned, tensor_bytes); });
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

    _planned_arena_bytes  = 0;
    _num_planned          = 0;
    _planned_tensor_bytes = 0;
    plan_tree(graph, _apply_arena, _planned_arena_bytes, _num_planned, _planned_tensor_bytes);

    report(1, fmt::format("peak host memory {} bytes (of {} total allocated){}", _peak_memory, _total_memory,
                          _device_peak_memory > 0 ? fmt::format(", device peak {} bytes", _device_peak_memory) : std::string{}));
    if (_num_planned > 0) {
        report(1, fmt::format("{} arena of {} bytes for {} intermediate(s) ({} bytes of buffers shared into it)",
                              _apply_arena ? "applied" : "planned", _planned_arena_bytes, _num_planned, _planned_tensor_bytes));
    }

    return _apply_arena && _num_planned > 0;
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
    if (_num_planned > 0) {
        os << fmt::format("  Arena ({}):          {} bytes hosting {} intermediates ({} bytes of buffers)\n",
                          _apply_arena ? "applied" : "planned", _planned_arena_bytes, _num_planned, _planned_tensor_bytes);
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
