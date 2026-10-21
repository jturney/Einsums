//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <ostream>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BudgetShare.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

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

/// @p held_elsewhere: buffers the sweep leaves out, because they are held somewhere that is not
/// freed after their last use (a placed arena).
MemStats analyze_one(Graph &graph, std::unordered_set<TensorId> const &held_elsewhere = {}) {
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

    // Liveness intervals come from the shared UsageAnalysis (owner-resolved:
    // a view and its parent count as ONE buffer, so bytes are not
    // double-counted). Subtree expansion is OFF: this pass aggregates body
    // tensors through its own recursive walk, and effective-IO's orphan
    // parent handles would double-count them here. The device-residency
    // flags stay a local scan - they are this pass's policy (GPU targets
    // and transfer descriptors), not a graph-wide property.
    auto const &ua = graph.usage();
    for (auto const &[tid, use] : ua.table()) {
        size_t const fu = use.first_use(/*include_subtree=*/false);
        if (fu == TensorUsage::npos) {
            continue;
        }
        auto const it = tensors.find(tid);
        if (it == tensors.end() || held_elsewhere.contains(tid)) {
            continue;
        }
        intervals[tid] = LiveInterval{fu, use.last_use(/*ignore_free=*/false, /*include_subtree=*/false), it->second.total_bytes(), false};
    }

    // find(), not operator[]: a default-constructed interval (first_use ==
    // SIZE_MAX) would blow up the event sweep below.
    auto mark_device = [&](TensorId tid) {
        if (auto it = intervals.find(graph.buffer_of(tid)); it != intervals.end()) {
            it->second.on_device = true;
        }
    };
    for (size_t i = 0; i < n; i++) {
        auto const &node = nodes[i];
        if (node.target == Target::GPU) {
            for (auto tid : node.inputs)
                mark_device(tid);
            for (auto tid : node.outputs)
                mark_device(tid);
        }
        if (node.kind == OpKind::HostToDevice) {
            auto const *desc = node.op_data.get_if<TransferDescriptor>();
            if (desc)
                mark_device(desc->tensor_id);
        }
    }

    // ── Host + device memory analysis ───────────────────────────────────────
    // Peak via an event sweep (+bytes at first_use, -bytes past last_use):
    // O(n + tids) rather than a per-position O(n * tids) rescan.
    std::vector<long long> delta(n + 1, 0);
    std::vector<long long> device_delta(n + 1, 0);
    for (auto const &[tid, interval] : intervals) {
        stats.total += interval.bytes;
        delta[interval.first_use] += static_cast<long long>(interval.bytes);
        delta[interval.last_use + 1] -= static_cast<long long>(interval.bytes);
        if (interval.on_device) {
            stats.device_total += interval.bytes;
            device_delta[interval.first_use] += static_cast<long long>(interval.bytes);
            device_delta[interval.last_use + 1] -= static_cast<long long>(interval.bytes);
        }
    }
    long long live = 0, device_live = 0;
    for (size_t i = 0; i < n; i++) {
        live += delta[i];
        device_live += device_delta[i];
        stats.peak        = std::max(stats.peak, static_cast<size_t>(live));
        stats.device_peak = std::max(stats.device_peak, static_cast<size_t>(device_live));
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
    explicit ArenaBlock(size_t bytes) : size(bytes), data(static_cast<std::byte *>(memory::aligned_alloc(bytes))) {
        if (data == nullptr && bytes != 0) {
            throw std::bad_alloc{};
        }
    }
    ~ArenaBlock() { memory::aligned_free(data); }
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

/// Records a declined candidate. A callback rather than the pass itself because
/// ``OptimizerPass::note_skip`` is protected, which is the right default: only a
/// pass says why IT declined.
using NoteSkip = std::function<void(std::string_view, std::string_view)>;

/// Plan storage for one flat host-only graph: every graph-owned intermediate
/// whose lifetime is bracketed by exactly one Materialize and one Free node
/// (placed by the Materialization / FreeInsertion passes that run earlier)
/// gets a first-fit-decreasing offset in a shared arena. Intervals are the
/// [materialize, free] node positions - anything outside that window cannot
/// touch the buffer.
ArenaPlan plan_arena(Graph &graph, NoteSkip const &note) {
    ArenaPlan plan;
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // GPU placement swaps buffers behind the slots, so device work declines the level.
    //
    // A loop or a branch at this level is one position in it. Materialization and FreeInsertion
    // bracket every tensor a body uses around the node that holds the body, never inside it, so
    // each such tensor's [Materialize, Free] interval here contains that position: it is live for
    // every iteration, and a tensor whose interval does not reach the position is not used in the
    // body. The interval test is sound at this level as it is on a flat graph. It is not inside a
    // loop body, where positions are local to one iteration and a value written in one and read in
    // the next would look dead between them; plan_tree never plans there.
    //
    // A Setup node is not a bail either. Only the tensors a Setup body touches may live ACROSS
    // replays, so they alone are excluded below; declining the whole level would cost the arena on
    // every fitting graph and every guarded antisymmetrizer fold.
    bool has_setup        = false;
    bool has_control_flow = false;
    for (auto const &node : nodes) {
        if (node.target == Target::GPU || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            note("graph holds device work at this level", graph.name());
            return plan;
        }
        has_setup        = has_setup || node.kind == OpKind::Setup;
        has_control_flow = has_control_flow || node.kind == OpKind::Loop || node.kind == OpKind::Conditional;
    }

    // Built only when a Setup is present, since it walks the whole sub-graph
    // tree to collect the pointers its bodies reference.
    std::optional<EscapeAnalysis> escapes;
    if (has_setup) {
        escapes = EscapeAnalysis::over(graph);
        // The per-tensor exclusion below is only as good as the pointer set it
        // reads. A body reference the sub-graph's map cannot resolve, or one
        // held by an unattached shell, adds no pointer and so reports its
        // tensor "untouched" - the one wrong answer this caller cannot absorb,
        // because it ends in two live buffers sharing a slot. Where the subtree
        // cannot be read exactly, decline the whole level.
        if (escapes->subtree_refs_unresolved()) {
            note("a setup subtree holds references that cannot be resolved to buffers", graph.name());
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

    // Tensors somebody views must keep their own storage: here, and in every body below, where a
    // view of this level's tensor names it through a handle of the body's own.
    std::unordered_set<TensorId>     view_targets;
    std::unordered_set<void const *> viewed_below;
    for (auto const &[tid, handle] : tensors) {
        if (handle.aliases != 0) {
            view_targets.insert(handle.aliases);
        }
    }
    if (has_control_flow) {
        graph.for_each_descendant(std::function<void(Graph &)>{[&](Graph &sub) {
            for (auto const &[tid, handle] : sub.tensors_map()) {
                if (handle.aliases == 0) {
                    continue;
                }
                if (auto const *parent = sub.find_tensor(sub.buffer_of(handle.aliases));
                    parent != nullptr && parent->tensor_ptr != nullptr) {
                    viewed_below.insert(parent->tensor_ptr);
                }
            }
        }});
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
        // A tensor viewed in a body, or one with no object to compare against the views there.
        if (has_control_flow && (handle.tensor_ptr == nullptr || viewed_below.contains(handle.tensor_ptr))) {
            continue;
        }
        // A Setup body may hold this buffer across replays. touched_by_subtree
        // compares POINTERS, so it can only answer for a tensor that has one;
        // a deferred shell is unattached at optimize time and would come back
        // "untouched" from a comparison against an empty set. That answer is
        // safe where a wrong one costs a missed rewrite and not here, where it
        // would alias two live buffers, so an unattached tensor is skipped
        // rather than trusted.
        if (has_setup && (handle.tensor_ptr == nullptr || escapes->touched_by_subtree(tid))) {
            note("a setup body may hold the buffer across replays", handle.name);
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
    // The arena is held for every replay, so Graph::last_run counts it once, beside the tensors
    // the graph allocates and frees.
    graph.run_context()->set_arena(plan.tensors.empty() ? 0 : plan.arena_bytes);
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

    // Storage-reuse WAW edge. Parallel executors order nodes only by
    // shared-TensorId hazards, never by node position. When Y's arena bytes
    // overlap X's and X frees before Y materializes, Materialize(Y) claims
    // X's storage - so declare it a writer of X's tid. rebuild_deps then
    // chains readers(X) < Free(X) < Materialize(Y) (Free declares its tensor
    // as both input and output, so it is X's last writer).
    for (auto const &x : plan.tensors) {
        for (auto const &y : plan.tensors) {
            if (x.tid == y.tid) {
                continue;
            }
            bool const bytes_overlap = x.offset < y.offset + align_up(y.bytes) && y.offset < x.offset + align_up(x.bytes);
            if (!bytes_overlap || x.free_node >= y.mat_node) {
                continue;
            }
            auto &outs = nodes[y.mat_node].outputs;
            if (std::ranges::find(outs, x.tid) == outs.end()) {
                outs.push_back(x.tid);
            }
        }
    }

    // io lists changed: order still valid, position-keyed deps are stale.
    // mark_sorted() vouches for the order and flags the deps stale; execute()
    // (serial or concurrent) rebuilds them in place before running, so the
    // storage-reuse edges are live without an explicit re-sort here.
    graph.mark_sorted();

    EINSUMS_LOG_INFO("MemoryPlanning: arena of {} bytes hosts {} intermediates ({} bytes of buffers)", plan.arena_bytes,
                     plan.tensors.size(), [&] {
                         size_t s = 0;
                         for (auto const &t : plan.tensors)
                             s += t.bytes;
                         return s;
                     }());
}

/// @p repeats: @p graph runs inside a loop, at this depth or above, so a tensor's liveness can cross
/// from one iteration to the next and the interval test does not hold. Nothing there is planned.
void plan_tree(Graph &graph, NoteSkip const &note, bool apply, bool repeats, size_t &arena_bytes, size_t &num_planned, size_t &tensor_bytes,
               size_t &peak_with_arena) {
    if (repeats) {
        // Said only where something would have been planned: a body holds no lifecycle node of its
        // own unless a pass left one there.
        if (std::ranges::any_of(graph.nodes(), [](Node const &node) { return node.kind == OpKind::Materialize; })) {
            note("the graph runs inside a loop, where a tensor's liveness crosses iterations", graph.name());
        }
    } else {
        ArenaPlan const plan = plan_arena(graph, note);
        // An arena is one block held for as long as the graph is, so the plan with it holds the
        // arena whole beside everything else at its own peak, while without it every intermediate
        // is freed after its last use. On the CCSD tutorial's file-backed stage the arena's 4.61 MB
        // took the peak from 9.50 MB to 11.30 MB for no measurable speed. It is kept wherever
        // memory allows, since elsewhere it has paid for itself, and declined only where it holds
        // the plan over --einsums:max-memory and the plan without it is lower.
        if (!plan.tensors.empty()) {
            std::unordered_set<TensorId> placed;
            for (auto const &t : plan.tensors) {
                placed.insert(graph.buffer_of(t.tid));
            }
            std::size_t const without = analyze_one(graph).peak;
            std::size_t const with    = analyze_one(graph, placed).peak + plan.arena_bytes;
            std::size_t const budget  = detail::memory_budget();
            if (budget != 0 && with > budget && with > without) {
                note(MemoryPlanning::ArenaOverBudget.text,
                     fmt::format("graph '{}': {} bytes at the peak with an arena of {}, {} without, against a budget of {}", graph.name(),
                                 with, plan.arena_bytes, without, budget));
                for (auto &node : graph.nodes()) {
                    bool const loop = node.kind == OpKind::Loop;
                    for_each_child_graph(node, [&](Graph &sub) {
                        plan_tree(sub, note, apply, repeats || loop, arena_bytes, num_planned, tensor_bytes, peak_with_arena);
                    });
                }
                return;
            }
            peak_with_arena = std::max(peak_with_arena, with);
        }
        arena_bytes += plan.arena_bytes;
        num_planned += plan.tensors.size();
        for (auto const &t : plan.tensors) {
            tensor_bytes += t.bytes;
        }
        if (apply) {
            apply_arena_plan(graph, plan);
        }
    }
    for (auto &node : graph.nodes()) {
        bool const loop = node.kind == OpKind::Loop;
        for_each_child_graph(node, [&](Graph &sub) {
            plan_tree(sub, note, apply, repeats || loop, arena_bytes, num_planned, tensor_bytes, peak_with_arena);
        });
    }
}

/// The replay's own count of what the graph holds, kept the way the lifecycle nodes keep it: a
/// Materialize counts its tensor when the tensor is not held already, a Free releases it, and a
/// tensor no Free releases stays held into the next replay. An arena's nodes count nothing; the
/// arena is counted whole.
struct ManagedCount {
    std::unordered_map<void const *, std::size_t> live;
    std::size_t                                   now{0};
    std::size_t                                   peak{0};

    void replay(Graph &graph) {
        for (auto &node : graph.nodes()) {
            if ((node.kind == OpKind::Materialize || node.kind == OpKind::Free) && node.estimated_bytes != 0) {
                auto const *desc   = node.op_data.get_if<AllocDescriptor>();
                auto const *handle = desc != nullptr ? graph.find_tensor(graph.buffer_of(desc->tensor_id)) : nullptr;
                if (handle != nullptr && handle->tensor_ptr != nullptr && handle->is_materialized_fn) {
                    if (node.kind == OpKind::Materialize) {
                        if (live.try_emplace(handle->tensor_ptr, handle->total_bytes()).second) {
                            now += handle->total_bytes();
                            peak = std::max(peak, now);
                        }
                    } else if (auto const it = live.find(handle->tensor_ptr); it != live.end()) {
                        now -= it->second;
                        live.erase(it);
                    }
                }
            }
            // A Setup body runs on the first replay only; the replay this models is a later one.
            for_each_child_graph(node, [&](Graph &sub) { replay(sub); }, /*include_setup=*/false);
        }
    }
};

/// The most a replay holds of the tensors the graph allocates and frees, plus @p arena: what
/// Graph::last_run reports as peak_bytes. A later replay, since the first begins with nothing held.
std::size_t managed_peak(Graph &graph, std::size_t arena) {
    ManagedCount first;
    first.replay(graph);
    ManagedCount steady;
    steady.live = first.live;
    steady.now  = first.now;
    steady.peak = first.now;
    steady.replay(graph);
    return steady.peak + arena;
}

} // namespace

void MemoryPlanning::reset_stats() {
    _managed_peak_memory  = 0;
    _planned_arena_bytes  = 0;
    _num_planned          = 0;
    _planned_tensor_bytes = 0;
}

bool MemoryPlanning::transform(Graph &graph) {
    PassCounter const planned{_num_planned};
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

    std::size_t peak_with_arena = 0;
    plan_tree(
        graph, [this](std::string_view reason, std::string_view detail) { note_skip(reason, detail); }, _apply_arena, /*repeats=*/false,
        _planned_arena_bytes, _num_planned, _planned_tensor_bytes, peak_with_arena);
    // The peak the plan holds: an applied arena is held whole, beside what is live at the peak.
    if (_apply_arena) {
        _peak_memory = std::max(_peak_memory, peak_with_arena);
    }
    _managed_peak_memory = managed_peak(graph, _apply_arena ? _planned_arena_bytes : 0);

    note(Peak,
         fmt::format("peak host memory {} bytes (of {} total allocated), {} of them graph-managed as a replay holds them{}", _peak_memory,
                     _total_memory, _managed_peak_memory,
                     _device_peak_memory > 0 ? fmt::format(", device peak {} bytes", _device_peak_memory) : std::string{}),
         Decision{.graph = graph.name(),
                  .facts = {Fact{.name = "peak", .value = static_cast<double>(_peak_memory), .unit = "B"},
                            Fact{.name = "managed_peak", .value = static_cast<double>(_managed_peak_memory), .unit = "B"},
                            Fact{.name = "total", .value = static_cast<double>(_total_memory), .unit = "B"},
                            Fact{.name = "device_peak", .value = static_cast<double>(_device_peak_memory), .unit = "B"}}},
         1);
    if (planned.moved()) {
        Decision    about{.graph = graph.name(),
                          .facts = {Fact{.name = "arena", .value = static_cast<double>(_planned_arena_bytes), .unit = "B"},
                                    Fact{.name = "intermediates", .value = static_cast<double>(_num_planned)},
                                    Fact{.name = "shared", .value = static_cast<double>(_planned_tensor_bytes), .unit = "B"}}};
        std::string text = fmt::format("{} arena of {} bytes for {} intermediate(s) ({} bytes of buffers shared into it)",
                                       _apply_arena ? "applied" : "planned", _planned_arena_bytes, _num_planned, _planned_tensor_bytes);
        if (_apply_arena) {
            applied(Arena, std::move(text), std::move(about), 1);
        } else {
            note(Arena, std::move(text), std::move(about), 1);
        }
    }

    return _apply_arena && planned.moved();
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

std::vector<std::string> MemoryPlanning::explain() const {
    if (total_memory() == 0) {
        return {};
    }
    std::vector<std::string> out{fmt::format("MemoryPlanning: peak {:.2f} MB of {:.2f} MB total",
                                             static_cast<double>(peak_memory()) / 1048576.0,
                                             static_cast<double>(total_memory()) / 1048576.0)};
    if (num_planned() > 0) {
        out.push_back(fmt::format("    arena: {:.2f} MB hosting {} intermediate(s) ({:.2f} MB of buffers)",
                                  static_cast<double>(planned_arena_bytes()) / 1048576.0, num_planned(),
                                  static_cast<double>(planned_tensor_bytes()) / 1048576.0));
    }
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
