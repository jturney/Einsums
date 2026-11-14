//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ReplayState.cpp
/// @brief What a graph's replays record: the run metrics of the outermost replay, the bytes the
///        tree holds, the timings and their labelled report.

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {

auto ReplayState::rebuild_profile_strings(std::uint32_t exec_zone, std::size_t nodes) -> std::vector<NodeProfileStrings> & {
    _exec_zone_id = exec_zone;
    _profile_strings.clear();
    _profile_strings.resize(nodes);
    return _profile_strings;
}

void ReplayState::record_timings(std::vector<NodeTimingSample> &&samples) {
    if (_timing_samples.empty()) {
        _timing_samples = std::move(samples);
    } else {
        _timing_samples.insert(_timing_samples.end(), samples.begin(), samples.end());
    }
}

void ReplayState::clear_timings() noexcept {
    _timing_samples.clear();
    _timing_report.clear();
    _timing_report_valid = true;
}

void ReplayState::begin_run_metrics(Graph const &graph, std::uint64_t tree_stamp) {
    // What the managed tensors hold as the replay starts: every tensor a Materialize or a Free node
    // of the tree allocates or releases, held now, once. An arena-placed one (its lifecycle nodes
    // carry no bytes) is the arena's, counted with it.
    std::uint64_t          held = 0;
    std::set<void const *> seen;
    auto const             managed = [&](Graph const &g) {
        for (auto const &node : g.nodes()) {
            if ((node.kind != OpKind::Materialize && node.kind != OpKind::Free) || node.estimated_bytes == 0) {
                continue;
            }
            auto const *desc   = node.op_data.get_if<AllocDescriptor>();
            auto const *handle = desc != nullptr ? g.find_tensor(g.buffer_of(desc->tensor_id)) : nullptr;
            if (handle == nullptr || handle->tensor_ptr == nullptr || !handle->materialized_now().has_value() ||
                !seen.insert(handle->tensor_ptr).second) {
                continue;
            }
            if (*handle->materialized_now()) {
                held += handle->total_bytes();
            }
        }
    };
    managed(graph);
    graph.for_each_descendant(managed);
    _run_context->begin_replay(held);
    // Every backing in the tree, once, with its counts as the replay starts: the replay's share is
    // the difference at its end, whichever body or thread did the reading.
    //
    // Which tensors are backed changes only with the tree's tensors, so the walk over every
    // tensor table runs when tensor_tree_stamp() moves; each replay reads only the counts.
    if (!_tree_backings_valid || _tree_backings_stamp != tree_stamp) {
        _tree_backings.clear();
        auto const collect = [this](Graph const &g) {
            for (auto const &[tid, handle] : g.tensors_map()) {
                if (!handle.backing || !handle.backing->counters ||
                    std::ranges::any_of(_tree_backings, [&](BackedTensor const &seen) { return seen.backing == handle.backing; })) {
                    continue;
                }
                _tree_backings.push_back(BackedTensor{.tensor = handle.name, .backing = handle.backing});
            }
        };
        collect(graph);
        graph.for_each_descendant(collect);
        _tree_backings_stamp = tree_stamp;
        _tree_backings_valid = true;
    }
    _replay_backings.clear();
    _replay_backings.reserve(_tree_backings.size());
    for (auto const &[tensor, backing] : _tree_backings) {
        auto const &counts = *backing->counters;
        _replay_backings.push_back(ReplayBacking{.tensor        = tensor,
                                                 .backing       = backing,
                                                 .bytes_read    = counts.bytes_read.load(std::memory_order_relaxed),
                                                 .bytes_written = counts.bytes_written.load(std::memory_order_relaxed),
                                                 .reads         = counts.reads.load(std::memory_order_relaxed),
                                                 .writes        = counts.writes.load(std::memory_order_relaxed),
                                                 .publishes     = counts.publishes.load(std::memory_order_relaxed)});
    }
}

void ReplayState::contribute_run_samples(Graph const *self) {
    std::vector<detail::RunSample> samples;
    samples.reserve(_timing_samples.size());
    for (auto const &sample : _timing_samples) {
        samples.push_back(
            detail::RunSample{.graph = self, .id = sample.id, .kind = sample.kind, .ms = sample.duration_ms, .width = sample.width});
    }
    _run_context->add_samples(std::move(samples));
}

void ReplayState::end_run_metrics(Graph const &graph, double wall_ms, std::uint64_t tree_stamp) {
    _last_samples = _run_context->take_samples();
    _last_backings.clear();
    _last_backings.reserve(_replay_backings.size());
    for (auto const &start : _replay_backings) {
        auto const &counts = *start.backing->counters;
        _last_backings.push_back(BackingRun{.tensor        = start.tensor,
                                            .storage       = start.backing->describe(),
                                            .bytes_read    = counts.bytes_read.load(std::memory_order_relaxed) - start.bytes_read,
                                            .bytes_written = counts.bytes_written.load(std::memory_order_relaxed) - start.bytes_written,
                                            .reads         = counts.reads.load(std::memory_order_relaxed) - start.reads,
                                            .writes        = counts.writes.load(std::memory_order_relaxed) - start.writes,
                                            .publishes     = counts.publishes.load(std::memory_order_relaxed) - start.publishes});
    }
    std::ranges::sort(_last_backings, {}, &BackingRun::tensor);
    _last_wall_ms        = wall_ms;
    _last_peak_bytes     = _run_context->peak_bytes();
    _last_arena_bytes    = _run_context->arena_bytes();
    _last_resident_bytes = resident_bytes(graph, tree_stamp);
    _last_run_valid      = false;

    // The passes plan against --einsums:max-memory, and a plan that cannot meet it is noted only in
    // the pass report. The budget bounds the program, so the replay's own count is set beside what
    // it held throughout, the caller's tensors among them, and a sum over the budget is a budget
    // missed whatever any pass believed; said once per graph, where a caller sees it.
    std::uint64_t const held = _last_peak_bytes + _last_resident_bytes;
    if (!_warned_over_budget && held > 0) {
        std::size_t const budget = string_util::memory_string(config::get(option::MaxMemory));
        if (budget != 0 && held > budget) {
            _warned_over_budget = true;
            EINSUMS_LOG_ERROR("graph '{}': a replay held {} bytes at its peak, {} of graph-managed tensors and {} resident throughout, "
                              "over --einsums:max-memory of {} bytes; the optimizer could not fit the program to the budget "
                              "(--einsums:pass:verbosity 3 reports why)",
                              graph.name(), held, _last_peak_bytes, _last_resident_bytes, budget);
        }
    }
}

std::uint64_t ReplayState::resident_bytes(Graph const &graph, std::uint64_t tree_stamp) const {
    // Every replay asks, for its run metrics and the memory-budget check, and the answer is a walk
    // of every node and tensor in the tree that changes only when those do.
    if (_resident_bytes_valid && _resident_bytes_stamp == tree_stamp) {
        return _resident_bytes_memo;
    }
    // The tensor objects a lifecycle node allocates or frees, anywhere in the tree: the replay's
    // own count holds those, an arena's among them.
    std::unordered_set<void const *> managed;
    // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
    std::function<void(Graph const &)> lifecycle = [&](Graph const &g) {
        for (auto const &node : g.nodes()) {
            if (node.kind == OpKind::Materialize || node.kind == OpKind::Free) {
                if (auto const *desc = node.op_data.get_if<AllocDescriptor>(); desc != nullptr) {
                    if (TensorHandle const *handle = g.find_tensor(g.buffer_of(desc->tensor_id));
                        handle != nullptr && handle->tensor_ptr != nullptr) {
                        managed.insert(handle->tensor_ptr);
                    }
                }
            }
            for_each_child_graph(node, [&](Graph const &child) { lifecycle(child); });
        }
    };
    lifecycle(graph);

    // A body names its parent's tensors through handles of its own over the same object, so a
    // tensor is counted once by that object; a handle with no object is its graph's own.
    std::unordered_set<void const *>             counted;
    std::set<std::pair<Graph const *, TensorId>> unowned;
    std::uint64_t                                total = 0;
    // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
    std::function<void(Graph const &)> count = [&](Graph const &g) {
        std::unordered_set<TensorId> named;
        for (auto const &node : g.nodes()) {
            bool has_body = false;
            for_each_child_graph(node, [&](Graph const &) { has_body = true; });
            if (has_body || node.kind == OpKind::Materialize || node.kind == OpKind::Free) {
                continue;
            }
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const tid : *ids) {
                    named.insert(g.buffer_of(tid));
                }
            }
        }
        for (auto const &[tid, handle] : g.tensors_map()) {
            if (handle.is_tensor_view || g.buffer_of(tid) != tid || !named.contains(tid)) {
                continue;
            }
            if (handle.tensor_ptr != nullptr && managed.contains(handle.tensor_ptr)) {
                continue;
            }
            bool const fresh = handle.tensor_ptr != nullptr ? counted.insert(handle.tensor_ptr).second : unowned.emplace(&g, tid).second;
            if (fresh) {
                total += handle.total_bytes();
            }
        }
        for (auto const &node : g.nodes()) {
            for_each_child_graph(node, [&](Graph const &child) { count(child); });
        }
    };
    count(graph);
    _resident_bytes_memo  = total;
    _resident_bytes_stamp = tree_stamp;
    _resident_bytes_valid = true;
    return total;
}

RunMetrics const &ReplayState::last_run(Graph const &graph) const {
    if (_last_run_valid) {
        return _last_run;
    }
    // Labels and graph names are attached here, once per report, not on the replay path.
    std::unordered_map<Graph const *, std::unordered_map<NodeId, std::string_view>> labels;
    auto const label_of = [&](Graph const &g, NodeId id) -> std::string_view {
        auto [at, added] = labels.try_emplace(&g);
        if (added) {
            for (auto const &node : g.nodes()) {
                at->second.emplace(node.id, node.label);
            }
        }
        auto const hit = at->second.find(id);
        return hit != at->second.end() ? hit->second : std::string_view{};
    };

    RunMetrics out;
    out.wall_ms        = _last_wall_ms;
    out.backings       = _last_backings;
    out.peak_bytes     = _last_peak_bytes;
    out.arena_bytes    = _last_arena_bytes;
    out.resident_bytes = _last_resident_bytes;
    std::map<std::pair<Graph const *, NodeId>, std::size_t> index;
    for (auto const &sample : _last_samples) {
        Graph const &owner        = sample.graph != nullptr ? *sample.graph : graph;
        auto const [at, new_node] = index.try_emplace({&owner, sample.id}, out.nodes.size());
        if (new_node) {
            out.nodes.push_back(NodeRun{.graph = owner.name(),
                                        .id    = sample.id,
                                        .kind  = sample.kind,
                                        .label = std::string{label_of(owner, sample.id)},
                                        .ms    = sample.ms,
                                        .width = sample.width,
                                        .runs  = 1});
        } else {
            NodeRun &run = out.nodes.at(at->second);
            run.ms += sample.ms;
            run.width = sample.width;
            ++run.runs;
        }
    }
    _last_run       = std::move(out);
    _last_run_valid = true;
    return _last_run;
}

std::vector<NodeTiming> const &ReplayState::timing_report(std::span<Node const> nodes) const {
    if (_timing_report_valid) {
        return _timing_report;
    }

    // Attach labels here rather than on the replay path: one string copy per
    // node per REPORT instead of per node per execute().
    std::unordered_map<NodeId, std::string_view> labels;
    labels.reserve(nodes.size());
    for (auto const &node : nodes) {
        labels.emplace(node.id, node.label);
    }

    _timing_report.clear();
    _timing_report.reserve(_timing_samples.size());
    for (auto const &sample : _timing_samples) {
        auto const it = labels.find(sample.id);
        _timing_report.push_back({.id          = sample.id,
                                  .label       = it != labels.end() ? std::string(it->second) : fmt::format("node {}", sample.id),
                                  .kind        = sample.kind,
                                  .duration_ms = sample.duration_ms,
                                  .width       = sample.width});
    }
    _timing_report_valid = true;
    return _timing_report;
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
