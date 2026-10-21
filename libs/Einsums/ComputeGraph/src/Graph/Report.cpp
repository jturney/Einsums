//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Report.cpp
/// @brief What a graph says about itself: timings, DOT, a summary, JSON.
///
/// Everything a human or a tool reads, and nothing execution depends on. The
/// timing report is the one piece with a live counterpart, since the samples it
/// summarizes are recorded by `Execute.cpp` as it runs.
///
/// The process-wide registry ends the file. It exists so that a debugger or an
/// out-of-process viewer can enumerate the graphs alive right now, which is the
/// same audience the renderers above serve.

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraphTypes/GraphData.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TypeSupport/JsonEscape.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

void Graph::share_run_context(Graph &child) {
    child._run_context = _run_context;
    child.for_each_descendant([this](Graph &sub) { sub._run_context = _run_context; });
}

void Graph::begin_run_metrics() {
    // What the managed tensors hold as the replay starts: every tensor a Materialize or a Free node
    // of the tree allocates or releases, held now, once. An arena-placed one (its lifecycle nodes
    // carry no bytes) is the arena's, counted with it.
    std::uint64_t          held = 0;
    std::set<void const *> seen;
    auto const             managed = [&](Graph const &graph) {
        for (auto const &node : graph.nodes()) {
            if ((node.kind != OpKind::Materialize && node.kind != OpKind::Free) || node.estimated_bytes == 0) {
                continue;
            }
            auto const *desc   = node.op_data.get_if<AllocDescriptor>();
            auto const *handle = desc != nullptr ? graph.find_tensor(graph.buffer_of(desc->tensor_id)) : nullptr;
            if (handle == nullptr || handle->tensor_ptr == nullptr || !handle->is_materialized_fn ||
                !seen.insert(handle->tensor_ptr).second) {
                continue;
            }
            if (handle->is_materialized_fn()) {
                held += handle->total_bytes();
            }
        }
    };
    managed(*this);
    std::as_const(*this).for_each_descendant(managed);
    _run_context->begin_replay(held);
    // Every backing in the tree, once, with its counts as the replay starts: the replay's share is
    // the difference at its end, whichever body or thread did the reading.
    _replay_backings.clear();
    auto const collect = [this](Graph const &graph) {
        for (auto const &[tid, handle] : graph.tensors_map()) {
            if (!handle.backing || !handle.backing->counters ||
                std::ranges::any_of(_replay_backings, [&](ReplayBacking const &seen) { return seen.backing == handle.backing; })) {
                continue;
            }
            auto const &counts = *handle.backing->counters;
            _replay_backings.push_back(ReplayBacking{.tensor        = handle.name,
                                                     .backing       = handle.backing,
                                                     .bytes_read    = counts.bytes_read.load(std::memory_order_relaxed),
                                                     .bytes_written = counts.bytes_written.load(std::memory_order_relaxed),
                                                     .reads         = counts.reads.load(std::memory_order_relaxed),
                                                     .writes        = counts.writes.load(std::memory_order_relaxed),
                                                     .publishes     = counts.publishes.load(std::memory_order_relaxed)});
        }
    };
    collect(*this);
    std::as_const(*this).for_each_descendant(collect);
}

void Graph::contribute_run_samples(bool outermost) {
    std::vector<detail::RunSample> samples;
    samples.reserve(_timing_samples.size());
    Graph const *const self = outermost ? nullptr : this;
    for (auto const &sample : _timing_samples) {
        samples.push_back(
            detail::RunSample{.graph = self, .id = sample.id, .kind = sample.kind, .ms = sample.duration_ms, .width = sample.width});
    }
    _run_context->add_samples(std::move(samples));
}

void Graph::end_run_metrics(double wall_ms) {
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
    _last_resident_bytes = resident_bytes();
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
                              _name, held, _last_peak_bytes, _last_resident_bytes, budget);
        }
    }
}

std::uint64_t Graph::resident_bytes() const {
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
    lifecycle(*this);

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
    count(*this);
    return total;
}

RunMetrics const &Graph::last_run() const {
    if (_last_run_valid) {
        return _last_run;
    }
    // Labels and graph names are attached here, once per report, not on the replay path.
    std::unordered_map<Graph const *, std::unordered_map<NodeId, std::string_view>> labels;
    auto const label_of = [&](Graph const &graph, NodeId id) -> std::string_view {
        auto [at, added] = labels.try_emplace(&graph);
        if (added) {
            for (auto const &node : graph.nodes()) {
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
        Graph const &graph        = sample.graph != nullptr ? *sample.graph : *this;
        auto const [at, new_node] = index.try_emplace({&graph, sample.id}, out.nodes.size());
        if (new_node) {
            out.nodes.push_back(NodeRun{.graph = graph.name(),
                                        .id    = sample.id,
                                        .kind  = sample.kind,
                                        .label = std::string{label_of(graph, sample.id)},
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

std::vector<NodeTiming> const &Graph::timing_report() const {
    if (_timing_report_valid) {
        return _timing_report;
    }

    // Attach labels here rather than on the replay path: one string copy per
    // node per REPORT instead of per node per execute().
    std::unordered_map<NodeId, std::string_view> labels;
    labels.reserve(_nodes.size());
    for (auto const &node : _nodes) {
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

void Graph::print_timing_report(std::ostream &os) const {
    auto const &report = timing_report();
    if (report.empty()) {
        os << "No timing data available. Call execute() first.\n";
        return;
    }

    // Sort by duration descending
    auto sorted = report;
    std::ranges::sort(sorted, [](auto const &a, auto const &b) { return a.duration_ms > b.duration_ms; });

    double total = 0.0;
    for (auto const &t : sorted)
        total += t.duration_ms;

    os << fmt::format("Timing report for graph '{}' ({} nodes, {:.3f} ms total):\n", _name, sorted.size(), total);
    for (auto const &t : sorted) {
        double pct = (total > 0) ? 100.0 * t.duration_ms / total : 0.0;
        os << fmt::format("  {:8.3f} ms ({:5.1f}%)  [{}] {} ({})\n", t.duration_ms, pct, t.id, t.label, t.kind);
    }
}

std::vector<std::uint32_t> Graph::node_zone_ids() {
    if (!_profile_strings_valid || _profile_strings.size() != _nodes.size()) {
        rebuild_profile_strings();
    }
    std::vector<std::uint32_t> ids;
    ids.reserve(_profile_strings.size());
    for (auto const &entry : _profile_strings) {
        ids.push_back(entry.zone_id);
    }
    return ids;
}

void Graph::rebuild_profile_strings() {
    _exec_zone_id = waggle::intern_string(fmt::format("ComputeGraph::execute({})", _name));
    _profile_strings.clear();
    _profile_strings.resize(_nodes.size());

    for (size_t idx = 0; idx < _nodes.size(); idx++) {
        auto const         &node  = _nodes[idx];
        NodeProfileStrings &entry = _profile_strings[idx];
        entry.node_id             = node.id;
        entry.zone_id             = waggle::intern_string(fmt::format("graph:{}/{}", _name, node.label));

        auto text = [&entry](std::string_view key, std::string_view value) {
            entry.texts.emplace_back(waggle::intern_string(key), waggle::intern_string(value));
        };
        auto number = [&entry](std::string_view key, int64_t value) { entry.numbers.emplace_back(waggle::intern_string(key), value); };

        if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>()) {
            number("transfer_bytes", static_cast<int64_t>(tdesc->size_bytes));
        }

        if (auto const *desc = node.op_data.get_if<EinsumDescriptor>()) {
            text("c_prefactor", to_string(live_c_prefactor(*desc)));
            text("ab_prefactor", to_string(live_ab_prefactor(*desc)));
            if (auto const lists = live_index_lists(*desc); !lists.c.empty()) {
                text("c_indices", fmt::format("{}", fmt::join(lists.c, ",")));
                text("a_indices", fmt::format("{}", fmt::join(lists.a, ",")));
                text("b_indices", fmt::format("{}", fmt::join(lists.b, ",")));
            }
        } else if (auto const *sdesc = node.op_data.get_if<ScaleDescriptor>()) {
            entry.reals.emplace_back(waggle::intern_string("scale_factor"), as_real<double>(sdesc->factor));
        } else if (auto const *cdesc = node.op_data.get_if<CommDescriptor>()) {
            number("comm_bytes", static_cast<int64_t>(cdesc->size_bytes));
            number("comm_tensor", static_cast<int64_t>(cdesc->tensor_id));
        }

        // A per-operand annotation is worth having for the handful of operands an
        // ordinary node carries, and is worth having for none of the operands of
        // a batched one. The strings are built once, but they are RE-EMITTED on
        // every replay: a 2048-member batched GEMM is 6144 annotation events per
        // execution, which measured 2.55x on the node and wrote a 191 KB report
        // line that nobody can read. Past the threshold the batch is described
        // rather than enumerated - which is the information anyway, since a
        // batch's members agree on their shape by construction.
        constexpr size_t kMaxOperandAnnotations = 16;

        auto annotate_tensors = [&](std::vector<TensorId> const &ids, char const *prefix) {
            if (ids.size() > kMaxOperandAnnotations) {
                number(fmt::format("{}.count", prefix), static_cast<int64_t>(ids.size()));
                std::vector<size_t> const *shared  = nullptr;
                bool                       uniform = true;
                for (TensorId const tid : ids) {
                    auto const *handle = find_tensor(tid);
                    if (handle == nullptr || handle->dims.empty()) {
                        continue;
                    }
                    if (shared == nullptr) {
                        shared = &handle->dims;
                    } else if (handle->dims != *shared) {
                        uniform = false;
                        break;
                    }
                }
                if (shared != nullptr) {
                    text(fmt::format("{}.shape", prefix), uniform ? fmt::format("{}", fmt::join(*shared, "x")) : "mixed");
                }
                return;
            }
            for (TensorId const tid : ids) {
                auto const *handle = find_tensor(tid);
                if (handle != nullptr && !handle->dims.empty()) {
                    text(fmt::format("{}.{}", prefix, handle->name), fmt::format("{}", fmt::join(handle->dims, "x")));
                    if (handle->is_distributed) {
                        text(fmt::format("{}.{}.distributed", prefix, handle->name), "true");
                    }
                }
            }
        };
        annotate_tensors(node.inputs, "input");
        annotate_tensors(node.outputs, "output");

        if (node.estimated_flops > 0) {
            number("estimated_flops", static_cast<int64_t>(node.estimated_flops));
        }
    }

    _profile_strings_valid = true;
}

void Graph::print_dot(std::ostream &os) const {
    os << "digraph \"" << _name << "\" {\n";
    os << "  rankdir=TB;\n";

    // Tensor nodes (rectangles)
    for (auto const &[id, handle] : _tensors) {
        os << fmt::format("  T{} [shape=box, label=\"{}\\n", id, handle.name);
        if (!handle.dims.empty()) {
            os << fmt::format("({})", fmt::join(handle.dims, "x"));
        }
        os << "\"];\n";
    }

    // Operation nodes (ellipses, colored by target/type)
    for (auto const &node : _nodes) {
        std::string style;
        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            style = ", style=filled, fillcolor=\"#FFA500\""; // orange for transfers
        } else if (node.target == Target::GPU) {
            style = ", style=filled, fillcolor=\"#6495ED\""; // cornflower blue for GPU
        }

        std::string label = node.label;
        if (auto const *desc = node.op_data.get_if<TransferDescriptor>()) {
            label += fmt::format("\\n({} bytes)", desc->size_bytes);
        }

        os << fmt::format("  N{} [shape=ellipse, label=\"{}\"{}];\n", node.id, label, style);

        for (auto tid : node.inputs) {
            os << fmt::format("  T{} -> N{};\n", tid, node.id);
        }
        for (auto tid : node.outputs) {
            os << fmt::format("  N{} -> T{};\n", node.id, tid);
        }
    }

    os << "}\n";
}

std::string Graph::to_dot() const {
    std::ostringstream out;
    print_dot(out);
    return out.str();
}

std::string Graph::summary() const {
    std::ostringstream out;
    print_summary(out);
    return out.str();
}

void Graph::print_summary(std::ostream &os) const {
    // The names an operand list reads as, with the placeholder for an id this graph has no
    // handle for. Both lists print the same way, so the join is written once.
    auto const name_list = [this](std::vector<TensorId> const &ids) {
        std::vector<std::string_view> names;
        names.reserve(ids.size());
        for (auto const tid : ids) {
            auto const *handle = find_tensor(tid);
            names.emplace_back(handle != nullptr ? std::string_view{handle->name} : std::string_view{"?"});
        }
        return fmt::format("{}", fmt::join(names, ", "));
    };

    os << fmt::format("Graph '{}': {} nodes, {} tensors\n", _name, _nodes.size(), _tensors.size());
    for (auto const &node : _nodes) {
        os << fmt::format("  [{}] {} ({})\n", node.id, node.label, node.kind);
        if (!node.inputs.empty()) {
            os << fmt::format("    inputs: {}\n", name_list(node.inputs));
        }
        if (!node.outputs.empty()) {
            os << fmt::format("    outputs: {}\n", name_list(node.outputs));
        }
    }
}

// ── JSON serialization ─────────────────────────────────────────────────────

std::string Graph::to_json() const {
    std::scoped_lock const lock(_content_mutex);
    // Build a ComputeGraphData struct from internal state, then serialize it.
    // This is cleaner than manual JSON string building and uses the shared types
    // that the viewer also understands.

    ComputeGraphData data;
    data.name           = _name;
    data.pipeline_name  = _pipeline_name;
    data.workspace_name = _workspace_name;
    data.stage_name     = _stage_name;
    data.stage_type     = _stage_type;
    data.stage_index    = _stage_index;

    // Tensors
    for (auto const &[id, h] : _tensors) {
        GraphTensorData td;
        td.id              = id;
        td.name            = h.name;
        td.rank            = h.rank;
        td.dims            = h.dims;
        td.element_size    = h.element_size;
        td.dtype           = scalar_type_name(h.dtype);
        td.is_intermediate = h.is_intermediate;
        data.tensors.push_back(std::move(td));
    }

    // Build timing lookup
    std::unordered_map<NodeId, double> timing_map;
    for (auto const &t : _timing_samples)
        timing_map[t.id] = t.duration_ms;

    // Nodes: use array index as ID
    for (size_t ni = 0; ni < _nodes.size(); ni++) {
        auto const   &node = _nodes[ni];
        GraphNodeData nd;
        nd.id        = ni;
        nd.kind      = std::string(op_kind_name(node.kind));
        nd.label     = node.label;
        nd.target    = (node.target == Target::GPU) ? "GPU" : "CPU";
        nd.stream_id = node.stream_id;
        nd.inputs    = node.inputs;
        nd.outputs   = node.outputs;

        // Timing
        auto tit = timing_map.find(node.id);
        if (tit != timing_map.end())
            nd.timing_ms = tit->second;

        // Operation-specific data
        if (auto const *desc = node.op_data.get_if<EinsumDescriptor>()) {
            // GraphNodeData is a viewer-facing snapshot; project complex
            // prefactors to their real part for display. The live value is
            // preserved on the descriptor itself.
            auto const lists = live_index_lists(*desc);
            nd.c_prefactor   = as_real<double>(live_c_prefactor(*desc));
            nd.ab_prefactor  = as_real<double>(live_ab_prefactor(*desc));
            nd.c_indices     = fmt::format("{}", fmt::join(lists.c, ","));
            nd.a_indices     = fmt::format("{}", fmt::join(lists.a, ","));
            nd.b_indices     = fmt::format("{}", fmt::join(lists.b, ","));
            nd.conj_a        = live_conj_a(*desc);
            nd.conj_b        = live_conj_b(*desc);
        } else if (auto const *desc = node.op_data.get_if<ScaleDescriptor>()) {
            // Same viewer-facing projection as the einsum prefactors above.
            nd.scale_factor = as_real<double>(desc->factor);
        } else if (auto const *desc = node.op_data.get_if<PermuteDescriptor>()) {
            // Same viewer-facing projection as the einsum prefactors above.
            nd.alpha     = desc->alpha.real();
            nd.beta      = desc->beta.real();
            nd.c_indices = fmt::format("{}", fmt::join(desc->c_indices, ","));
            nd.a_indices = fmt::format("{}", fmt::join(desc->a_indices, ","));
        }

        data.nodes.push_back(std::move(nd));
    }

    // Dependency edges
    {
        // Key by the owning buffer: a write through a view of T and a read of T
        // (or of another view of it) share storage, so their dependency edge is
        // only found once aliases resolve to the same id -- the same reason the
        // schedulers resolve_alias before deriving edges.
        std::unordered_map<TensorId, std::vector<size_t>> writers, readers;
        for (size_t i = 0; i < _nodes.size(); i++) {
            for (auto in_tid : _nodes[i].inputs)
                readers[resolve_alias(in_tid)].push_back(i);
            for (auto out_tid : _nodes[i].outputs)
                writers[resolve_alias(out_tid)].push_back(i);
        }

        std::set<std::pair<size_t, size_t>> emitted;
        for (auto const &[tid, reader_list] : readers) {
            auto wit = writers.find(tid);
            if (wit == writers.end())
                continue;

            for (size_t const r : reader_list) {
                size_t best_w = SIZE_MAX;
                for (size_t const w : wit->second)
                    if (w < r && w != r && (best_w == SIZE_MAX || w > best_w))
                        best_w = w;
                if (best_w == SIZE_MAX)
                    for (size_t const w : wit->second)
                        if (w > r && w != r && (best_w == SIZE_MAX || w > best_w))
                            best_w = w;
                if (best_w == SIZE_MAX || best_w == r)
                    continue;

                if (emitted.count({best_w, r}))
                    continue;
                emitted.insert({best_w, r});

                GraphEdgeData edge;
                edge.from      = best_w;
                edge.to        = r;
                edge.tensor_id = tid;
                edge.loop_back = (best_w > r);
                data.edges.push_back(edge);
            }
        }
    }

    // Serialize to JSON manually (structured, no Glaze dependency in library)
    std::string j;
    j.reserve(4096);

    auto esc = [](std::string const &s) -> std::string { return json_escape(s); };

    j += R"({"name":")" + esc(data.name) + "\"";
    if (!data.pipeline_name.empty())
        j += R"(,"pipeline_name":")" + esc(data.pipeline_name) + "\"";
    if (!data.workspace_name.empty())
        j += R"(,"workspace_name":")" + esc(data.workspace_name) + "\"";
    if (!data.stage_name.empty())
        j += R"(,"stage_name":")" + esc(data.stage_name) + "\"";
    if (!data.stage_type.empty())
        j += R"(,"stage_type":")" + esc(data.stage_type) + "\"";
    if (data.stage_index >= 0)
        j += ",\"stage_index\":" + std::to_string(data.stage_index);

    j += ",\"tensors\":[";
    for (size_t i = 0; i < data.tensors.size(); i++) {
        auto const &t = data.tensors[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"id":{},"name":"{}","rank":{},"dims":[)", t.id, esc(t.name), t.rank);
        j += fmt::format("{}", fmt::join(t.dims, ","));
        j += fmt::format(R"(],"element_size":{},"dtype":"{}","is_intermediate":{}}})", t.element_size, esc(t.dtype),
                         t.is_intermediate ? "true" : "false");
    }
    j += "]";

    j += ",\"nodes\":[";
    for (size_t i = 0; i < data.nodes.size(); i++) {
        auto const &n = data.nodes[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"id":{},"kind":"{}","label":"{}","target":"{}","stream_id":{})", n.id, esc(n.kind), esc(n.label),
                         esc(n.target), n.stream_id);
        j += fmt::format(R"(,"inputs":[{}])", fmt::join(n.inputs, ","));
        j += fmt::format(R"(,"outputs":[{}])", fmt::join(n.outputs, ","));
        if (n.timing_ms >= 0)
            j += fmt::format(",\"timing_ms\":{:.6f}", n.timing_ms);
        if (n.c_prefactor != 0.0 || n.ab_prefactor != 1.0)
            j += fmt::format(R"(,"c_prefactor":{},"ab_prefactor":{})", n.c_prefactor, n.ab_prefactor);
        if (n.scale_factor != 1.0)
            j += fmt::format(",\"scale_factor\":{}", n.scale_factor);
        if (n.alpha != 1.0 || n.beta != 0.0)
            j += fmt::format(R"(,"alpha":{},"beta":{})", n.alpha, n.beta);
        if (!n.c_indices.empty())
            j += R"(,"c_indices":")" + esc(n.c_indices) + "\"";
        if (!n.a_indices.empty())
            j += R"(,"a_indices":")" + esc(n.a_indices) + "\"";
        if (!n.b_indices.empty())
            j += R"(,"b_indices":")" + esc(n.b_indices) + "\"";
        if (n.conj_a)
            j += ",\"conj_a\":true";
        if (n.conj_b)
            j += ",\"conj_b\":true";
        j += "}";
    }
    j += "]";

    j += ",\"edges\":[";
    for (size_t i = 0; i < data.edges.size(); i++) {
        auto const &e = data.edges[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"from":{},"to":{},"tensor_id":{})", e.from, e.to, e.tensor_id);
        if (e.loop_back)
            j += ",\"loop_back\":true";
        j += "}";
    }
    j += "]}";

    return j;
}

// ── Global graph registry ──────────────────────────────────────────────────

namespace {

std::mutex           g_registry_mutex;
std::vector<Graph *> g_registered_graphs;

} // namespace

void register_graph(Graph *graph) {
#if defined(EINSUMS_HAVE_PROFILER)
    // On first registration, give the viewer and session files the graphs. Outside the registry
    // lock: both callbacks take it while the profiler holds its handler table's lock, so taking
    // the two in the other order here could deadlock.
    static std::once_flag profiler_wired;
    std::call_once(profiler_wired, [] {
        waggle::register_handler("get_compute_graphs", [](std::string const &) { return registered_graphs_json(); });
        waggle::register_session_section("einsums.compute_graphs", [] {
            // The bare array of graphs, as session files have always held it.
            std::string const json  = registered_graphs_json();
            auto const        first = json.find('[');
            auto const        last  = json.rfind(']');
            return first != std::string::npos && last != std::string::npos && last > first ? json.substr(first, last - first + 1)
                                                                                           : std::string("[]");
        });
    });
#endif

    std::scoped_lock const lock(g_registry_mutex);

    // Replace if same name already registered
    for (auto &g : g_registered_graphs) {
        if (g->name() == graph->name()) {
            g = graph;
            return;
        }
    }
    g_registered_graphs.push_back(graph);
}

namespace {
/// Cache of graph JSON for graphs that have been destroyed, keyed by graph
/// name with the same replace-on-collision rule as the live registry. This
/// allows export_session() to include graph data even after the Graph objects
/// go out of scope. Keying by name is what bounds it: a run that builds and
/// destroys a same-named graph repeatedly re-caches one entry instead of
/// appending forever.
std::vector<std::pair<std::string, std::string>> g_cached_graph_jsons;

/// Whether anything can ever read a dead graph's JSON: the shutdown exporter
/// (--einsums:profile:save) or an attached viewer client. Serializing a graph
/// is O(nodes) string building and runs in the destructor, inside whatever
/// phase happens to drop the graph - measured at 41 ms of a 345 ms DLPNO
/// transform phase before it was gated - so it must not happen on the default
/// path, where the profiler server listens but nobody is collecting. A viewer
/// that attaches later still sees every live graph; what it loses is only the
/// post-mortem record of graphs that died before it connected.
bool graph_json_cache_wanted() {
#if defined(EINSUMS_HAVE_PROFILER)
    if (!waggle::enabled())
        return false;
    // Read the settings each time rather than latching them: a graph dies
    // once, so this is not a hot path, and the tests change the setting.
    if (!waggle::settings().save.empty())
        return true;
    return waggle::viewer_connected();
#else
    return false;
#endif
}
} // namespace

void transfer_graph_registration(Graph const *from, Graph *to) {
    std::scoped_lock const lock(g_registry_mutex);
    if (auto it = std::ranges::find(g_registered_graphs, from); it != g_registered_graphs.end()) {
        *it = to;
    }
}

void unregister_graph(Graph *graph) {
    std::scoped_lock const lock(g_registry_mutex);

    auto it = std::ranges::find(g_registered_graphs, graph);
    if (it == g_registered_graphs.end())
        return;
    g_registered_graphs.erase(it);

    if (!graph_json_cache_wanted())
        return;

    // Cache the graph's JSON before it is gone, so it survives destruction.
    auto cached = std::ranges::find_if(g_cached_graph_jsons, [&](auto const &entry) { return entry.first == graph->name(); });
    if (cached != g_cached_graph_jsons.end()) {
        cached->second = graph->to_json();
    } else {
        g_cached_graph_jsons.emplace_back(graph->name(), graph->to_json());
    }
}

std::string registered_graphs_json() {
    std::scoped_lock const lock(g_registry_mutex);
    std::string            result = "{\"graphs\":[";
    bool                   first  = true;

    // Include live graphs.
    for (auto *g : g_registered_graphs) {
        if (!first)
            result += ",";
        first = false;
        result += g->to_json();
    }

    // Include cached JSON from destroyed graphs, skipping any name a live
    // graph has since reclaimed - the live one is the current truth.
    for (auto const &[name, json] : g_cached_graph_jsons) {
        if (std::ranges::any_of(g_registered_graphs, [&](Graph const *g) { return g->name() == name; }))
            continue;
        if (!first)
            result += ",";
        first = false;
        result += json;
    }

    result += "]}";
    return result;
}

EINSUMS_NAMESPACE_END(compute_graph)
