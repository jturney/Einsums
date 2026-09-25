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

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp> // For OptimizerPass and PassManager
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraphTypes/GraphData.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
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
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::vector<Graph::NodeTiming> const &Graph::timing_report() const {
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
        os << fmt::format("  {:8.3f} ms ({:5.1f}%)  [{}] {} ({})\n", t.duration_ms, pct, t.id, t.label, op_kind_name(t.kind));
    }
}

void Graph::rebuild_profile_strings() {
    _exec_zone_name = fmt::format("ComputeGraph::execute({})", _name);
    _exec_zone_id   = profile::intern_string(_exec_zone_name);
    _profile_strings.clear();
    _profile_strings.resize(_nodes.size());

    for (size_t idx = 0; idx < _nodes.size(); idx++) {
        auto const         &node  = _nodes[idx];
        NodeProfileStrings &entry = _profile_strings[idx];
        entry.node_id             = node.id;
        entry.zone                = fmt::format("graph:{}/{}", _name, node.label);
        entry.zone_id             = profile::intern_string(entry.zone);

        auto text = [&entry](std::string_view key, std::string_view value) {
            entry.texts.emplace_back(profile::intern_string(key), profile::intern_string(value));
        };
        auto number = [&entry](std::string_view key, int64_t value) { entry.numbers.emplace_back(profile::intern_string(key), value); };

        if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>()) {
            number("transfer_bytes", static_cast<int64_t>(tdesc->size_bytes));
        }

        if (auto const *desc = node.op_data.get_if<EinsumDescriptor>()) {
            text("c_prefactor", to_string(desc->c_prefactor));
            text("ab_prefactor", to_string(desc->ab_prefactor));
            if (!desc->spec.c_indices.empty()) {
                text("c_indices", fmt::format("{}", fmt::join(desc->spec.c_indices, ",")));
                text("a_indices", fmt::format("{}", fmt::join(desc->spec.a_indices, ",")));
                text("b_indices", fmt::format("{}", fmt::join(desc->spec.b_indices, ",")));
            }
        } else if (auto const *sdesc = node.op_data.get_if<ScaleDescriptor>()) {
            entry.reals.emplace_back(profile::intern_string("scale_factor"), as_real<double>(sdesc->factor));
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
                    auto it = _tensors.find(tid);
                    if (it == _tensors.end() || it->second.dims.empty()) {
                        continue;
                    }
                    if (shared == nullptr) {
                        shared = &it->second.dims;
                    } else if (it->second.dims != *shared) {
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
                auto it = _tensors.find(tid);
                if (it != _tensors.end() && !it->second.dims.empty()) {
                    text(fmt::format("{}.{}", prefix, it->second.name), fmt::format("{}", fmt::join(it->second.dims, "x")));
                    if (it->second.is_distributed) {
                        text(fmt::format("{}.{}.distributed", prefix, it->second.name), "true");
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

void Graph::print_summary(std::ostream &os) const {
    // The names an operand list reads as, with the placeholder for an id this graph has no
    // handle for. Both lists print the same way, so the join is written once.
    auto const name_list = [this](std::vector<TensorId> const &ids) {
        std::vector<std::string_view> names;
        names.reserve(ids.size());
        for (auto const tid : ids) {
            auto const it = _tensors.find(tid);
            names.emplace_back(it != _tensors.end() ? std::string_view{it->second.name} : std::string_view{"?"});
        }
        return fmt::format("{}", fmt::join(names, ", "));
    };

    os << fmt::format("Graph '{}': {} nodes, {} tensors\n", _name, _nodes.size(), _tensors.size());
    for (auto const &node : _nodes) {
        os << fmt::format("  [{}] {} ({})\n", node.id, node.label, op_kind_name(node.kind));
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
    std::scoped_lock const lock(*_content_mutex);
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
            nd.c_prefactor  = as_real<double>(desc->c_prefactor);
            nd.ab_prefactor = as_real<double>(desc->ab_prefactor);
            nd.c_indices    = fmt::format("{}", fmt::join(desc->spec.c_indices, ","));
            nd.a_indices    = fmt::format("{}", fmt::join(desc->spec.a_indices, ","));
            nd.b_indices    = fmt::format("{}", fmt::join(desc->spec.b_indices, ","));
            nd.conj_a       = desc->conj_a;
            nd.conj_b       = desc->conj_b;
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
    std::scoped_lock const lock(g_registry_mutex);

    // On first registration, wire up the profiler handler
    static bool handler_registered = false;
    if (!handler_registered) {
#if defined(EINSUMS_HAVE_PROFILER)
        auto *srv = profile::Profiler::instance().server();
        if (srv) {
            srv->register_handler("get_compute_graphs", [](std::string const &) { return registered_graphs_json(); });
        }
#endif
        handler_registered = true;
    }

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
    auto &prof = profile::Profiler::instance();
    if (!prof.enabled())
        return false;
    // Read the config each time rather than latching it: a graph dies once,
    // so this is not a hot path, and the tests flip the key at runtime.
    bool save_configured = false;
    try {
        save_configured = !config::get(option::ProfileSave).empty();
    } catch (...) { // NOLINT
    }
    if (save_configured)
        return true;
    auto const *srv = prof.server();
    return srv != nullptr && srv->has_client();
#else
    return false;
#endif
}
} // namespace

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
