//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GraphIR.cpp
/// @brief The ``einsums_graph_ir`` public entry points, and the member walk.
///
/// The schema itself is documented once, in ``GraphIR.hpp``. The implementation
/// of it is three passes that meet in ``GraphIR/Common.hpp``, which says how
/// they fit together:
///
///  1. ``GraphIR/Write.cpp``  a graph, rendered as a document.
///  2. ``GraphIR/Read.cpp``   a document, read into a plain intermediate form,
///     collecting EVERY problem rather than stopping at the first.
///  3. ``GraphIR/Build.cpp``  a clean intermediate form, turned into a graph.
///
/// This file is what callers reach: `save_graph`, `load_graph`,
/// `validate_graph_ir`, `Graph::content_hash`, and the Python spelling of each.
/// It is also where the reader-builder split pays off, in two functions that
/// differ only in what they do with the problem list: `load_graph` reports the
/// first, `validate_graph_ir` reports all of them.
///
/// @par The member walk (every member of detail::GraphState)
/// Saved, because it is structure:
///   `_name`, `_nodes` (kind, label, inputs, outputs, descriptor), `_tensors`
///   (name, dtype, rank, dims, dim symbols, spaces, intermediate flag, scope,
///   init kind, alias parent), `_symbol_spaces`, `_params`' entries,
///   `_named_gate_flags`, `_slot_redirects`.
///
/// NOT saved, because it is machine-dependent tuning that a load re-derives:
///   `Node::thread_width`, `Node::admission_priority`, `Node::stream_id`,
///   `Node::estimated_flops`, `Node::estimated_bytes`, `_planned_thread_count`,
///   `_plan_trial` and the two plan snapshots, `_timing_samples`,
///   `_timing_report`, `_last_optimize_report`.
///
/// NOT saved, because it is a live-process resource with no meaning in a file:
///   every `std::function` on a `TensorHandle`, `_owned_tensors`,
///   `_adopted_cleanups`, `_slot_map`, `_ptr_index`, `_device_shadows`,
///   `_executor`, `_content_mutex`, `_pending_binds`,
///   `_deps` / `_usage` and the version counters that guard them,
///   `_profile_strings`, `_bound_operands`, `_ragged_extents`, `_space_extents`,
///   `_space_tiles`, `_scope_maps`. `_ragged_extents`, `_bound_operands` and the
///   two space maps are bind-time state about the CALLER's problem (what each
///   space measures, how it is partitioned), and a loaded graph is bound afresh.
///
/// `_space_registry` is a non-owning pointer into a registry the caller owns, so
/// what travels is the space NAMES; a loaded graph resolves them against the
/// registry it is given.

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "GraphIR/Common.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// ── content_hash ───────────────────────────────────────────────────────────

std::uint64_t Graph::content_hash() const {
    std::string canonical;
    try {
        canonical = json::emit(json::Value{graph_ir::write_structure(*this)}, json::EmitOptions{.style = json::EmitStyle::Canonical});
    } catch (graph_ir::SaveRefusal const &refusal) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph::content_hash: {}", refusal.what());
    }
    // FNV-1a over the canonical bytes. Not a cryptographic digest and not meant
    // to be: the property wanted is that a structural change moves it, which is
    // what a differential test and a cache key need.
    std::uint64_t hash = 14695981039346656037ULL; // the 64-bit FNV offset basis
    for (char const byte : canonical) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ── save ───────────────────────────────────────────────────────────────────

expected<std::string, GraphError> save_graph_string(Graph const &graph, SaveOptions const &options) {
    try {
        json::Value const document = graph_ir::write_document(graph, options);
        return json::emit(document, json::EmitOptions{.style = options.pretty ? json::EmitStyle::Pretty : json::EmitStyle::Canonical});
    } catch (graph_ir::SaveRefusal const &refusal) {
        return unexpected(GraphError::validation(fmt::format("save_graph: {}", refusal.what())));
    } catch (std::exception const &error) {
        return unexpected(GraphError::validation(fmt::format("save_graph: {}", error.what())));
    }
}

expected<void, GraphError> save_graph(Graph const &graph, std::string const &path, SaveOptions const &options) {
    auto text = save_graph_string(graph, options);
    if (!text) {
        return unexpected(text.error());
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return unexpected(GraphError::io(fmt::format("save_graph: cannot open '{}' for writing", path)));
    }
    file << *text;
    if (!file) {
        return unexpected(GraphError::io(fmt::format("save_graph: writing '{}' failed", path)));
    }
    return {};
}

// ── load and validate ──────────────────────────────────────────────────────

namespace {

/// Read @p path whole, or say why not.
expected<std::string, GraphError> read_file(std::string const &path, std::string_view who) {
    std::ifstream const file(path, std::ios::binary);
    if (!file) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", who, path)));
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (file.bad()) {
        return unexpected(GraphError::io(fmt::format("{}: reading '{}' failed", who, path)));
    }
    return buffer.str();
}

/// Parse and read @p text, returning the intermediate form and every problem.
///
/// The unconsumed-key audit runs LAST and over the whole document, because a
/// leftover key is only leftover once every reader has had its turn.
graph_ir::IrDocument inspect(std::string_view text, graph_ir::Problems &problems, json::Value &document, SpaceRegistry const &registry) {
    auto parsed = json::parse(text);
    if (!parsed) {
        problems.push_back(parsed.error().to_string());
        return {};
    }
    document                 = std::move(*parsed);
    graph_ir::IrDocument out = graph_ir::read_document(document, problems, registry);

    std::vector<std::string> unconsumed;
    json::collect_unconsumed(document, "$", unconsumed);
    for (auto const &key : unconsumed) {
        problems.push_back(fmt::format("{}: this build does not understand that key. A field may be ADDED to the schema and a newer "
                                       "build then reads an older file, but a reader never silently ignores content",
                                       key));
    }
    return out;
}

} // namespace

expected<Graph, GraphError> load_graph_string(std::string_view text) {
    return load_graph_string(text, global_space_registry());
}

expected<Graph, GraphError> load_graph_string(std::string_view text, SpaceRegistry &registry) {
    graph_ir::Problems         problems;
    json::Value                document;
    graph_ir::IrDocument const ir = inspect(text, problems, document, registry);
    if (!problems.empty()) {
        return unexpected(GraphError::parse(fmt::format("load_graph: {}", problems.front())));
    }
    try {
        return graph_ir::build_graph(ir, registry);
    } catch (graph_ir::BuildFailure const &failure) {
        return unexpected(GraphError::validation(fmt::format("load_graph: {}", failure.what())));
    } catch (std::exception const &error) {
        return unexpected(GraphError::validation(fmt::format("load_graph: {}", error.what())));
    }
}

expected<Graph, GraphError> load_graph(std::string const &path) {
    return load_graph(path, global_space_registry());
}

expected<Graph, GraphError> load_graph(std::string const &path, SpaceRegistry &registry) {
    auto text = read_file(path, "load_graph");
    if (!text) {
        return unexpected(text.error());
    }
    auto graph = load_graph_string(*text, registry);
    if (!graph) {
        return unexpected(GraphError{.kind = graph.error().kind, .message = fmt::format("{} (reading '{}')", graph.error().message, path)});
    }
    return graph;
}

expected<void, GraphError> validate_graph_ir_string(std::string_view text) {
    return validate_graph_ir_string(text, global_space_registry());
}

expected<void, GraphError> validate_graph_ir_string(std::string_view text, SpaceRegistry &registry) {
    graph_ir::Problems         problems;
    json::Value                document;
    graph_ir::IrDocument const ir = inspect(text, problems, document, registry);
    if (problems.empty()) {
        // A document that reads clean still has to BUILD, and a build failure is
        // one more problem to report rather than a separate outcome.
        try {
            Graph const graph = graph_ir::build_graph(ir, registry);
            (void)graph;
        } catch (std::exception const &error) {
            problems.emplace_back(error.what());
        }
    }
    if (problems.empty()) {
        return {};
    }
    std::string message = fmt::format("validate_graph_ir: {} problem(s)", problems.size());
    for (auto const &problem : problems) {
        message += "\n  " + problem;
    }
    return unexpected(GraphError::parse(std::move(message)));
}

expected<void, GraphError> validate_graph_ir(std::string const &path) {
    return validate_graph_ir(path, global_space_registry());
}

expected<void, GraphError> validate_graph_ir(std::string const &path, SpaceRegistry &registry) {
    auto text = read_file(path, "validate_graph_ir");
    if (!text) {
        return unexpected(text.error());
    }
    auto done = validate_graph_ir_string(*text, registry);
    if (!done) {
        return unexpected(GraphError{.kind = done.error().kind, .message = fmt::format("{} (reading '{}')", done.error().message, path)});
    }
    return done;
}

// ── The Python spelling ────────────────────────────────────────────────────

void save_graph_file(Graph const &graph, std::string const &path) {
    if (auto const done = save_graph(graph, path); !done) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", done.error().message);
    }
}

Graph *load_graph_file(std::string const &path) {
    return load_graph_file_into(path, global_space_registry());
}

Graph *load_graph_file_into(std::string const &path, SpaceRegistry &registry) {
    auto graph = load_graph(path, registry);
    if (!graph) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", graph.error().message);
    }
    // A raw pointer because the Python binding takes ownership of it; there is
    // no C++ caller for this overload.
    return new Graph(std::move(*graph)); // NOLINT(cppcoreguidelines-owning-memory)
}

void validate_graph_ir_file(std::string const &path) {
    if (auto const done = validate_graph_ir(path); !done) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", done.error().message);
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
