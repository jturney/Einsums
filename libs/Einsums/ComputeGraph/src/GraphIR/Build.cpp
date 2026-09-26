//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Build.cpp
/// @brief The intermediate form, turned back into a graph.
///
/// The last of the three passes described in `Common.hpp`, and the only one that
/// allocates. It runs on a document `Read.cpp` has already reported clean, so
/// what it refuses is a different class of thing: a document that PARSES but
/// whose pieces cannot be assembled, which it reports by throwing
/// @ref BuildFailure rather than by appending to a problem list.
///
/// The storage a load allocates is placeholder storage, and every deleter is
/// adopted by the ROOT graph so a loop body's tensors outlive the fragment that
/// named them. A caller replaces it slot by slot with `Graph::bind`; a slot
/// nobody binds still has something to execute against, which is what keeps an
/// unbound loaded graph from dereferencing nothing.

#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/ElementOps.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::graph_ir)

namespace {

/// One frame's loaded storage: the dense id's tensor object, its id in this
/// frame's graph, and whether it is a scalar.
struct LoadedTensor {
    void    *object{nullptr};
    TensorId id{0};
    bool     scalar{false};
};

/// Allocate placeholder storage for one tensor and register it.
///
/// The storage is the graph's own, and every deleter is adopted by the ROOT
/// graph so a body's tensors outlive the fragment that named them. A caller
/// replaces it slot by slot with @ref Graph::bind; a slot nobody binds still has
/// something to execute against, which is what keeps an unbound loaded graph
/// from dereferencing nothing.
template <typename T>
LoadedTensor allocate_tensor(Graph &root, Graph &graph, IrTensor const &spec) {
    if (spec.rank == 0) {
        auto *scalar = new T{};
        root.adopt([scalar]() { delete scalar; });
        TensorId const id = graph.register_tensor(make_scalar_handle(scalar, 0, spec.name));
        return LoadedTensor{.object = static_cast<void *>(scalar), .id = id, .scalar = true};
    }
    using TensorType = GeneralRuntimeTensor<T, std::allocator<T>>;

    // A tensor the file says is DEFERRED comes back deferred, with the lifecycle hooks its
    // declaration installed. Allocating it here instead would look harmless (the pass would
    // just find it already materialized) and would quietly cost the graph the one thing the
    // save exists to enable: only a deferred intermediate can be resized by a bind, so a
    // materialized one refuses the move to a different-sized problem.
    bool const deferred = spec.alloc == AllocState::Deferred;
    auto      *tensor =
        deferred ? new TensorType(typename TensorType::DeferredAlloc{}, spec.name, spec.dims) : new TensorType(spec.name, spec.dims);
    if (!deferred) {
        tensor->zero();
    }
    root.adopt([tensor]() { delete tensor; });
    auto handle            = make_handle(*tensor, 0);
    handle.is_intermediate = spec.intermediate;
    handle.ownership       = spec.scope;
    handle.init_kind       = spec.init;
    if (deferred) {
        handle.alloc_state        = AllocState::Deferred;
        handle.materialize_fn     = [tensor]() { tensor->materialize(); };
        handle.release_fn         = [tensor]() { tensor->release(); };
        handle.is_materialized_fn = [tensor]() { return tensor->is_materialized(); };
        handle.resize_deferred_fn = [tensor](std::vector<size_t> const &new_dims) { tensor->resize_deferred(new_dims); };
        handle.zero_fn            = make_zero_fn(tensor);
    }
    TensorId const id = graph.register_tensor(std::move(handle));
    graph.get_or_create_slot(*tensor, id);
    return LoadedTensor{.object = static_cast<void *>(tensor), .id = id, .scalar = false};
}

/// Register a handle in @p graph over storage an enclosing frame already owns.
template <typename T>
LoadedTensor adopt_outer_tensor(Graph &graph, IrTensor const &spec, LoadedTensor const &outer) {
    if (outer.scalar) {
        TensorId const id = graph.register_tensor(make_scalar_handle(static_cast<T *>(outer.object), 0, spec.name));
        return LoadedTensor{.object = outer.object, .id = id, .scalar = true};
    }
    using TensorType       = GeneralRuntimeTensor<T, std::allocator<T>>;
    auto *tensor           = static_cast<TensorType *>(outer.object);
    auto  handle           = make_handle(*tensor, 0);
    handle.is_intermediate = spec.intermediate;
    handle.ownership       = spec.scope;
    handle.init_kind       = spec.init;
    TensorId const id      = graph.register_tensor(std::move(handle));
    graph.get_or_create_slot(*tensor, id);
    return LoadedTensor{.object = outer.object, .id = id, .scalar = false};
}

/// Dispatch @p spec's dtype and allocate or adopt accordingly.
LoadedTensor materialize_tensor(Graph &root, Graph &graph, IrTensor const &spec, LoadedTensor const *outer) {
    if (spec.dtype == packed_gemm::ScalarType::Unknown) {
        throw BuildFailure(fmt::format("tensor '{}' has dtype 'unknown', which names no storage the loader can allocate", spec.name));
    }
    return detail::dispatch_scalar_type(spec.dtype, [&]<typename T>(T /*tag*/) {
        return outer != nullptr ? adopt_outer_tensor<T>(graph, spec, *outer) : allocate_tensor<T>(root, graph, spec);
    });
}

std::vector<LoadedTensor> build_frame(Graph &root, Graph &graph, std::vector<IrTensor> const &tensors, std::vector<IrNode> const &nodes,
                                      std::vector<LoadedTensor> const *parent, GateFlagTable const &gates, SpaceRegistry const &registry);

/// Turn one fragment into a live sub-graph.
///
/// The body shares the root's @ref ParamTable, which is what a captured body
/// does: a ``BoundExpr::Param`` inside a loop resolves against the pipeline's
/// parameters, not against a table private to the iteration.
// NOLINTNEXTLINE(misc-no-recursion): fragments nest.
std::shared_ptr<Graph> build_fragment(Graph &root, IrFragment const &fragment, std::vector<LoadedTensor> const &parent,
                                      GateFlagTable const &gates, SpaceRegistry const &registry) {
    auto body = std::make_shared<Graph>(fragment.name);
    body->set_params_ptr(root.params_ptr());
    build_frame(root, *body, fragment.tensors, fragment.nodes, &parent, gates, registry);
    return body;
}

/// Materialize one frame's tensors, rebuild its nodes, and recurse into any
/// fragment a control-flow node names.
// NOLINTNEXTLINE(misc-no-recursion): see build_fragment.
std::vector<LoadedTensor> build_frame(Graph &root, Graph &graph, std::vector<IrTensor> const &tensors, std::vector<IrNode> const &nodes,
                                      std::vector<LoadedTensor> const *parent, GateFlagTable const &gates, SpaceRegistry const &registry) {
    std::vector<LoadedTensor> loaded(tensors.size());
    for (auto const &spec : tensors) {
        if (spec.id >= tensors.size()) {
            throw BuildFailure(
                fmt::format("tensor id {} is outside the frame's {} tensors; ids must be dense from 0", spec.id, tensors.size()));
        }
        if (loaded[spec.id].id != 0) {
            throw BuildFailure(fmt::format("two tensors both claim id {}", spec.id));
        }
        LoadedTensor const *outer = nullptr;
        if (spec.outer.has_value()) {
            if (parent == nullptr) {
                throw BuildFailure(
                    fmt::format("tensor '{}' declares an outer reference, but the top-level frame has no enclosing one", spec.name));
            }
            if (*spec.outer >= parent->size()) {
                throw BuildFailure(
                    fmt::format("tensor '{}' references outer id {}, which the enclosing frame does not define", spec.name, *spec.outer));
            }
            outer = &(*parent)[*spec.outer];
        }
        loaded[spec.id] = materialize_tensor(root, graph, spec, outer);
    }

    // Annotations, spaces before symbols so a (symbol, space) tie is recorded
    // from whichever side names it first.
    for (auto const &spec : tensors) {
        if (spec.spaces.empty()) {
            continue;
        }
        // An EMPTY name is an axis nobody spoke for, which a partial annotation has and a
        // complete one never does. The two are applied through different calls because
        // annotate_spaces rightly refuses a hole: a caller handing over a whole vector has said
        // something about every axis, so a hole there is a mistake, and annotate_space_axis is
        // where one is made deliberately.
        bool const           partial = std::ranges::any_of(spec.spaces, [](std::string const &name) { return name.empty(); });
        std::vector<SpaceId> ids;
        ids.reserve(spec.spaces.size());
        for (auto const &name : spec.spaces) {
            if (partial && name.empty()) {
                ids.emplace_back();
                continue;
            }
            auto const id = registry.find(name);
            if (!id.has_value()) {
                throw BuildFailure(fmt::format("tensor '{}' names index space '{}', which is not registered", spec.name, name));
            }
            ids.push_back(*id);
        }
        if (partial) {
            for (std::size_t axis = 0; axis < ids.size(); ++axis) {
                if (ids[axis].valid()) {
                    graph.annotate_space_axis(loaded[spec.id].id, axis, ids[axis]);
                }
            }
        } else {
            graph.annotate_spaces(loaded[spec.id].id, std::move(ids));
        }
        // annotate_spaces records a DECLARATION; the file says whether the
        // original was one, and a derived annotation must read back as derived
        // so a validation pass still reports the weaker verdict on it.
        graph.tensor(loaded[spec.id].id).spaces_inferred = spec.spaces_inferred;
    }
    for (auto const &spec : tensors) {
        if (!spec.dim_symbols.empty()) {
            graph.annotate_dims(loaded[spec.id].id, spec.dim_symbols);
        }
    }

    // Provenance last, and independent of the two above: a tag is a statement about what the
    // tensor IS, and it neither constrains nor is constrained by extents or spaces.
    for (auto const &spec : tensors) {
        if (spec.tag.valid()) {
            graph.annotate_tag(loaded[spec.id].id, spec.tag);
        }
    }

    for (auto const &spec : nodes) {
        Node node;
        node.id    = graph.reserve_node_id();
        node.kind  = spec.kind;
        node.label = spec.label;

        auto const remap = [&](std::vector<std::size_t> const &dense, char const *role) {
            std::vector<TensorId> out;
            out.reserve(dense.size());
            for (auto const id : dense) {
                if (id >= loaded.size()) {
                    throw BuildFailure(
                        fmt::format("node '{}' names {} tensor id {}, which the frame does not define", spec.label, role, id));
                }
                out.push_back(loaded[id].id);
            }
            return out;
        };
        node.inputs  = remap(spec.inputs, "input");
        node.outputs = remap(spec.outputs, "output");
        node.op_data = spec.descriptor;

        if (auto *einsum = node.op_data.get_if<EinsumDescriptor>()) {
            // The live blocks a captured node carries, seeded from the loaded snapshot the
            // way every builder seeds them. Restoring them is not cosmetic: a loaded graph is
            // optimized after loading, and a pass that rewrites a prefactor writes through them.
            // The link, target and all-index lists are derived data. A file written before the
            // target list followed C's order holds it sorted, so they are derived again here and
            // a loaded contraction describes itself exactly as the same one captured would.
            detail::derive_index_roles(einsum->spec);
            detail::attach_live_state(*einsum);
            if (einsum->gemm_hint != nullptr) {
                if (spec.hint_ids.size() != 3) {
                    throw BuildFailure(
                        fmt::format("node '{}' carries a GEMM hint with {} operands, not three", spec.label, spec.hint_ids.size()));
                }
                auto const hint_id = [&](std::size_t dense) {
                    if (dense >= loaded.size()) {
                        throw BuildFailure(
                            fmt::format("node '{}' GEMM hint names tensor id {}, which the frame does not define", spec.label, dense));
                    }
                    return loaded[dense].id;
                };
                einsum->gemm_hint->a.id = hint_id(spec.hint_ids[0]);
                einsum->gemm_hint->b.id = hint_id(spec.hint_ids[1]);
                einsum->gemm_hint->c.id = hint_id(spec.hint_ids[2]);
                if (spec.dtype == packed_gemm::ScalarType::Unknown) {
                    throw BuildFailure(fmt::format("node '{}' carries a GEMM hint with dtype 'unknown'", spec.label));
                }
                einsum->gemm_hint->scalar = detail::blas_scalar_from(spec.dtype);
            }
        }
        if (auto *conditional = node.op_data.get_if<ConditionalDescriptor>()) {
            if (spec.then_branch == nullptr) {
                throw BuildFailure(fmt::format("conditional node '{}' has no then-branch", spec.label));
            }
            conditional->then_branch = build_fragment(root, *spec.then_branch, loaded, gates, registry);
            conditional->else_branch = spec.else_branch != nullptr ? build_fragment(root, *spec.else_branch, loaded, gates, registry)
                                                                   : std::make_shared<Graph>("else");
        }
        if (auto *loop = node.op_data.get_if<LoopDescriptor>()) {
            if (spec.body == nullptr) {
                throw BuildFailure(fmt::format("loop node '{}' has no body", spec.label));
            }
            loop->body  = build_fragment(root, *spec.body, loaded, gates, registry);
            loop->state = std::make_shared<LoopState>();
        }
        if (auto *setup = node.op_data.get_if<SetupDescriptor>()) {
            if (spec.body == nullptr) {
                throw BuildFailure(fmt::format("setup node '{}' has no body", spec.label));
            }
            setup->body = build_fragment(root, *spec.body, loaded, gates, registry);
            // Fresh, so a loaded graph knows it has fitted nothing. Anything else would be
            // a file claiming the factors are on hand in a process that has never bound.
            setup->state = std::make_shared<SetupState>();
        }

        try {
            node.execute = build_executor(spec.kind, spec.dtype, spec.rank, node.op_data, graph, node.inputs, node.outputs);
        } catch (std::exception const &error) {
            throw BuildFailure(fmt::format("node '{}' ({}): {}", spec.label, spec.kind, error.what()));
        }
        graph.add_node(std::move(node));
    }
    return loaded;
}

} // namespace

/// Turn a clean document into a graph.
Graph build_graph(IrDocument const &document, SpaceRegistry &registry) {
    Graph graph(document.name);

    // Before anything is annotated. The ids about to be resolved come from THIS registry, and
    // a graph reading its spaces back through a different one would resolve them to whatever
    // happens to sit at those indices there.
    graph.set_space_registry(registry);

    // Exactly the buffers the predicates read out of the document already hold.
    for (auto const &[name, size] : document.gate_flags) {
        auto const found = document.gate_buffers.find(name);
        if (found == document.gate_buffers.end()) {
            throw BuildFailure(fmt::format("gate-flag array '{}' was declared but no buffer was made for it", name));
        }
        graph.name_gate_flags(name, found->second);
    }

    for (auto const &[name, value] : document.params) {
        graph.params_ptr()->set(name, value);
    }

    // The frame's tensor list is the manifest entries and the intermediates,
    // placed at their dense ids.
    std::vector<IrTensor> tensors(document.manifest.size() + document.tensors.size());
    auto const            place = [&tensors](IrTensor const &entry) {
        if (entry.id >= tensors.size()) {
            throw BuildFailure(
                fmt::format("tensor '{}' claims id {}, past the {} tensors this file defines", entry.name, entry.id, tensors.size()));
        }
        tensors[entry.id] = entry;
    };
    for (auto const &entry : document.manifest) {
        place(entry);
    }
    for (auto const &entry : document.tensors) {
        place(entry);
    }

    std::vector<LoadedTensor> const loaded = build_frame(graph, graph, tensors, document.nodes, nullptr, document.gate_buffers, registry);

    // The history the file carries, put back on the graph. Nothing acts on it; it is here so a
    // load, further optimization and a re-save produce a file that still names everything that
    // shaped this graph rather than only what the second pipeline did.
    for (auto const &pass_name : document.structural_passes) {
        graph.note_structural_pass(pass_name);
    }

    // The accuracy history, which unlike the pass list IS acted on: a lossy pass applied to
    // the loaded graph composes against these, and a differential comparison widens by them.
    // Installed directly rather than through note_approximation, which would re-run
    // can_approximate against a budget this process has not set and could refuse a record
    // that was perfectly legal in the process that wrote it. A file is a statement of what
    // WAS applied, not a request to apply it.
    graph.restore_approximations(document.approximations);

    // Slot redirects, once every slot exists. Not a derived cache: a node whose
    // dataflow still names a merged-away duplicate reads the survivor's buffer
    // only because of these.
    for (auto const &[from, to] : document.slot_redirects) {
        if (from >= loaded.size() || to >= loaded.size()) {
            throw BuildFailure(fmt::format("slot redirect {} -> {} names a tensor this file does not define", from, to));
        }
        graph.redirect_slot(loaded[from].id, loaded[to].id);
    }

    // Manifest-declared aliases, by name. This is the only aliasing relation
    // that survives a save at all - a pair that shares storage with no ``View``
    // node recording it exists solely as two addresses that happen to coincide,
    // and a file has neither.
    std::unordered_map<std::string, TensorId> by_name;
    for (auto const &entry : document.manifest) {
        by_name.emplace(entry.name, loaded[entry.id].id);
    }
    for (auto const &entry : document.manifest) {
        if (entry.aliases_input.empty()) {
            continue;
        }
        auto const parent = by_name.find(entry.aliases_input);
        if (parent == by_name.end()) {
            throw BuildFailure(fmt::format("manifest entry '{}' declares itself an alias of '{}', which is not a manifest entry",
                                           entry.name, entry.aliases_input));
        }
        graph.declare_alias(loaded[entry.id].id, parent->second);
    }
    graph.link_alias_storage();

    // The manifest is DERIVED from what the nodes do, so the file's copy is a
    // declaration to check rather than state to install. A disagreement means
    // the node list and the interface describe different programs, which is
    // exactly the corruption a strict reader exists to catch.
    InterfaceManifest const contract = graph.manifest();
    for (auto const &entry : document.manifest) {
        ManifestEntry const *derived = contract.find(entry.name);
        if (derived == nullptr) {
            throw BuildFailure(
                fmt::format("the file declares interface tensor '{}', but no node in the rebuilt graph reads or writes it", entry.name));
        }
        if (derived->direction != entry.direction) {
            throw BuildFailure(fmt::format("interface tensor '{}' is declared '{}' but the rebuilt nodes make it '{}'", entry.name,
                                           entry.direction, derived->direction));
        }
    }
    if (contract.size() != document.manifest.size()) {
        // Naming the difference, not just counting it. A mismatch here means the node list and
        // the interface describe different programs, and the only actionable part of that is
        // WHICH tensor appeared or vanished; a bare count sends the reader back to diff two
        // manifests by hand, which is what this check already did for them.
        std::vector<std::string> declared;
        for (auto const &entry : document.manifest) {
            declared.push_back(entry.name);
        }
        std::vector<std::string> derived = contract.names();
        std::ranges::sort(declared);
        std::ranges::sort(derived);
        std::vector<std::string> only_derived;
        std::ranges::set_difference(derived, declared, std::back_inserter(only_derived));
        std::vector<std::string> only_declared;
        std::ranges::set_difference(declared, derived, std::back_inserter(only_declared));
        throw BuildFailure(fmt::format(
            "the rebuilt graph has {} interface tensors but the file declares {}. Only in the rebuilt graph: [{}]. Only in the file: [{}]",
            contract.size(), document.manifest.size(), fmt::join(only_derived, ", "), fmt::join(only_declared, ", ")));
    }

    return graph;
}

EINSUMS_NAMESPACE_END(compute_graph::graph_ir)
