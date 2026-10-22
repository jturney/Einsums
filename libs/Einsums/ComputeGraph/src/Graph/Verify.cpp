//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Verify.cpp
/// @brief The structural invariants a graph must keep through every pass.
///
/// Each check here is one a pass once broke without the graph failing to run: the damage showed
/// up later, far from the pass, as a lookup that found the wrong node or a contraction that read
/// the wrong axes. Keeping them in one walk is what lets the test suite run it after every pass.

#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/Detail/BackedWrites.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Backed storage visible from one graph, by tensor object, with the backing that names it.
///
/// By object and not by id, because a loop body numbers its tensors on its own and its handle for
/// a parent's backed tensor carries no backing (TensorHandle::backing): the object is the identity
/// the two levels share.
using BackedStorage = std::unordered_map<void const *, TensorBacking const *>;

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
void verify_into(Graph const &graph, std::string const &where, std::vector<std::string> &problems, BackedStorage backed) {
    auto const note = [&](Node const &node, std::string const &message) {
        problems.push_back(fmt::format("{}node #{} '{}' ({}): {}", where, node.id, node.label, node.kind, message));
    };
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.backing && handle.tensor_ptr != nullptr) {
            backed.emplace(handle.tensor_ptr, handle.backing.get());
        }
    }
    auto const rank_of = [&graph](TensorId id) -> std::optional<std::size_t> {
        auto const *handle = graph.find_tensor(id);
        return handle != nullptr ? std::optional<std::size_t>{handle->rank} : std::nullopt;
    };

    auto const dims_of = [&graph](TensorId id) -> std::vector<std::size_t> const * {
        auto const *handle = graph.find_tensor(id);
        return handle != nullptr ? &handle->dims : nullptr;
    };
    // Every letter of a contraction or a permute names one extent, whichever operand carries it.
    // An intermediate declared with a different extent than the nodes using it runs, and writes or
    // reads a different number of elements than its neighbours expect.
    auto const check_extents = [&](Node const                                                                  &node,
                                   std::initializer_list<std::pair<std::vector<std::string> const *, TensorId>> operands) {
        std::unordered_map<std::string, std::size_t> extent;
        for (auto const &[letters, id] : operands) {
            auto const *dims = dims_of(id);
            if (dims == nullptr || dims->size() != letters->size()) {
                continue; // the rank check reports it
            }
            for (std::size_t axis = 0; axis < dims->size(); axis++) {
                auto const [it, fresh] = extent.try_emplace((*letters)[axis], (*dims)[axis]);
                if (!fresh && it->second != (*dims)[axis]) {
                    note(node, fmt::format("index '{}' has extent {} on one operand and {} on another", (*letters)[axis], it->second,
                                           (*dims)[axis]));
                }
            }
        }
    };

    // A slot redirect says a merged-away tensor now reads another's storage. The pass that made it
    // removed the source's writer; one that still writes it writes into the target, and a
    // redirect pointed the wrong way sends every access at storage nothing allocates.
    for (auto const &[from, to] : graph.slot_redirects()) {
        for (auto const &node : graph.nodes()) {
            if (!is_lifecycle(node.kind) && std::ranges::find(node.outputs, from) != node.outputs.end()) {
                note(node, fmt::format("writes tensor #{}, whose slot is redirected to tensor #{}", from, to));
            }
        }
    }

    std::unordered_set<NodeId> ids;
    for (auto const &node : graph.nodes()) {
        if (node.id == unassigned_node_id) {
            note(node, "carries no id");
        } else if (!ids.insert(node.id).second) {
            note(node, "shares its id with an earlier node");
        }

        for (auto const &[list, role] : {std::pair{&node.inputs, "input"}, std::pair{&node.outputs, "output"}}) {
            for (TensorId const id : *list) {
                if (graph.find_tensor(id) == nullptr) {
                    note(node, fmt::format("{} tensor #{} is not registered in this graph", role, id));
                }
            }
        }

        if (!is_control_flow(node.kind) && !node.execute) {
            note(node, "has no executor");
        }

        if (node.kind == OpKind::Alloc || node.kind == OpKind::Free || node.kind == OpKind::Materialize) {
            auto const *desc = node.op_data.get_if<AllocDescriptor>();
            if (desc == nullptr) {
                note(node, "is a lifecycle node without an AllocDescriptor");
            } else if (graph.find_tensor(desc->tensor_id) == nullptr) {
                note(node, fmt::format("manages tensor #{}, which is not registered in this graph", desc->tensor_id));
            }
        } else if (node.kind == OpKind::Initialize) {
            auto const *desc = node.op_data.get_if<InitializeDescriptor>();
            if (desc == nullptr) {
                note(node, "is an Initialize node without an InitializeDescriptor");
            } else if (graph.find_tensor(desc->tensor_id) == nullptr) {
                note(node, fmt::format("initializes tensor #{}, which is not registered in this graph", desc->tensor_id));
            }
        }

        if (node.kind == OpKind::Einsum) {
            if (auto const *desc = node.op_data.get_if<EinsumDescriptor>()) {
                if (desc->params == nullptr || desc->indices == nullptr || desc->site == nullptr) {
                    note(node, "is a contraction without its live params, indices and site");
                } else if (node.inputs.size() >= 2 && !node.outputs.empty()) {
                    auto const lists = live_index_lists(*desc);
                    auto const check = [&](std::vector<std::string> const &letters, TensorId id, char const *operand) {
                        if (auto const rank = rank_of(id); rank.has_value() && *rank != letters.size()) {
                            note(node, fmt::format("operand {} has rank {} but the spec names {} indices for it", operand, *rank,
                                                   letters.size()));
                        }
                    };
                    check(lists.a, node.inputs[0], "A");
                    check(lists.b, node.inputs[1], "B");
                    if (!lists.c.empty()) {
                        check(lists.c, node.outputs[0], "C");
                    }
                    check_extents(node, {{&lists.a, node.inputs[0]}, {&lists.b, node.inputs[1]}, {&lists.c, node.outputs[0]}});
                }
            }
        } else if (node.kind == OpKind::Permute) {
            if (auto const *desc = node.op_data.get_if<PermuteDescriptor>();
                desc != nullptr && !node.inputs.empty() && !node.outputs.empty()) {
                for (auto const &[letters, id, operand] :
                     {std::tuple{&desc->a_indices, node.inputs[0], "A"}, std::tuple{&desc->c_indices, node.outputs[0], "C"}}) {
                    if (auto const rank = rank_of(id); rank.has_value() && *rank != letters->size()) {
                        note(node,
                             fmt::format("operand {} has rank {} but the spec names {} indices for it", operand, *rank, letters->size()));
                    }
                }
                check_extents(node, {{&desc->a_indices, node.inputs[0]}, {&desc->c_indices, node.outputs[0]}});
            }
        } else if (node.kind == OpKind::Scale || node.kind == OpKind::ElementTransform) {
            // In place: the executor rewrites one tensor, so a listed input can only be that output.
            if (!node.outputs.empty() && std::ranges::any_of(node.inputs, [&](TensorId id) { return id != node.outputs[0]; })) {
                note(node, node.kind == OpKind::Scale ? "reads a tensor other than the one it scales in place"
                                                      : "reads a tensor other than the one it transforms in place");
            }
        } else if (node.kind == OpKind::Axpby) {
            // y is read in place: the executor takes x from the first input and y from the output,
            // and a second input, when listed, is that same y.
            if (!node.outputs.empty() && node.inputs.size() > 1 && node.inputs[1] != node.outputs[0]) {
                note(node, "lists a y it does not write");
            }
            if (!node.inputs.empty() && !node.outputs.empty()) {
                auto const *x = dims_of(node.inputs[0]);
                auto const *y = dims_of(node.outputs[0]);
                if (x != nullptr && y != nullptr && *x != *y) {
                    note(node, fmt::format("adds x of shape [{}] into y of shape [{}]", fmt::join(*x, ","), fmt::join(*y, ",")));
                }
            }
        }

        // A tensor with a READ backing is read-only to the graph. A write would land in the copy in
        // memory and never in the storage, so the next replay's load would silently undo it; for an
        // input file that is the protection against overwriting it by mistake. Its own lifecycle
        // (the Materialize and the Initialize that fills it) is the one writer allowed. A View
        // writes nothing: its output is an alias, and a node that writes THROUGH the alias is
        // caught here, since the alias resolves to the backed buffer underneath it. A HostToDevice
        // writes nothing either: it fills the tensor's device copy and leaves the host copy the
        // backing loaded as it was. A WRITE backing names a tensor the graph produces, so its
        // writers are expected.
        if (!backed.empty() && !is_lifecycle(node.kind) && node.kind != OpKind::View && node.kind != OpKind::HostToDevice) {
            for (TensorId const id : node.outputs) {
                TensorHandle const *handle = graph.find_tensor(graph.buffer_of(id));
                if (handle == nullptr || handle->tensor_ptr == nullptr) {
                    continue;
                }
                if (auto const hit = backed.find(handle->tensor_ptr); hit != backed.end() && !hit->second->write) {
                    note(node, fmt::format("writes '{}', which is backed by {}; a backed tensor is read-only to the graph", handle->name,
                                           hit->second->describe()));
                }
            }
        }

        // The destination rule (DestinationRead.hpp): operands first, then the destination once,
        // exactly when the node reads it.
        if (auto const violation = destination_rule_violation(graph, node)) {
            note(node, *violation);
        }

        // A registered descriptor's own account of its output extents (DescriptorHooks.hpp).
        if (auto const mismatch = hooked_extent_mismatch(graph, node)) {
            note(node, *mismatch);
        }

        for_each_child_graph(node, [&](Graph const &child) {
            verify_into(child, fmt::format("{}{} '{}' / ", where, node.kind, node.label), problems, backed);
        });
    }
}

} // namespace

std::vector<std::string> Graph::verify() const {
    std::vector<std::string> problems;
    verify_into(*this, fmt::format("graph '{}': ", _name), problems, {});

    // The other direction: a write backing declares the tensor an output the graph produces, and
    // one that nothing produces, in memory or straight into its storage, is a declaration the
    // program does not keep. Asked of the whole tree, since the producer may be in a loop body.
    auto const check = [&](Graph const &graph) {
        for (auto const &[tid, handle] : graph.tensors_map()) {
            if (handle.backing && handle.backing->write && handle.tensor_ptr != nullptr &&
                !detail::written_in_memory(*this, handle.tensor_ptr) && !detail::written_to_storage(*this, *handle.backing)) {
                problems.push_back(fmt::format("graph '{}': tensor '{}' has a write backing, {}, but no node writes it", _name, handle.name,
                                               handle.backing->describe()));
            }
        }
    };
    check(*this);
    for_each_descendant([&](Graph const &graph) { check(graph); });
    return problems;
}

EINSUMS_NAMESPACE_END(compute_graph)
