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

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
void verify_into(Graph const &graph, std::string const &where, std::vector<std::string> &problems) {
    auto const note = [&](Node const &node, std::string const &message) {
        problems.push_back(fmt::format("{}node #{} '{}' ({}): {}", where, node.id, node.label, node.kind, message));
    };
    auto const rank_of = [&graph](TensorId id) -> std::optional<std::size_t> {
        auto const *handle = graph.find_tensor(id);
        return handle != nullptr ? std::optional<std::size_t>{handle->rank} : std::nullopt;
    };

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
            }
        }

        for_each_child_graph(
            node, [&](Graph const &child) { verify_into(child, fmt::format("{}{} '{}' / ", where, node.kind, node.label), problems); });
    }
}

} // namespace

std::vector<std::string> Graph::verify() const {
    std::vector<std::string> problems;
    verify_into(*this, fmt::format("graph '{}': ", _name), problems);
    return problems;
}

EINSUMS_NAMESPACE_END(compute_graph)
