//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

void CommunicationElimination::reset_stats() {
    _num_eliminated = 0;
}

bool CommunicationElimination::run(Graph &graph) {
    auto &nodes = graph.nodes();
    if (nodes.empty())
        return false;

    // Track which tensors have already been allreduced.
    std::unordered_set<TensorId> already_reduced;
    std::vector<bool>            remove(nodes.size(), false);
    // Count THIS graph's removals separately: _num_eliminated is a per-apply
    // total that the recursive driver accumulates across subgraphs, so using it
    // to size the filtered vector below would underflow on the second call.
    size_t eliminated_here = 0;

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        if (node.kind == OpKind::Allreduce) {
            auto const *desc = std::get_if<CommDescriptor>(&node.op_data);
            if (desc && already_reduced.count(desc->tensor_id)) {
                // Redundant: this tensor was already allreduced and hasn't been modified since.
                remove[idx] = true;
                ++eliminated_here;
                ++_num_eliminated;
                EINSUMS_LOG_INFO("CommunicationElimination: removed redundant Allreduce for tensor id={}", desc->tensor_id);
                report(2, fmt::format("remove redundant Allreduce for tensor id={} (already reduced, unmodified since)", desc->tensor_id));
                continue;
            }
            if (desc) {
                already_reduced.insert(desc->tensor_id);
            }
        }

        // If a compute node writes to a tensor, invalidate its "already reduced" status.
        for (auto tid : node.outputs) {
            if (node.kind != OpKind::Allreduce && node.kind != OpKind::Broadcast && node.kind != OpKind::Allgather) {
                already_reduced.erase(tid);
            }
        }
    }

    if (eliminated_here == 0)
        return false;

    // Remove marked nodes.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size() - eliminated_here);
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (!remove[idx])
            filtered.push_back(std::move(nodes[idx]));
    }
    nodes = std::move(filtered);
    graph.note_structural_change();
    graph.mark_sorted();

    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
