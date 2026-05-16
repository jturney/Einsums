//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

bool CommunicationElimination::run(Graph &graph) {
    _num_eliminated = 0;

    auto &nodes = graph.nodes();
    if (nodes.empty())
        return false;

    // Track which tensors have already been allreduced.
    std::unordered_set<TensorId> already_reduced;
    std::vector<bool>            remove(nodes.size(), false);

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        if (node.kind == OpKind::Allreduce) {
            auto const *desc = std::get_if<CommDescriptor>(&node.op_data);
            if (desc && already_reduced.count(desc->tensor_id)) {
                // Redundant: this tensor was already allreduced and hasn't been modified since.
                remove[idx] = true;
                _num_eliminated++;
                EINSUMS_LOG_INFO("CommunicationElimination: removed redundant Allreduce for tensor id={}", desc->tensor_id);
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

    if (_num_eliminated == 0)
        return false;

    // Remove marked nodes.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size() - _num_eliminated);
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (!remove[idx])
            filtered.push_back(std::move(nodes[idx]));
    }
    nodes = std::move(filtered);
    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
