//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

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

    // Track which buffers have already been allreduced, keyed by the storage an id reads and
    // writes so a write through another name over the same buffer still invalidates it.
    std::unordered_set<TensorId> already_reduced;
    std::vector<bool>            remove(nodes.size(), false);
    PassCounter const            eliminated{_num_eliminated};

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        // A node this pass does not understand may write anything, through a view, a redirected
        // slot or a sub-graph, and none of that shows in its output ids. Forget every reduction
        // rather than trust one across it; a control-flow node is the same case, since its body
        // runs any number of times and writes what it likes.
        if (!understands(graph, node) || is_control_flow(node.kind)) {
            if (node.kind == OpKind::Allreduce) {
                note_skip("the node carries a feature this pass does not understand",
                          fmt::format("node '{}': {}", node.label, describe_features(features_of(graph, node))));
            }
            already_reduced.clear();
            continue;
        }

        if (node.kind == OpKind::Allreduce) {
            auto const *desc = node.op_data.get_if<CommDescriptor>();
            if (desc && already_reduced.count(graph.buffer_of(desc->tensor_id))) {
                // Redundant: this tensor was already allreduced and hasn't been modified since.
                remove[idx] = true;
                ++_num_eliminated;
                EINSUMS_LOG_INFO("CommunicationElimination: removed redundant Allreduce for tensor id={}", desc->tensor_id);
                report(2, fmt::format("remove redundant Allreduce for tensor id={} (already reduced, unmodified since)", desc->tensor_id));
                continue;
            }
            if (desc) {
                already_reduced.insert(graph.buffer_of(desc->tensor_id));
            }
        }

        // If a compute node writes to a tensor, invalidate its "already reduced" status.
        for (auto tid : node.outputs) {
            if (node.kind != OpKind::Allreduce && node.kind != OpKind::Broadcast && node.kind != OpKind::Allgather) {
                already_reduced.erase(graph.buffer_of(tid));
            }
        }
    }

    if (!eliminated.moved())
        return false;

    graph.erase_nodes(remove);
    graph.mark_sorted();

    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
