//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

bool DeadNodeElimination::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        _num_eliminated = 0;
        return false;
    }

    size_t const n = nodes.size();

    // Build the set of tensors that are READ by at least one node
    std::unordered_set<TensorId> consumed_tensors;
    for (auto const &node : nodes) {
        for (auto tid : node.inputs) {
            consumed_tensors.insert(tid);
        }
    }

    // Identify which tensors are graph-owned intermediates
    std::unordered_set<TensorId> intermediate_tensors;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.is_intermediate) {
            intermediate_tensors.insert(tid);
        }
    }

    // A node is dead if ALL its outputs are:
    // 1. Graph-owned intermediates (is_intermediate=true)
    // 2. Not consumed by any other node
    // Control flow and memory nodes are never eliminated.

    std::vector<bool> dead(n, false);
    _num_eliminated = 0;

    for (size_t idx = 0; idx < n; idx++) {
        auto const &node = nodes[idx];

        // Never eliminate control flow or memory nodes
        if (node.kind == OpKind::Conditional || node.kind == OpKind::Loop || node.kind == OpKind::Alloc || node.kind == OpKind::Free) {
            continue;
        }

        // Node with no outputs is side-effect only (e.g., scale) — keep it
        if (node.outputs.empty()) {
            continue;
        }

        // Check if all outputs are dead (intermediate + not consumed)
        bool all_outputs_dead = true;
        for (auto tid : node.outputs) {
            bool const is_intermediate = intermediate_tensors.count(tid) > 0;
            bool const is_consumed     = consumed_tensors.count(tid) > 0;
            if (!is_intermediate || is_consumed) {
                all_outputs_dead = false;
                break;
            }
        }

        if (all_outputs_dead) {
            dead[idx] = true;
            _num_eliminated++;
            EINSUMS_LOG_INFO("DeadNodeElimination: removing dead node {} ({})", node.id, node.label);
        }
    }

    if (_num_eliminated == 0) {
        return false;
    }

    // Remove dead nodes
    std::vector<Node> filtered;
    filtered.reserve(n - _num_eliminated);
    for (size_t idx = 0; idx < n; idx++) {
        if (!dead[idx]) {
            filtered.push_back(std::move(nodes[idx]));
        }
    }
    nodes = std::move(filtered);
    graph.mark_sorted();

    return true;
}

} // namespace einsums::compute_graph::passes
