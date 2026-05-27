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

    // Build the set of tensors that are READ by at least one node. Resolve
    // through view aliases to the owning buffer: reading a view of T (or T
    // itself) keeps T live, and conversely a *write* through a view of T must
    // be judged against whether T is read — see the output check below.
    std::unordered_set<TensorId> consumed_tensors;
    for (auto const &node : nodes) {
        for (auto tid : node.inputs) {
            consumed_tensors.insert(graph.resolve_alias(tid));
        }
    }

    // Tensors referenced anywhere in a child sub-graph (a nested loop body
    // or conditional branch) are live, even if no node in THIS graph reads
    // them — a Loop node doesn't list its body's tensor reads as inputs, so
    // without this a producer feeding only a nested loop would look dead.
    // Compared by underlying pointer because the nested body uses different
    // TensorIds for the same tensor (see Graph::collect_subtree_referenced_ptrs).
    std::unordered_set<void const *> subtree_referenced;
    graph.collect_subtree_referenced_ptrs(subtree_referenced);

    // Identify which tensors are graph-owned intermediates
    std::unordered_set<TensorId> intermediate_tensors;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.is_intermediate) {
            intermediate_tensors.insert(tid);
        }
    }

    // A node is dead if ALL its outputs are:
    // 1. Graph-owned intermediates (is_intermediate=true)
    // 2. Not consumed by any other node in this graph
    // 3. Not referenced by any descendant sub-graph
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

        // Check if all outputs are dead (intermediate + not consumed here + not
        // used by a child body). Resolve each output through view aliases to its
        // owning buffer: a write through a view of T is dead only if T itself is
        // — otherwise removing it would drop a partial update to a live tensor
        // (the view tid is an unread intermediate, but the owner is what counts).
        bool all_outputs_dead = true;
        for (auto raw_tid : node.outputs) {
            TensorId const tid             = graph.resolve_alias(raw_tid);
            bool const     is_intermediate = intermediate_tensors.count(tid) > 0;
            bool const     is_consumed     = consumed_tensors.count(tid) > 0;

            bool used_by_subgraph = false;
            if (auto it = graph.tensors_map().find(tid); it != graph.tensors_map().end() && it->second.tensor_ptr != nullptr) {
                used_by_subgraph = subtree_referenced.count(it->second.tensor_ptr) > 0;
            }

            if (!is_intermediate || is_consumed || used_by_subgraph) {
                all_outputs_dead = false;
                break;
            }
        }

        if (all_outputs_dead) {
            dead[idx] = true;
            _num_eliminated++;
            EINSUMS_LOG_INFO("DeadNodeElimination: removing dead node {} ({})", node.id, node.label);
            report(2, fmt::format("remove dead node {} ({}) — outputs never consumed", node.id, node.label));
        }
    }

    if (_num_eliminated == 0) {
        return false;
    }
    report(1, fmt::format("eliminated {} dead node(s)", _num_eliminated));

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
