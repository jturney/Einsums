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

namespace {

// Buffer pointers referenced (read or written) by @p g's OWN direct nodes,
// each TensorId resolved through g's own map. Lifecycle nodes
// (Alloc/Free/Materialize/Initialize) are skipped: the hoisted-lifecycle fix
// gives a PARENT graph Materialize/Free nodes that carry the parent tids of a
// BODY's buffers, so counting them here would flag a genuinely dead body
// producer's output as externally referenced and keep it alive. effective_io
// skips the same kinds for the same reason (see Graph.cpp is_lifecycle).
void collect_own_node_ptrs(Graph const &g, std::unordered_set<void const *> &out) {
    auto is_lifecycle = [](OpKind kind) {
        return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
    };
    auto add = [&](TensorId tid) {
        if (auto const *h = g.find_tensor(g.resolve_alias(tid)); h != nullptr && h->tensor_ptr != nullptr) {
            out.insert(h->tensor_ptr);
        }
    };
    for (auto const &node : g.nodes()) {
        if (is_lifecycle(node.kind)) {
            continue;
        }
        for (auto tid : node.inputs) {
            add(tid);
        }
        for (auto tid : node.outputs) {
            add(tid);
        }
    }
}

} // namespace

bool DeadNodeElimination::run(Graph &graph) {
    _num_eliminated = 0;
    std::unordered_set<void const *> const no_external;
    return run_one(graph, no_external);
}

bool DeadNodeElimination::run_one(Graph &graph, std::unordered_set<void const *> const &external_refs) {
    graph.topological_sort();

    bool modified = false;

    auto &nodes = graph.nodes();
    if (!nodes.empty()) {
        size_t const n = nodes.size();

        // Build the set of tensors that are READ by at least one node. Resolve
        // through view aliases to the owning buffer: reading a view of T (or T
        // itself) keeps T live, and conversely a *write* through a view of T must
        // be judged against whether T is read, see the output check below.
        std::unordered_set<TensorId> consumed_tensors;
        for (auto const &node : nodes) {
            for (auto tid : node.inputs) {
                consumed_tensors.insert(graph.resolve_alias(tid));
            }
        }

        // Tensors referenced anywhere in a child sub-graph (a nested loop body
        // or conditional branch) are live, even if no node in THIS graph reads
        // them: a Loop node doesn't list its body's tensor reads as inputs, so
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
        // 4. Not referenced by an enclosing or sibling graph (external_refs)
        // Control flow and memory nodes are never eliminated.

        std::vector<bool> dead(n, false);
        size_t            eliminated_here = 0;

        for (size_t idx = 0; idx < n; idx++) {
            auto const &node = nodes[idx];

            // Never eliminate control flow or memory nodes
            if (node.kind == OpKind::Conditional || node.kind == OpKind::Loop || node.kind == OpKind::Alloc || node.kind == OpKind::Free) {
                continue;
            }

            // Node with no outputs is side-effect only (e.g., scale), keep it
            if (node.outputs.empty()) {
                continue;
            }

            // Check if all outputs are dead (intermediate + not consumed here + not
            // used by a child body + not referenced from outside this graph).
            // Resolve each output through view aliases to its owning buffer: a
            // write through a view of T is dead only if T itself is, otherwise
            // removing it would drop a partial update to a live tensor (the view
            // tid is an unread intermediate, but the owner is what counts).
            bool all_outputs_dead = true;
            for (auto raw_tid : node.outputs) {
                TensorId const tid             = graph.resolve_alias(raw_tid);
                bool const     is_intermediate = intermediate_tensors.count(tid) > 0;
                bool const     is_consumed     = consumed_tensors.count(tid) > 0;

                bool used_by_subgraph = false;
                bool used_externally  = false;
                if (auto it = graph.tensors_map().find(tid); it != graph.tensors_map().end() && it->second.tensor_ptr != nullptr) {
                    used_by_subgraph = subtree_referenced.count(it->second.tensor_ptr) > 0;
                    // Any reference from an enclosing graph (a parent node after
                    // this control-flow child, or a sibling loop body) keeps this
                    // output live. Conservative on purpose: a body tensor read
                    // only by an outside consumer must not have its producer
                    // eliminated, that would leave the reader observing unwritten
                    // storage.
                    used_externally = external_refs.count(it->second.tensor_ptr) > 0;
                }

                if (!is_intermediate || is_consumed || used_by_subgraph || used_externally) {
                    all_outputs_dead = false;
                    break;
                }
            }

            if (all_outputs_dead) {
                dead[idx] = true;
                eliminated_here++;
                EINSUMS_LOG_INFO("DeadNodeElimination: removing dead node {} ({})", node.id, node.label);
                report(2, fmt::format("remove dead node {} ({}) — outputs never consumed", node.id, node.label));
            }
        }

        if (eliminated_here > 0) {
            _num_eliminated += eliminated_here;
            report(1, fmt::format("eliminated {} dead node(s)", eliminated_here));

            std::vector<Node> filtered;
            filtered.reserve(n - eliminated_here);
            for (size_t idx = 0; idx < n; idx++) {
                if (!dead[idx]) {
                    filtered.push_back(std::move(nodes[idx]));
                }
            }
            nodes = std::move(filtered);
            graph.mark_sorted();
            modified = true;
        }
    }

    // Descend into each direct sub-graph. A child sees, as external references:
    // this graph's own-node references (the parent may read a body tensor after
    // the loop) plus every OTHER sibling sub-tree's references (one loop body
    // may feed another). Collect each child's full sub-tree once so all-but-self
    // is a simple union.
    std::vector<Graph *> children;
    graph.for_each_subgraph([&](Graph &sub) { children.push_back(&sub); });

    if (!children.empty()) {
        std::unordered_set<void const *> own_ptrs;
        collect_own_node_ptrs(graph, own_ptrs);

        std::vector<std::unordered_set<void const *>> child_subtrees(children.size());
        for (size_t i = 0; i < children.size(); i++) {
            collect_own_node_ptrs(*children[i], child_subtrees[i]);          // the child's own references
            children[i]->collect_subtree_referenced_ptrs(child_subtrees[i]); // and its descendants'
        }

        for (size_t i = 0; i < children.size(); i++) {
            std::unordered_set<void const *> child_external = external_refs;
            child_external.insert(own_ptrs.begin(), own_ptrs.end());
            for (size_t j = 0; j < children.size(); j++) {
                if (j == i) {
                    continue;
                }
                child_external.insert(child_subtrees[j].begin(), child_subtrees[j].end());
            }
            if (run_one(*children[i], child_external)) {
                modified = true;
            }
        }
    }

    return modified;
}

} // namespace einsums::compute_graph::passes
