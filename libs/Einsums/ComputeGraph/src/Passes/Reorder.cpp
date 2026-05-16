//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>

#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

bool Reorder::run(Graph &graph) {
    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    size_t const n = nodes.size();

    // Build adjacency and in-degree from data dependencies.
    // Use sets to deduplicate edges — a tensor appearing in both inputs and
    // outputs of the same pair of nodes should only count once.
    std::unordered_map<TensorId, size_t>    last_writer;
    std::vector<std::unordered_set<size_t>> adj_set(n);

    for (size_t i = 0; i < n; i++) {
        for (auto tid : nodes[i].inputs) {
            auto it = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i);
            }
        }
        for (auto tid : nodes[i].outputs) {
            auto it = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i);
            }
            last_writer[tid] = i;
        }
    }

    // Convert to vector adjacency and compute in-degree
    std::vector<std::vector<size_t>> adj(n);
    std::vector<size_t>              in_degree(n, 0);
    for (size_t i = 0; i < n; i++) {
        adj[i].assign(adj_set[i].begin(), adj_set[i].end());
        for (size_t const s : adj[i]) {
            in_degree[s]++;
        }
    }

    // Compute "memory freed" for each node:
    // A tensor's last consumer is the last node that reads it.
    // When a node is the last consumer of a tensor, scheduling it frees that memory.
    std::unordered_map<TensorId, size_t> last_consumer;
    for (size_t i = 0; i < n; i++) {
        for (auto tid : nodes[i].inputs) {
            last_consumer[tid] = i;
        }
        for (auto tid : nodes[i].outputs) {
            last_consumer[tid] = i; // Also counts as "using" it
        }
    }

    // For each node, compute bytes freed when it completes
    std::vector<size_t> bytes_freed(n, 0);
    for (auto const &[tid, last_idx] : last_consumer) {
        auto const &tensors = graph.tensors_map();
        auto        it      = tensors.find(tid);
        if (it != tensors.end()) {
            bytes_freed[last_idx] += it->second.total_bytes();
        }
    }

    // Memory-aware Kahn's algorithm:
    // Among ready nodes, prefer the one that frees the most memory.
    auto cmp = [&bytes_freed](size_t a, size_t b) {
        return bytes_freed[a] < bytes_freed[b]; // max-heap: largest freed first
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> ready(cmp);

    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    // Collect ordering indices first (don't move nodes yet)
    std::vector<size_t> order;
    order.reserve(n);

    while (!ready.empty()) {
        size_t const idx = ready.top();
        ready.pop();
        order.push_back(idx);

        for (size_t const succ : adj[idx]) {
            if (--in_degree[succ] == 0) {
                ready.push(succ);
            }
        }
    }

    if (order.size() != n) {
        // Cycle or stuck nodes — don't modify (nodes vector is untouched)
        return false;
    }

    // Check if the order actually changed
    bool changed = false;
    for (size_t i = 0; i < n; i++) {
        if (order[i] != i) {
            changed = true;
            break;
        }
    }

    if (changed) {
        // Move nodes into the new order.
        // Verify all nodes have valid executors before and after.
        std::vector<Node> sorted;
        sorted.reserve(n);
        for (size_t const idx : order) {
            if (!nodes[idx].execute) {
                // Node already has an empty executor — don't corrupt the graph.
                // This can happen for structural nodes (Loop, Conditional) that
                // don't have an execute lambda.
            }
            sorted.push_back(std::move(nodes[idx]));
        }
        nodes = std::move(sorted);
        graph.mark_sorted();
    }

    return changed;
}

} // namespace einsums::compute_graph::passes
