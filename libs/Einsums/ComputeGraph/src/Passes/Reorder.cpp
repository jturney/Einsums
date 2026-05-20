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

    // Build adjacency and in-degree from data dependencies. We must encode all
    // three hazard classes, not just true dependencies — otherwise the
    // memory-aware reordering below can float a write ahead of a read of the
    // old value (or ahead of an earlier write), corrupting in-place reuse:
    //
    //   RAW (true):  writer → reader   — reader needs the produced value.
    //   WAW (output): writer → writer  — the later write must win.
    //   WAR (anti):  reader → writer   — the overwrite can't happen until
    //                                    every reader of the old value is done.
    //
    // Tracking only ``last_writer`` (the original implementation) covered RAW
    // and WAW but silently dropped WAR, so e.g. a gemm reading t0 followed by
    // an axpy overwriting t0 had no edge and could be swapped. Every edge added
    // here points from a lower to a higher program index, so program order
    // remains a valid topological order and no cycle is ever introduced.
    //
    // Use sets to deduplicate edges — a tensor appearing in both inputs and
    // outputs of the same pair of nodes should only count once.
    std::unordered_map<TensorId, size_t>              last_writer;
    std::unordered_map<TensorId, std::vector<size_t>> readers_since_write;
    std::vector<std::unordered_set<size_t>>           adj_set(n);

    for (size_t i = 0; i < n; i++) {
        // Use effective I/O so Loop/Conditional nodes (whose own input/output
        // lists are empty) carry dependency edges for the tensors their bodies
        // touch — otherwise the memory-aware reorder below can float a loop
        // past a producer or consumer of one of those tensors.
        auto [eff_in, eff_out] = graph.effective_io(nodes[i]);
        for (auto raw_tid : eff_in) {
            // Resolve view aliases to the owning buffer, or a write through a
            // view of T looks unrelated to a read of T and gets reordered past it.
            TensorId const tid = graph.resolve_alias(raw_tid);
            auto           it  = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i); // RAW: writer → reader
            }
            readers_since_write[tid].push_back(i); // remember for a later WAR edge
        }
        for (auto raw_tid : eff_out) {
            TensorId const tid = graph.resolve_alias(raw_tid);
            auto           it  = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i); // WAW: previous writer → this writer
            }
            // WAR: every node that read the old value must precede this write.
            auto rit = readers_since_write.find(tid);
            if (rit != readers_since_write.end()) {
                for (size_t const r : rit->second) {
                    if (r != i) {
                        adj_set[r].insert(i);
                    }
                }
                rit->second.clear(); // reads before this write are now satisfied
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
