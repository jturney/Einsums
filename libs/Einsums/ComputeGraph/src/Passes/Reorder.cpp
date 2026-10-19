//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/BackedWrites.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../NamedHazards.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

bool Reorder::run(Graph &graph) {
    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    size_t const n = nodes.size();

    // Build adjacency and in-degree from data dependencies. We must encode all
    // three hazard classes, not just true dependencies, otherwise the
    // memory-aware reordering below can float a write ahead of a read of the
    // old value (or ahead of an earlier write), corrupting in-place reuse:
    //
    //   RAW (true):  writer → reader, reader needs the produced value.
    //   WAW (output): writer → writer, the later write must win.
    //   WAR (anti):  reader → writer, the overwrite can't happen until
    //                                    every reader of the old value is done.
    //
    // Without WAR, a gemm reading t0 followed by an axpy overwriting t0 would
    // have no edge and could be swapped. Every edge added
    // here points from a lower to a higher program index, so program order
    // remains a valid topological order and no cycle is ever introduced.
    //
    // Use sets to deduplicate edges, a tensor appearing in both inputs and
    // outputs of the same pair of nodes should only count once.
    std::unordered_map<TensorId, size_t>              last_writer;
    std::unordered_map<TensorId, std::vector<size_t>> readers_since_write;
    std::vector<std::unordered_set<size_t>>           adj_set(n);

    // The same hazards over the ParamTable. A WriteParam and the
    // parameter-bound Views that resolve against it share no tensor, so
    // without these edges the reschedule below is free to float a slice ahead
    // of the write that positions it -- which reads the previous iteration's
    // value, or throws outright on the first iteration when the parameter has
    // never been set. The same keys order a disk read after the graph's own
    // write of that dataset. See named_writes / named_reads in Node.hpp.
    detail::NamedHazards named;

    // A write-backed tensor that no operation writes in memory is produced into its storage by
    // SlabStreaming's block loops. Until Materialization puts a fill from that storage before its
    // first reader, nothing links a node that reads it in memory to those loops, so the reader
    // reads the storage's key as well and follows every write of it.
    std::unordered_map<TensorId, std::string> written_to_storage_only;
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.backing && handle.backing->write && handle.tensor_ptr != nullptr &&
            !compute_graph::detail::written_in_memory(graph, handle.tensor_ptr)) {
            written_to_storage_only.emplace(tid, compute_graph::detail::storage_key(*handle.backing));
        }
    }

    for (size_t i = 0; i < n; i++) {
        std::vector<std::string> reads = named_reads(nodes[i]);
        // Use effective I/O so Loop/Conditional nodes (whose own input/output
        // lists are empty) carry dependency edges for the tensors their bodies
        // touch: otherwise the memory-aware reorder below can float a loop
        // past a producer or consumer of one of those tensors.
        auto [eff_in, eff_out] = graph.effective_io(nodes[i]);
        for (auto const raw_tid : eff_in) {
            // Resolve view aliases to the owning buffer, or a write through a
            // view of T looks unrelated to a read of T and gets reordered past it.
            TensorId const tid = graph.buffer_of(raw_tid);
            auto const     it  = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i); // RAW: writer → reader
            }
            readers_since_write[tid].push_back(i); // remember for a later WAR edge
            if (auto const stored = written_to_storage_only.find(tid); stored != written_to_storage_only.end()) {
                reads.push_back(stored->second);
            }
        }
        for (auto const raw_tid : eff_out) {
            TensorId const tid = graph.buffer_of(raw_tid);
            auto const     it  = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj_set[it->second].insert(i); // WAW: previous writer → this writer
            }
            // WAR: every node that read the old value must precede this write.
            auto const rit = readers_since_write.find(tid);
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
        named.visit(i, reads, named_writes(nodes[i]), [&adj_set](size_t producer, size_t consumer) { adj_set[producer].insert(consumer); });
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
    // Use the same effective-I/O + alias resolution as the hazard scan above:
    // a raw inputs/outputs scan double-counts a view against its parent buffer
    // and misses tensors a Loop/Conditional body touches, so the freed-bytes
    // heuristic would attribute the wrong bytes to the wrong node.
    std::unordered_map<TensorId, size_t> last_consumer;
    for (size_t i = 0; i < n; i++) {
        auto [eff_in, eff_out] = graph.effective_io(nodes[i]);
        for (auto const raw_tid : eff_in) {
            last_consumer[graph.buffer_of(raw_tid)] = i;
        }
        for (auto const raw_tid : eff_out) {
            last_consumer[graph.buffer_of(raw_tid)] = i; // Also counts as "using" it
        }
    }

    // Each node's score is the bytes the graph frees after it, less the bytes it first touches.
    // Only tensors the graph frees count: a caller's tensor or a write-backed one stays allocated
    // whatever the order, and counting it pulled the last reader of a large caller input ahead of
    // a consumer that frees an intermediate. Charging a producer for what it allocates keeps a
    // ready producer behind a ready consumer.
    std::unordered_map<TensorId, size_t>              first_use;
    std::unordered_map<TensorId, std::vector<size_t>> uses;
    for (size_t i = 0; i < n; i++) {
        auto [eff_in, eff_out] = graph.effective_io(nodes[i]);
        for (auto const raw_tid : eff_in) {
            first_use.try_emplace(graph.buffer_of(raw_tid), i);
            uses[graph.buffer_of(raw_tid)].push_back(i);
        }
        for (auto const raw_tid : eff_out) {
            first_use.try_emplace(graph.buffer_of(raw_tid), i);
            uses[graph.buffer_of(raw_tid)].push_back(i);
        }
    }
    auto const freed_by_graph = [](TensorHandle const &h) {
        return h.is_intermediate || (h.backing != nullptr && !h.backing->write && h.ownership == TensorOwnership::Graph);
    };
    std::vector<std::int64_t> bytes_freed(n, 0);
    for (auto const &[tid, last_idx] : last_consumer) {
        auto const &tensors = graph.tensors_map();
        auto const  it      = tensors.find(tid);
        if (it != tensors.end() && freed_by_graph(it->second)) {
            bytes_freed[last_idx] += static_cast<std::int64_t>(it->second.total_bytes());
            bytes_freed[first_use.at(tid)] -= static_cast<std::int64_t>(it->second.total_bytes());
        }
    }

    // Memory-aware Kahn's algorithm: among ready nodes, prefer the highest score, and among equal
    // scores the earliest in program order. Most nodes score the same, and without the program
    // order the heap returned them in whatever order its layout gave, which reshuffled
    // producers ahead of the consumers that free memory.
    auto const cmp = [&bytes_freed](size_t a, size_t b) {
        if (bytes_freed[a] != bytes_freed[b]) {
            return bytes_freed[a] < bytes_freed[b]; // max-heap: largest freed first
        }
        return a > b; // then program order
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
        // Cycle or stuck nodes, don't modify (nodes vector is untouched)
        return false;
    }

    // Never worse than the order given. The greedy choice is local and can raise the peak, so the
    // reschedule is kept only when its peak is lower. An order's peak is the most bytes live at
    // once, with each tensor the graph frees live from its first use to its last in that order.
    auto const peak_of = [&](std::vector<size_t> const &seq) {
        std::vector<size_t> position(n);
        for (size_t p = 0; p < n; ++p) {
            position[seq[p]] = p;
        }
        std::vector<std::int64_t> delta(n + 1, 0);
        for (auto const &[tid, node_uses] : uses) {
            auto const &tensors = graph.tensors_map();
            auto const  it      = tensors.find(tid);
            if (it == tensors.end() || !freed_by_graph(it->second)) {
                continue;
            }
            size_t lo = n, hi = 0;
            for (size_t const node : node_uses) {
                lo = std::min(lo, position[node]);
                hi = std::max(hi, position[node]);
            }
            delta[lo] += static_cast<std::int64_t>(it->second.total_bytes());
            delta[hi + 1] -= static_cast<std::int64_t>(it->second.total_bytes());
        }
        std::int64_t live = 0, peak = 0;
        for (size_t p = 0; p < n; ++p) {
            live += delta[p];
            peak = std::max(peak, live);
        }
        return peak;
    };
    std::vector<size_t> given(n);
    for (size_t i = 0; i < n; ++i) {
        given[i] = i;
    }
    if (std::int64_t const given_peak = peak_of(given), new_peak = peak_of(order); new_peak >= given_peak) {
        report(2, fmt::format("kept the given order: its peak of {} bytes is no higher than the reschedule's {}", given_peak, new_peak));
        order = std::move(given);
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
                // Node already has an empty executor, don't corrupt the graph.
                // This can happen for structural nodes (Loop, Conditional) that
                // don't have an execute lambda.
            }
            sorted.push_back(std::move(nodes[idx]));
        }
        nodes = std::move(sorted);
        graph.mark_sorted();

        if (_verbosity >= 1) {
            size_t moved = 0;
            for (size_t i = 0; i < n; i++) {
                if (order[i] != i) {
                    ++moved;
                }
            }
            report(1, fmt::format("memory-aware reschedule moved {} of {} node(s)", moved, n));
        }
    }

    return changed;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
