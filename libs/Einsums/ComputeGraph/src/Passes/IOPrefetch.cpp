//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

bool is_lifecycle(OpKind kind) {
    return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
}

/// Count real (non-lifecycle) writers of each tensor across @p g and every
/// descendant sub-graph, keyed by the tensor's underlying pointer (stable
/// across graphs, unlike per-graph TensorIds). A DiskRead is hoistable only
/// when its destination has exactly one such writer anywhere in the loop's
/// subtree (the read itself) — that proves nothing, at any nesting depth,
/// overwrites the loaded data. Reads of the destination don't count, so a
/// nested loop merely *reading* the loaded tensor doesn't block hoisting.
void count_subtree_writers_by_ptr(Graph const &g, std::unordered_map<void const *, int> &writers) {
    for (auto const &n : g.nodes()) {
        if (is_lifecycle(n.kind)) {
            continue;
        }
        for (auto tid : n.outputs) {
            auto it = g.tensors_map().find(tid);
            if (it != g.tensors_map().end() && it->second.tensor_ptr != nullptr) {
                writers[it->second.tensor_ptr]++;
            }
        }
    }
    g.for_each_subgraph([&](Graph const &sub) { count_subtree_writers_by_ptr(sub, writers); });
}

/// Move each DiskRead in @p graph as early as legally possible. "Legal"
/// means after (a) every producer of the read's inputs and (b) every prior
/// node that reads or writes the read's *output* tensor — moving a load
/// before a node that touches its destination would change what that node
/// sees (a WAR/WAW violation). The original pass only checked (a); since a
/// DiskRead has no inputs it always slid to position 0, even past a writer
/// of its destination. Returns true if anything moved.
bool prefetch_within(Graph &graph, size_t &num_prefetched) {
    auto &nodes = graph.nodes();
    if (nodes.size() < 2)
        return false;

    std::vector<size_t> read_positions;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (nodes[idx].kind == OpKind::DiskRead) {
            read_positions.push_back(idx);
        }
    }
    if (read_positions.empty())
        return false;

    bool moved_any = false;

    for (unsigned long pos : std::views::reverse(read_positions)) {
        // The read may have shifted from an earlier move; find it.
        while (pos < nodes.size() && nodes[pos].kind != OpKind::DiskRead)
            pos++;
        if (pos >= nodes.size())
            continue;

        // Earliest legal slot.
        size_t earliest = 0;

        // (a) after producers of the read's inputs.
        std::unordered_map<TensorId, size_t> last_writer;
        for (size_t idx = 0; idx < nodes.size(); idx++) {
            for (auto tid : nodes[idx].outputs) {
                last_writer[tid] = idx;
            }
        }
        for (auto tid : nodes[pos].inputs) {
            auto wit = last_writer.find(tid);
            if (wit != last_writer.end() && wit->second < pos) {
                earliest = std::max(earliest, wit->second + 1);
            }
        }

        // (b) after any prior node that reads or writes the read's outputs.
        for (auto out_tid : nodes[pos].outputs) {
            for (size_t idx = 0; idx < pos; idx++) {
                bool touches = false;
                for (auto tid : nodes[idx].inputs) {
                    if (tid == out_tid) {
                        touches = true;
                        break;
                    }
                }
                if (!touches) {
                    for (auto tid : nodes[idx].outputs) {
                        if (tid == out_tid) {
                            touches = true;
                            break;
                        }
                    }
                }
                if (touches) {
                    earliest = std::max(earliest, idx + 1);
                }
            }
        }

        if (earliest < pos) {
            EINSUMS_LOG_INFO("IOPrefetch: moved '{}' from position {} to {} (prefetched by {} nodes)", nodes[pos].label, pos, earliest,
                             pos - earliest);
            std::rotate(nodes.begin() + static_cast<ptrdiff_t>(earliest), nodes.begin() + static_cast<ptrdiff_t>(pos),
                        nodes.begin() + static_cast<ptrdiff_t>(pos) + 1);
            num_prefetched++;
            moved_any = true;
        }
    }

    if (moved_any)
        graph.mark_sorted();
    return moved_any;
}

/// Find a tensor in @p g by pointer, or register a copy of @p src_handle so
/// @p g has a TensorId for it. Used when hoisting a read across a graph
/// boundary: each graph keys tensors by its own ids, but the underlying
/// tensor pointer is stable, so we re-register the destination in the
/// target graph and give the moved read a valid id there. (A fresh id is
/// unused by other target nodes, so no spurious dependency arises; keeping
/// a valid id — rather than clearing it — is what lets a deeply nested read
/// be recognized and re-hoisted level by level.)
TensorId find_or_register_by_ptr(Graph &g, TensorHandle const &src_handle) {
    for (auto const &[id, h] : g.tensors_map()) {
        if (h.tensor_ptr == src_handle.tensor_ptr) {
            return id;
        }
    }
    TensorHandle copy = src_handle;
    return g.register_tensor(std::move(copy));
}

/// Hoist loop-invariant DiskReads out of @p child (a loop body) into @p
/// parent, just before the owning Loop node at @p loop_idx. A read is
/// hoistable when its destination tensor:
///   - is written by exactly one *value* node in the body (the read itself;
///     lifecycle Alloc/Initialize don't count), so nothing in the body
///     overwrites the loaded data;
///   - isn't referenced by any nested sub-graph of the body (a nested loop
///     could write it without the body's node list showing it); and
///   - is already materialized (eager) — IOPrefetch runs before the
///     MaterializationPass, so hoisting a read whose destination is a
///     deferred shell would load into not-yet-allocated storage. Deferred
///     destinations keep their per-iteration read.
/// Such a read produces identical data every iteration, so reading it once
/// before the loop is equivalent and avoids per-iteration disk I/O.
///
/// The moved read keeps a *valid* destination id in the parent (via
/// find_or_register_by_ptr), so a read sitting deep in nested loops gets
/// lifted one level per recursion step until it lands before the outermost
/// loop. The stable topological sort keeps the inserted read before the
/// Loop node; the load writes through the captured tensor pointer, which
/// the body's consumers share.
bool hoist_reads_from_body(Graph &parent, size_t loop_idx, Graph &child, size_t &num_prefetched) {
    // Single-writer test by pointer across the whole loop subtree: a read is
    // hoistable iff nothing, at any depth, overwrites its destination.
    std::unordered_map<void const *, int> writers;
    count_subtree_writers_by_ptr(child, writers);

    auto       &body_nodes = child.nodes();
    auto const &body_map   = child.tensors_map();

    std::vector<Node>   hoisted;
    std::vector<size_t> remove_idx;

    for (size_t idx = 0; idx < body_nodes.size(); idx++) {
        Node const &n = body_nodes[idx];
        if (n.kind != OpKind::DiskRead || n.outputs.size() != 1)
            continue;

        TensorId const out_tid = n.outputs[0];
        auto           dest    = body_map.find(out_tid);
        if (dest == body_map.end())
            continue;

        void const *dest_ptr = dest->second.tensor_ptr;
        if (dest_ptr == nullptr)
            continue;
        if (auto it = writers.find(dest_ptr); it == writers.end() || it->second != 1)
            continue; // overwritten somewhere in the subtree — re-read each iteration matters.

        // Only hoist when the destination is already materialized (eager).
        // IOPrefetch runs before the MaterializationPass, so hoisting a read
        // whose destination is a deferred shell would load into not-yet-
        // allocated storage. Deferred destinations keep their per-iteration read.
        if (dest->second.alloc_state != AllocState::Materialized)
            continue;

        Node copy    = n; // keep the executor (it captures the tensor ptr)
        copy.inputs  = {};
        copy.outputs = {find_or_register_by_ptr(parent, dest->second)};
        copy.label   = fmt::format("prefetch_hoisted({})", n.label);
        hoisted.push_back(std::move(copy));
        remove_idx.push_back(idx);
    }

    if (hoisted.empty())
        return false;

    for (auto it = remove_idx.rbegin(); it != remove_idx.rend(); ++it) {
        body_nodes.erase(body_nodes.begin() + static_cast<ptrdiff_t>(*it));
    }
    child.mark_sorted();

    auto &pnodes = parent.nodes();
    pnodes.insert(pnodes.begin() + static_cast<ptrdiff_t>(loop_idx), std::make_move_iterator(hoisted.begin()),
                  std::make_move_iterator(hoisted.end()));
    parent.mark_sorted();
    num_prefetched += remove_idx.size();

    EINSUMS_LOG_INFO("IOPrefetch: hoisted {} loop-invariant DiskRead(s) out of loop body before position {}", remove_idx.size(), loop_idx);
    return true;
}

/// Recursively prefetch within @p graph and hoist loop-invariant reads out
/// of its loop bodies. Depth-first: a deeply-nested read is lifted one loop
/// level per recursion step until it lands before the outermost loop.
bool process(Graph &graph, size_t &num_prefetched) {
    bool modified = false;

    // Walk parent nodes; for each Loop, process its body first (so nested
    // reads bubble up into the body), then hoist the body's reads out.
    // Indices are re-read each step because hoisting mutates the node list.
    for (size_t idx = 0; idx < graph.nodes().size(); idx++) {
        auto *loop = std::get_if<LoopDescriptor>(&graph.nodes()[idx].op_data);
        if (loop == nullptr || !loop->body) {
            auto *cond = std::get_if<ConditionalDescriptor>(&graph.nodes()[idx].op_data);
            if (cond != nullptr) {
                if (cond->then_branch) {
                    modified |= process(*cond->then_branch, num_prefetched);
                    modified |= hoist_reads_from_body(graph, idx, *cond->then_branch, num_prefetched);
                }
                if (cond->else_branch) {
                    modified |= process(*cond->else_branch, num_prefetched);
                    modified |= hoist_reads_from_body(graph, idx, *cond->else_branch, num_prefetched);
                }
            }
            continue;
        }
        modified |= process(*loop->body, num_prefetched);
        modified |= hoist_reads_from_body(graph, idx, *loop->body, num_prefetched);
    }

    // Within-graph prefetch (sound) after any hoists landed here.
    modified |= prefetch_within(graph, num_prefetched);
    return modified;
}

} // namespace

bool IOPrefetch::run(Graph &graph) {
    _num_prefetched = 0;
    return process(graph, _num_prefetched);
}

} // namespace einsums::compute_graph::passes
