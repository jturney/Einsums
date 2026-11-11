//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <ranges>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Move each DiskRead in @p graph as early as legally possible. "Legal"
/// means after (a) every producer of the read's inputs and (b) every prior
/// node that reads or writes the read's *output* tensor, moving a load
/// before a node that touches its destination would change what that node
/// sees (a WAR/WAW violation). The original pass only checked (a); since a
/// DiskRead has no inputs it always slid to position 0, even past a writer
/// of its destination. Returns true if anything moved.
bool prefetch_within(Graph &graph, size_t &num_prefetched) {
    auto const &nodes = graph.nodes();
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

    // Each node's buffer reads and writes, kept in step with the node list as reads move.
    std::vector<BufferIo> io = buffer_io(graph);

    bool moved_any = false;

    for (unsigned long pos : std::views::reverse(read_positions)) {
        // The read may have shifted from an earlier move; find it.
        while (pos < nodes.size() && nodes[pos].kind != OpKind::DiskRead)
            pos++;
        if (pos >= nodes.size())
            continue;

        // Earliest legal slot.
        size_t earliest = 0;

        // (a) and (b): after every prior node that writes what the read reads, or reads or writes
        // what it writes. Asked of buffers, so a view or a redirected slot of the destination is
        // the destination, and of effective I/O, so a Loop or Conditional counts for what its
        // body touches: its own lists are empty, and the read floated ahead of a loop that still
        // read the destination's old value.
        for (size_t idx = 0; idx < pos; idx++) {
            if (conflicts(io[idx], io[pos])) {
                earliest = std::max(earliest, idx + 1);
            }
        }

        // (c) after any prior write of the dataset it reads. A write names only its source tensor
        // and the read only its destination, so (a) and (b) cannot see the file they share, and a
        // read moved ahead of the graph's own write returns the file's previous contents.
        for (auto const &key : named_reads(nodes[pos])) {
            for (size_t idx = 0; idx < pos; idx++) {
                if (named_keys_contain(named_writes(nodes[idx]), key)) {
                    earliest = std::max(earliest, idx + 1);
                }
            }
        }

        if (earliest < pos) {
            EINSUMS_LOG_INFO("IOPrefetch: moved '{}' from position {} to {} (prefetched by {} nodes)", nodes[pos].label, pos, earliest,
                             pos - earliest);
            graph.move_node(pos, earliest);
            std::rotate(io.begin() + static_cast<ptrdiff_t>(earliest), io.begin() + static_cast<ptrdiff_t>(pos),
                        io.begin() + static_cast<ptrdiff_t>(pos) + 1);
            num_prefetched++;
            moved_any = true;
        }
    }

    return moved_any;
}

/// Hoist loop-invariant DiskReads out of @p child (a loop body) into @p
/// parent, just before the owning Loop node at @p loop_idx. A read is
/// hoistable when its destination tensor:
///   - is written by exactly one *value* node in the body (the read itself;
///     lifecycle Alloc/Initialize don't count), so nothing in the body
///     overwrites the loaded data;
///   - isn't referenced by any nested sub-graph of the body (a nested loop
///     could write it without the body's node list showing it); and
///   - is already materialized (eager), IOPrefetch runs before the
///     MaterializationPass, so hoisting a read whose destination is a
///     deferred shell would load into not-yet-allocated storage. Deferred
///     destinations keep their per-iteration read.
/// Such a read produces identical data every iteration, so reading it once
/// before the loop is equivalent and avoids per-iteration disk I/O.
///
/// The moved read is registered against the destination BUFFER in the parent
/// (via Graph::find_or_register_tensor_ptr), so it owns the same parent
/// TensorId that the Loop's body reads map back to through effective_io. That
/// shared id is a real RAW edge DiskRead -> Loop: any executor (Dataflow
/// included) orders the read before the loop by dependency, not by node
/// position. A read sitting deep in nested loops is lifted one level per
/// recursion step until it lands before the outermost loop, gaining the edge
/// at each level. The load writes through the captured tensor pointer, which
/// the body's consumers share.
/// Whether any node of @p graph, at any depth, writes the named resource @p key. A control-flow
/// node's named writes include its child graphs', so one level covers the subtree.
bool writes_anywhere(Graph const &graph, std::string const &key) {
    return std::ranges::any_of(graph.nodes(), [&key](Node const &node) { return named_keys_contain(named_writes(node), key); });
}

bool hoist_reads_from_body(Graph &parent, size_t loop_idx, Graph &child, size_t &num_prefetched) {
    // Single-writer test by pointer across the whole loop subtree: a read is
    // hoistable iff nothing, at any depth, overwrites its destination. Reads do
    // not count, so a nested loop merely READING the loaded tensor does not
    // block the hoist; lifecycle nodes do not count either.
    EscapeAnalysis const writers = EscapeAnalysis::over(child);

    auto const &body_nodes = child.nodes();
    auto const &body_map   = child.tensors_map();

    std::vector<size_t> picks;

    for (size_t idx = 0; idx < body_nodes.size(); idx++) {
        Node const &n = body_nodes[idx];
        if (n.kind != OpKind::DiskRead || n.outputs.size() != 1)
            continue;

        TensorId const out_tid = n.outputs[0];
        auto           dest    = body_map.find(out_tid);
        if (dest == body_map.end())
            continue;

        // A destination with no pointer yet is counted by nobody, so it answers
        // zero here and declines, which is the conservative direction.
        if (writers.subtree_writer_count(out_tid) != 1)
            continue; // overwritten somewhere in the subtree, re-read each iteration matters.

        // Nor may the loop write the dataset: each iteration then reads what the last one stored.
        if (std::ranges::any_of(named_reads(n), [&](std::string const &key) { return writes_anywhere(child, key); }))
            continue;

        // Only hoist when the destination is already materialized (eager).
        // IOPrefetch runs before the MaterializationPass, so hoisting a read
        // whose destination is a deferred shell would load into not-yet-
        // allocated storage. Deferred destinations keep their per-iteration read.
        if (dest->second.alloc_state != AllocState::Materialized)
            continue;

        picks.push_back(idx);
    }

    if (picks.empty())
        return false;

    // The parent names each destination BUFFER by the id the body's reads resolve to through
    // effective_io, which hoist gives it. That id, not the read's position, is what gives the
    // Loop a RAW dependency on the read under a concurrent executor.
    for (std::size_t const at : parent.hoist(child, picks, loop_idx)) {
        parent.edit(at, [](NodeEdit &e) { e->label = fmt::format("prefetch_hoisted({})", e->label); });
    }
    num_prefetched += picks.size();

    EINSUMS_LOG_INFO("IOPrefetch: hoisted {} loop-invariant DiskRead(s) out of loop body before position {}", picks.size(), loop_idx);
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
        // hoist_reads_from_body inserts hoisted reads into THIS graph's node vector, which can
        // reallocate it; for_each_child finds every body before the first visit, so the second
        // hoist does not read a moved descriptor.
        graph.for_each_child(
            idx,
            [&](Graph &child) {
                modified |= process(child, num_prefetched);
                modified |= hoist_reads_from_body(graph, idx, child, num_prefetched);
            },
            /*include_setup=*/false);
    }

    // Within-graph prefetch (sound) after any hoists landed here.
    modified |= prefetch_within(graph, num_prefetched);
    return modified;
}

} // namespace

bool IOPrefetch::transform(Graph &graph) {
    PassCounter const prefetched{_stats.num_prefetched};
    bool const        result = process(graph, _stats.num_prefetched);
    if (prefetched.moved()) {
        report(1, fmt::format("prefetched {} DiskRead(s) earlier to overlap I/O with compute", _stats.num_prefetched));
    }
    return result;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
