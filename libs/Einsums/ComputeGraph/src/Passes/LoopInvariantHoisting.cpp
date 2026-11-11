//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

bool LoopInvariantHoisting::transform(Graph &graph) {
    PassCounter const hoisted_count{_stats.num_hoisted};
    run_recursive(graph);
    return hoisted_count.moved();
}

void LoopInvariantHoisting::run_recursive(Graph &graph) {
    // Innermost-first: descend into each Loop body BEFORE hoisting at this
    // level. An invariant produced inside an inner loop is first lifted into
    // its enclosing body here, then (if still invariant there) lifted again by
    // this level's sweep. Multi-level hoisting is thus the composition of
    // single-level hoists within one run() call.
    //
    // We descend into Loop bodies ONLY, never Conditional branches. A node
    // inside a branch runs only when the predicate selects that branch, so
    // lifting it into the enclosing graph would execute it unconditionally and
    // change semantics when the predicate is false. (Even the "safe" case, a
    // graph-owned intermediate consumed solely inside the branch, only wastes
    // work.) The single-level driver likewise never treats a Conditional as a
    // hoist candidate and refuses any candidate whose input is written inside a
    // conditional subtree, so nothing crosses a branch boundary in either
    // direction.
    for (auto &node : graph.nodes()) {
        auto *loop_desc = node.op_data.get_if<LoopDescriptor>();
        if (loop_desc != nullptr && loop_desc->body) {
            run_recursive(*loop_desc->body);
        }
    }

    hoist_one_level(graph);
}

void LoopInvariantHoisting::hoist_one_level(Graph &graph) {
    auto        &nodes          = graph.nodes();
    size_t const hoisted_before = _stats.num_hoisted;

    // Cost caveat (known, out of scope here): this pass has no cost/benefit
    // test. Hoisting a cheap producer of a huge tensor out of a loop extends
    // that tensor's live range across the whole loop, which can cost more
    // memory than it saves in recomputation. A future cost model belongs here.

    // Find all Loop nodes
    for (size_t loop_idx = 0; loop_idx < nodes.size(); loop_idx++) {
        auto *loop_desc = nodes[loop_idx].op_data.get_if<LoopDescriptor>();
        if (!loop_desc || !loop_desc->body)
            continue;

        auto &body_nodes = loop_desc->body->nodes();
        if (body_nodes.empty())
            continue;

        // Build set of tensors WRITTEN by any body node
        std::unordered_set<TensorId> body_writes;
        for (auto const &bnode : body_nodes) {
            for (auto tid : bnode.outputs) {
                body_writes.insert(tid);
            }
        }

        // Real value-writer count per underlying buffer across the loop subtree.
        // Used to refuse hoisting a producer whose output is also written by
        // another node in the loop (which would change which write wins).
        // Shared with SymmetryPropagation and the region framework: three
        // derivations of one relation disagreeing in the corner nobody tested is
        // this module's signature bug, so there is only ever one.
        auto const escapes = EscapeAnalysis::over(*loop_desc->body);

        // Body nodes by the named resources (parameters, disk datasets) they read and write. A
        // control-flow node reports its child graphs' keys too, so a nested access counts.
        std::unordered_map<std::string, std::vector<size_t>> named_readers;
        std::unordered_map<std::string, std::vector<size_t>> named_writers;
        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            for (auto &key : named_reads(body_nodes[bi])) {
                named_readers[std::move(key)].push_back(bi);
            }
            for (auto &key : named_writes(body_nodes[bi])) {
                named_writers[std::move(key)].push_back(bi);
            }
        }
        auto const touched_by_other = [](std::unordered_map<std::string, std::vector<size_t>> const &by_key, std::string const &key,
                                         size_t self) {
            return std::ranges::any_of(by_key, [&key, self](auto const &entry) {
                return named_keys_overlap(entry.first, key) &&
                       std::ranges::any_of(entry.second, [self](size_t other) { return other != self; });
            });
        };

        // Identify invariant nodes: all inputs are NOT written by any body node
        // Iterate in order and propagate (hoisted outputs become invariant)
        std::unordered_set<TensorId> hoisted_outputs;
        std::vector<bool>            invariant(body_nodes.size(), false);

        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            auto const &bnode = body_nodes[bi];

            // Skip control flow and memory nodes
            if (is_control_flow(bnode.kind) || bnode.kind == OpKind::Alloc || bnode.kind == OpKind::Free) {
                continue;
            }

            // Nodes whose per-iteration effect does not travel through their
            // tensor inputs. A WriteParam writes the ParamTable, and its
            // callback form has no tensor inputs at all, so the invariance test
            // below waves it through; hoisting it pins every downstream
            // parametric View at whatever the first iteration resolved to. A
            // parameter-bound View is the mirror image: invariant parent, moving
            // slice. Neither is inspectable as dataflow, so both are refused
            // outright. (The single-writer guard below catches the View case
            // only incidentally, and not once the parent is written per
            // iteration, as a blocked residual does.)
            if (bnode.kind == OpKind::WriteParam || has_runtime_view_bounds(bnode)) {
                note_skip("node's per-iteration effect is a parameter write or a parameter-bound slice, not visible as dataflow",
                          fmt::format("body node '{}'", bnode.label));
                continue;
            }

            // A node whose effects its tensor lists do not show: a caller's closure may read
            // state the loop changes (a captured counter, a parameter through the ParamTable) or
            // act once per iteration by design (a log line). Run once, either is wrong.
            if (NodeEffects const effects = effects_of(bnode); !effects.deterministic || effects.external_effects) {
                note_skip("node reads state or has effects its tensor lists do not show (a custom closure not declared pure)",
                          fmt::format("body node '{}'", bnode.label));
                continue;
            }

            // Ordering through a named resource rather than a tensor. A DiskRead of a dataset the
            // body also stores has no tensor inputs, so the test below calls it invariant, and
            // hoisted it loads the file once, before the first store. Symmetrically, a write the
            // rest of the body reads or rewrites must keep its place among them.
            bool const named_conflict =
                std::ranges::any_of(named_reads(bnode), [&](std::string const &key) { return touched_by_other(named_writers, key, bi); }) ||
                std::ranges::any_of(named_writes(bnode), [&](std::string const &key) {
                    return touched_by_other(named_writers, key, bi) || touched_by_other(named_readers, key, bi);
                });
            if (named_conflict) {
                note_skip("node reads or writes a parameter or disk dataset another body node also touches",
                          fmt::format("body node '{}'", bnode.label));
                continue;
            }

            // A node that reads the tensor it writes is self-modifying
            // (scale(C), or an accumulating gemm C = C + A·B). Never hoist
            // these: the per-iteration update would be lost. ``reads_destination``
            // covers the always-accumulating ops and nonzero-prefactor einsum/
            // permute/gemm; the explicit input==output scan catches any other op
            // that lists the same tensor as both an input and an output.
            if (reads_destination(bnode) || std::ranges::find_first_of(bnode.outputs, bnode.inputs) != bnode.outputs.end())
                continue;

            bool all_inputs_invariant = true;
            for (auto tid : bnode.inputs) {
                // A tensor counts as written-in-body if a *direct* body node
                // writes it, OR if any node anywhere in the loop body subtree
                // writes the same underlying buffer. The latter catches writes
                // performed inside nested subgraphs, a conditional branch or
                // an inner loop, which never appear in this body's own output
                // lists. Without it, an input mutated only inside a conditional
                // would look invariant and the consumer would be wrongly
                // hoisted out of the loop.
                bool const written_in_body = body_writes.contains(tid) || escapes.subtree_writer_count(tid) > 0;
                bool const from_hoisted    = hoisted_outputs.contains(tid);
                if (written_in_body && !from_hoisted) {
                    all_inputs_invariant = false;
                    break;
                }
            }

            // Refuse to hoist a producer whose output is written by more than
            // one value-node in the loop subtree. Removing its per-iteration
            // write would change which write wins each iteration, e.g. an
            // einsum that resets C, followed by an in-place op that would then
            // accumulate across iterations instead of starting fresh. (A
            // DiskRead with no inputs is "invariant" by the input check above;
            // this guard stops it being hoisted when something else in the
            // loop overwrites its destination.) Reads of the output are fine.
            // A count of exactly one is the proof; anything else, including a
            // tensor with no buffer to count against, leaves it unproven and so
            // refused.
            bool const single_writer_outputs =
                std::ranges::all_of(bnode.outputs, [&escapes](TensorId out_tid) { return escapes.subtree_writer_count(out_tid) == 1; });
            if (!single_writer_outputs) {
                note_skip("node's output is written more than once in the loop subtree", fmt::format("body node '{}'", bnode.label));
                continue;
            }

            // Refuse to hoist a producer whose output is read by an *earlier*
            // body node. That earlier read observes the value from the previous
            // iteration (the output is loop-carried *through* this producer), so
            // computing it once before the loop would change what the earlier
            // reader sees. Reads by *later* body nodes are fine, they consume
            // this iteration's value, which is loop-invariant once hoisted.
            //
            // Asked of BUFFERS and of EFFECTIVE reads (buffer_io). A view made outside the capture
            // reads the output under its own id, with no View node in the body naming the parent,
            // so an id comparison saw no earlier reader and the hoist made every iteration read
            // this iteration's value. And an earlier control-flow node (a nested loop or
            // conditional) that reads the output inside its subtree lists nothing of it itself.
            bool output_read_earlier = false;
            for (size_t bj = 0; bj < bi && !output_read_earlier; bj++) {
                BufferIo const earlier = buffer_io(*loop_desc->body, body_nodes[bj]);
                output_read_earlier    = std::ranges::any_of(
                    bnode.outputs, [&](TensorId out_tid) { return earlier.reads.contains(loop_desc->body->buffer_of(out_tid)); });
            }
            if (output_read_earlier) {
                // Loop-carried through this producer: an earlier reader sees the
                // previous iteration's value, which computing it once would change.
                note_skip("node's output is read by an earlier body node, so the value is loop-carried",
                          fmt::format("body node '{}'", bnode.label));
                continue;
            }

            if (!all_inputs_invariant) {
                note_skip("node reads a tensor the loop body rewrites each iteration", fmt::format("body node '{}'", bnode.label));
            }
            if (all_inputs_invariant) {
                invariant[bi] = true;
                for (auto tid : bnode.outputs) {
                    hoisted_outputs.insert(tid);
                }
            }
        }

        // Bail out cheaply if nothing's invariant, otherwise we'd move-from
        // every body_nodes entry just to put them back, and a ``continue``
        // path that left ``body_nodes`` with moved-from std::function
        // executors would silently turn the loop body into a no-op.
        bool any_invariant = false;
        for (bool const v : invariant) {
            if (v) {
                any_invariant = true;
                break;
            }
        }
        if (!any_invariant)
            continue;

        // Move the invariant nodes out, directly BEFORE the loop. ``topological_sort`` (Kahn's
        // algorithm) processes nodes in their current order when building dataflow edges and again
        // when ties exist in the ready queue, so producers placed after the loop would neither get
        // their edge recorded nor sort ahead of it. ``Graph::hoist`` gives each moved node the
        // parent's id for every tensor it names, reusing an id the parent already has for a buffer
        // so the scheduler sees each hazard between a hoisted node and the parent's own (minting a
        // fresh one would hide them, and a later pass like Reorder could swap the two writes), and
        // hands a body-declared deferred intermediate's declaration out with its writer.
        //
        // Held as a reference to the body itself: loop_desc points into this graph's node list,
        // which the hoist may reallocate.
        Graph              &body = *loop_desc->body;
        std::vector<size_t> picks;
        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            if (!invariant[bi]) {
                continue;
            }
            picks.push_back(bi);
            EINSUMS_LOG_INFO("LoopInvariantHoisting: hoisting '{}' out of loop '{}'", body_nodes[bi].label, nodes[loop_idx].label);
            report(2, fmt::format("hoist '{}' out of loop '{}': its inputs are invariant across iterations", body_nodes[bi].label,
                                  nodes[loop_idx].label));
        }

        // The outputs the body still reads become inputs of the Loop node below: without that
        // data-flow edge, ``topological_sort`` has nothing tying the hoisted producers to the
        // body that consumes them. Only those. An output nothing left in the body reads, such as
        // an intermediate whose reader was hoisted with its writer, would make the Loop node a
        // reader of a tensor it never touches, and a pass that slices that tensor declines a
        // statement it cannot slice: AxisTiling left a hoisted 79 MB ladder intermediate whole.
        // Asked of the nodes that stay, before they are the only ones left.
        std::unordered_set<void const *> body_reads;
        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            if (invariant[bi]) {
                continue;
            }
            auto const [reads, writes] = body.effective_io(body_nodes[bi]);
            for (TensorId const tid : reads) {
                if (TensorHandle const *read = body.find_tensor(body.buffer_of(tid)); read != nullptr) {
                    body_reads.insert(read->tensor_ptr);
                }
            }
        }

        auto const moved = graph.hoist(body, picks, loop_idx);
        _stats.num_hoisted += moved.size();
        loop_idx += moved.size(); // The loop has shifted; keep the outer index in sync.

        std::vector<TensorId> hoisted_output_ids;
        for (size_t const at : moved) {
            for (auto tid : nodes[at].outputs) {
                TensorHandle const *written = graph.find_tensor(graph.buffer_of(tid));
                if (written == nullptr || written->tensor_ptr == nullptr || body_reads.contains(written->tensor_ptr)) {
                    hoisted_output_ids.push_back(tid);
                }
            }
        }
        if (!hoisted_output_ids.empty()) {
            graph.edit(loop_idx, [&](NodeEdit &e) {
                for (auto tid : hoisted_output_ids) {
                    e->inputs.push_back(tid);
                }
            });
        }
    }

    if (_stats.num_hoisted > hoisted_before) {
        // The hoisted producers were inserted directly before their loop, so
        // the order is already valid; declare the mutation so the sort below
        // rebuilds the stale dependency lists instead of early-returning on
        // the pre-hoist state. Sort per level: each recursion level owns its
        // own graph and re-sorts only when it actually hoisted something here.
        graph.mark_sorted();
        graph.topological_sort();
    }
}

std::vector<std::string> LoopInvariantHoisting::explain() const {
    if (num_hoisted() == 0) {
        return {};
    }
    return {fmt::format("LoopInvariantHoisting: hoisted {} node(s) out of loops", num_hoisted())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
