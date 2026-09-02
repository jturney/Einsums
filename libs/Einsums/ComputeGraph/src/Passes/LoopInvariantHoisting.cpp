//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

void LoopInvariantHoisting::reset_stats() {
    _num_hoisted = 0;
}

bool LoopInvariantHoisting::run(Graph &graph) {
    PassCounter const hoisted_count{_num_hoisted};
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
        auto *loop_desc = std::get_if<LoopDescriptor>(&node.op_data);
        if (loop_desc != nullptr && loop_desc->body) {
            run_recursive(*loop_desc->body);
        }
    }

    hoist_one_level(graph);
}

void LoopInvariantHoisting::hoist_one_level(Graph &graph) {
    auto        &nodes          = graph.nodes();
    size_t const hoisted_before = _num_hoisted;

    // Cost caveat (known, out of scope here): this pass has no cost/benefit
    // test. Hoisting a cheap producer of a huge tensor out of a loop extends
    // that tensor's live range across the whole loop, which can cost more
    // memory than it saves in recomputation. A future cost model belongs here.

    // Find all Loop nodes
    for (size_t loop_idx = 0; loop_idx < nodes.size(); loop_idx++) {
        auto *loop_desc = std::get_if<LoopDescriptor>(&nodes[loop_idx].op_data);
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
            // outright rather than proven safe. (The View case is also caught
            // today by the single-writer guard below, but incidentally -- that
            // guard resolves a view's output through its alias to the parent
            // buffer, which usually has no writer in the body at all -- so it
            // stops holding the moment the parent is written once per
            // iteration, which is exactly what a blocked residual does.)
            if (bnode.kind == OpKind::WriteParam || has_runtime_view_bounds(bnode)) {
                note_skip("node's per-iteration effect is a parameter write or a parameter-bound slice, not visible as dataflow",
                          fmt::format("body node '{}'", bnode.label));
                continue;
            }

            // A node that reads the tensor it writes is self-modifying
            // (scale(C), or an accumulating gemm C = C + A·B). Never hoist
            // these: the per-iteration update would be lost. ``reads_destination``
            // covers the always-accumulating ops and nonzero-prefactor einsum/
            // permute/gemm; the explicit input==output scan catches any other op
            // that lists the same tensor as both an input and an output.
            bool self_modifying = reads_destination(bnode);
            for (auto out_tid : bnode.outputs) {
                if (self_modifying)
                    break;
                for (auto in_tid : bnode.inputs) {
                    if (out_tid == in_tid) {
                        self_modifying = true;
                        break;
                    }
                }
            }
            if (self_modifying)
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
                bool const written_in_body = body_writes.count(tid) > 0 || escapes.subtree_writer_count(tid) > 0;
                bool const from_hoisted    = hoisted_outputs.count(tid) > 0;
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
            bool output_read_earlier = false;
            for (auto out_tid : bnode.outputs) {
                for (size_t bj = 0; bj < bi && !output_read_earlier; bj++) {
                    // Use *effective* reads so an earlier control-flow node (a
                    // nested loop / conditional) that reads the output inside its
                    // subtree counts, its own raw input list is empty.
                    auto [ein, eout] = loop_desc->body->effective_io(body_nodes[bj]);
                    if (std::ranges::find(ein, out_tid) != ein.end()) {
                        output_read_earlier = true;
                    }
                }
                if (output_read_earlier) {
                    break;
                }
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

        // Move invariant nodes from body to parent graph. Hoisted nodes
        // reference TensorIds from the body graph's tensor table, but those
        // IDs are not registered in the parent graph, naively appending the
        // node to the parent leaves later passes (and the executor) unable
        // to resolve its tensors. So for each TensorId the hoisted node
        // touches, we register the corresponding TensorHandle in the parent
        // and rewrite the node's input/output IDs to use the parent's ID.
        // The body's tensor table is left untouched so non-hoisted body
        // nodes still see their original IDs.
        std::unordered_map<TensorId, TensorId> id_remap;
        auto                                   remap_or_register = [&](TensorId body_tid) -> TensorId {
            auto it = id_remap.find(body_tid);
            if (it != id_remap.end())
                return it->second;
            TensorHandle const handle = loop_desc->body->tensor(body_tid);
            // If the parent already has a TensorId for this underlying buffer,
            // reuse it rather than minting a fresh one. The buffer's identity is
            // its pointer; registering a *new* id for an already-known buffer
            // would hide the write-after-write / read-after-write relationship
            // between the hoisted node and the parent nodes that touch the same
            // tensor (the scheduler keys on TensorId), so a later pass like
            // Reorder could swap them and the wrong write would win.
            //
            // That reuse-or-mint rule is exactly Graph::find_or_register_tensor_ptr,
            // which states the same orphan-parent-handle convention effective_io uses.
            TensorId const parent_tid = graph.find_or_register_tensor_ptr(handle);
            id_remap[body_tid]        = parent_tid;
            return parent_tid;
        };

        std::vector<Node> hoisted;
        std::vector<Node> remaining;
        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            if (invariant[bi]) {
                Node h = std::move(body_nodes[bi]);
                for (auto &tid : h.inputs)
                    tid = remap_or_register(tid);
                for (auto &tid : h.outputs)
                    tid = remap_or_register(tid);
                EINSUMS_LOG_INFO("LoopInvariantHoisting: hoisting '{}' out of loop '{}'", h.label, nodes[loop_idx].label);
                report(2, fmt::format("hoist '{}' out of loop '{}' — inputs invariant across iterations", h.label, nodes[loop_idx].label));
                hoisted.push_back(std::move(h));
                _num_hoisted++;
            } else {
                remaining.push_back(std::move(body_nodes[bi]));
            }
        }

        // Collect each hoisted node's output IDs so we can wire them as
        // inputs of the Loop node below, without this data-flow edge,
        // ``topological_sort`` has nothing tying the hoisted producers
        // to the loop's body.
        std::vector<TensorId> hoisted_output_ids;
        for (auto const &h : hoisted) {
            for (auto tid : h.outputs)
                hoisted_output_ids.push_back(tid);
        }

        // Update body to only contain remaining (non-hoisted) nodes.
        // Removing nodes keeps the body's relative order valid, but its
        // position-keyed dependency lists are stale; declare that.
        body_nodes = std::move(remaining);
        // Both node vectors were rebuilt in place, so each graph's node-set
        // counter has to be moved by hand: the body lost the hoisted nodes and
        // the parent gains them a few lines below.
        loop_desc->body->note_structural_change();
        loop_desc->body->mark_sorted();

        // Insert hoisted nodes directly BEFORE the loop in the parent's
        // ``nodes`` vector. ``topological_sort`` (Kahn's algorithm)
        // processes nodes in their current order when building dataflow
        // edges and again when ties exist in the ready queue. Appending
        // at the end leaves the loop ahead of its newly-introduced
        // producers, which both prevents the edge from being recorded
        // (the writer is seen after the reader) and biases the queue
        // toward the wrong order.
        size_t const n_hoisted = hoisted.size();
        graph.note_structural_change();
        nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(loop_idx), std::make_move_iterator(hoisted.begin()),
                     std::make_move_iterator(hoisted.end()));
        loop_idx += n_hoisted; // The loop has shifted; keep the outer index in sync.

        // Wire the loop's data-flow dependency on the hoisted nodes' outputs.
        // With the hoisted nodes placed before the loop, the topo sort will
        // now build the writer→reader edge correctly.
        for (auto tid : hoisted_output_ids) {
            nodes[loop_idx].inputs.push_back(tid);
        }
    }

    if (_num_hoisted > hoisted_before) {
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
