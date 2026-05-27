//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

bool is_lifecycle(OpKind kind) {
    return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
}

bool prefactor_is_zero(PrefactorScalar const &pf) {
    return std::visit([](auto v) { return v == decltype(v){}; }, pf);
}

/// A node that reads its own destination is self-modifying across iterations
/// and must never be hoisted out of a loop — hoisting drops the per-iteration
/// update. This covers the always-accumulating ops (scale/axpy/axpby/element-
/// transform) and any einsum/permute/batched-gemm with a *nonzero* destination
/// prefactor (``C = c_pf*C + …`` reads the old C). A pure overwrite (prefactor
/// zero) does not read its output and may still be hoisted.
bool reads_its_output(Node const &nd) {
    switch (nd.kind) {
    case OpKind::Scale:
    case OpKind::Axpy:
    case OpKind::Axpby:
    case OpKind::ElementTransform:
        return true;
    default:
        break;
    }
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return !prefactor_is_zero(e->c_prefactor);
    }
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->beta != 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta != 0.0;
    }
    return false;
}

/// Count real (non-lifecycle) writers of each tensor across @p g and every
/// descendant sub-graph, keyed by the tensor's underlying pointer (stable
/// across graphs). A producer can only be hoisted out of a loop when each
/// of its outputs has exactly one such writer in the loop subtree — itself.
/// Otherwise another node in the loop also writes that tensor, and removing
/// the producer's per-iteration write changes which write wins (e.g. an
/// einsum that resets C followed by an in-place scale that would then
/// accumulate). Reads don't count, so a consumer of the produced value
/// doesn't block the hoist.
void count_subtree_writers_by_ptr(Graph const &g, std::unordered_map<void const *, int> &writers) {
    for (auto const &n : g.nodes()) {
        if (is_lifecycle(n.kind)) {
            continue;
        }
        for (auto tid : n.outputs) {
            // Resolve view aliases to the owning buffer: a write through a view
            // of T is a write to T, so it must count against T's pointer (not
            // the view object's). Otherwise an invariant-looking consumer of T
            // would be hoisted past a view-write that mutates T each iteration.
            auto it = g.tensors_map().find(g.resolve_alias(tid));
            if (it != g.tensors_map().end() && it->second.tensor_ptr != nullptr) {
                writers[it->second.tensor_ptr]++;
            }
        }
    }
    g.for_each_subgraph([&](Graph const &sub) { count_subtree_writers_by_ptr(sub, writers); });
}

} // namespace

bool LoopInvariantHoisting::run(Graph &graph) {
    auto &nodes  = graph.nodes();
    _num_hoisted = 0;

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

        // Real value-writer count per tensor pointer across the loop subtree.
        // Used to refuse hoisting a producer whose output is also written by
        // another node in the loop (which would change which write wins).
        std::unordered_map<void const *, int> subtree_writers;
        count_subtree_writers_by_ptr(*loop_desc->body, subtree_writers);

        // Identify invariant nodes: all inputs are NOT written by any body node
        // Iterate in order and propagate (hoisted outputs become invariant)
        std::unordered_set<TensorId> hoisted_outputs;
        std::vector<bool>            invariant(body_nodes.size(), false);

        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            auto const &bnode = body_nodes[bi];

            // Skip control flow and memory nodes
            if (bnode.kind == OpKind::Conditional || bnode.kind == OpKind::Loop || bnode.kind == OpKind::Alloc ||
                bnode.kind == OpKind::Free) {
                continue;
            }

            // A node that reads the tensor it writes is self-modifying
            // (scale(C), or an accumulating gemm C = C + A·B). Never hoist
            // these — the per-iteration update would be lost. ``reads_its_output``
            // covers the always-accumulating ops and nonzero-prefactor einsum/
            // permute/gemm; the explicit input==output scan catches any other op
            // that lists the same tensor as both an input and an output.
            bool self_modifying = reads_its_output(bnode);
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
                // performed inside nested subgraphs — a conditional branch or
                // an inner loop — which never appear in this body's own output
                // lists. Without it, an input mutated only inside a conditional
                // would look invariant and the consumer would be wrongly
                // hoisted out of the loop.
                bool written_in_body = body_writes.count(tid) > 0;
                if (!written_in_body) {
                    auto hit = loop_desc->body->tensors_map().find(loop_desc->body->resolve_alias(tid));
                    if (hit != loop_desc->body->tensors_map().end() && hit->second.tensor_ptr != nullptr) {
                        auto wit = subtree_writers.find(hit->second.tensor_ptr);
                        if (wit != subtree_writers.end() && wit->second > 0) {
                            written_in_body = true;
                        }
                    }
                }
                bool const from_hoisted = hoisted_outputs.count(tid) > 0;
                if (written_in_body && !from_hoisted) {
                    all_inputs_invariant = false;
                    break;
                }
            }

            // Refuse to hoist a producer whose output is written by more than
            // one value-node in the loop subtree. Removing its per-iteration
            // write would change which write wins each iteration — e.g. an
            // einsum that resets C, followed by an in-place op that would then
            // accumulate across iterations instead of starting fresh. (A
            // DiskRead with no inputs is "invariant" by the input check above;
            // this guard stops it being hoisted when something else in the
            // loop overwrites its destination.) Reads of the output are fine.
            bool single_writer_outputs = true;
            for (auto out_tid : bnode.outputs) {
                auto hit = loop_desc->body->tensors_map().find(loop_desc->body->resolve_alias(out_tid));
                if (hit == loop_desc->body->tensors_map().end() || hit->second.tensor_ptr == nullptr) {
                    single_writer_outputs = false; // can't prove single-writer → be safe
                    break;
                }
                auto wit = subtree_writers.find(hit->second.tensor_ptr);
                if (wit == subtree_writers.end() || wit->second != 1) {
                    single_writer_outputs = false;
                    break;
                }
            }
            if (!single_writer_outputs) {
                continue;
            }

            // Refuse to hoist a producer whose output is read by an *earlier*
            // body node. That earlier read observes the value from the previous
            // iteration (the output is loop-carried *through* this producer), so
            // computing it once before the loop would change what the earlier
            // reader sees. Reads by *later* body nodes are fine — they consume
            // this iteration's value, which is loop-invariant once hoisted.
            bool output_read_earlier = false;
            for (auto out_tid : bnode.outputs) {
                for (size_t bj = 0; bj < bi && !output_read_earlier; bj++) {
                    // Use *effective* reads so an earlier control-flow node (a
                    // nested loop / conditional) that reads the output inside its
                    // subtree counts — its own raw input list is empty.
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
                continue;
            }

            if (all_inputs_invariant) {
                invariant[bi] = true;
                for (auto tid : bnode.outputs) {
                    hoisted_outputs.insert(tid);
                }
            }
        }

        // Bail out cheaply if nothing's invariant — otherwise we'd move-from
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
        // IDs are not registered in the parent graph — naively appending the
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
            TensorHandle handle = loop_desc->body->tensor(body_tid);
            // If the parent already has a TensorId for this underlying buffer,
            // reuse it rather than minting a fresh one. The buffer's identity is
            // its pointer; registering a *new* id for an already-known buffer
            // would hide the write-after-write / read-after-write relationship
            // between the hoisted node and the parent nodes that touch the same
            // tensor (the scheduler keys on TensorId), so a later pass like
            // Reorder could swap them and the wrong write would win.
            TensorId parent_tid = 0;
            bool     reused     = false;
            if (handle.tensor_ptr != nullptr) {
                for (auto const &[tid, h] : graph.tensors_map()) {
                    if (h.tensor_ptr == handle.tensor_ptr) {
                        parent_tid = tid;
                        reused     = true;
                        break;
                    }
                }
            }
            if (!reused) {
                parent_tid = graph.register_tensor(std::move(handle));
            }
            id_remap[body_tid] = parent_tid;
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
        // inputs of the Loop node below — without this data-flow edge,
        // ``topological_sort`` has nothing tying the hoisted producers
        // to the loop's body.
        std::vector<TensorId> hoisted_output_ids;
        for (auto const &h : hoisted) {
            for (auto tid : h.outputs)
                hoisted_output_ids.push_back(tid);
        }

        // Update body to only contain remaining (non-hoisted) nodes
        body_nodes = std::move(remaining);

        // Insert hoisted nodes directly BEFORE the loop in the parent's
        // ``nodes`` vector. ``topological_sort`` (Kahn's algorithm)
        // processes nodes in their current order when building dataflow
        // edges and again when ties exist in the ready queue. Appending
        // at the end leaves the loop ahead of its newly-introduced
        // producers, which both prevents the edge from being recorded
        // (the writer is seen after the reader) and biases the queue
        // toward the wrong order.
        size_t const n_hoisted = hoisted.size();
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

    if (_num_hoisted > 0) {
        // Nodes were appended; need topological_sort to place them correctly
        graph.topological_sort();
    }

    return _num_hoisted > 0;
}

} // namespace einsums::compute_graph::passes
