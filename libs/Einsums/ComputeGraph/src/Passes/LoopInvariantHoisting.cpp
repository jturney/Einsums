//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

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

            // A node that writes to a tensor it also reads is self-modifying
            // (e.g., scale(C) reads and writes C). Never hoist these.
            bool self_modifying = false;
            for (auto out_tid : bnode.outputs) {
                for (auto in_tid : bnode.inputs) {
                    if (out_tid == in_tid) {
                        self_modifying = true;
                        break;
                    }
                }
                // Also check: does the node write to a tensor with no inputs listed
                // but the operation inherently reads it? (scale has no inputs, only outputs)
                // Scale reads the tensor it writes — check by OpKind
                if (bnode.kind == OpKind::Scale || bnode.kind == OpKind::Axpy || bnode.kind == OpKind::Axpby ||
                    bnode.kind == OpKind::ElementTransform) {
                    self_modifying = true;
                }
                if (self_modifying)
                    break;
            }
            if (self_modifying)
                continue;

            bool all_inputs_invariant = true;
            for (auto tid : bnode.inputs) {
                bool const written_in_body = body_writes.count(tid) > 0;
                bool const from_hoisted    = hoisted_outputs.count(tid) > 0;
                if (written_in_body && !from_hoisted) {
                    all_inputs_invariant = false;
                    break;
                }
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
        for (bool v : invariant) {
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
            TensorHandle handle     = loop_desc->body->tensor(body_tid);
            TensorId     parent_tid = graph.register_tensor(std::move(handle));
            id_remap[body_tid]      = parent_tid;
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
