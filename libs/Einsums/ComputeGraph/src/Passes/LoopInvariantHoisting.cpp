//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/Logging.hpp>

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

        // Move invariant nodes from body to parent graph, before the loop node
        std::vector<Node> hoisted;
        std::vector<Node> remaining;
        for (size_t bi = 0; bi < body_nodes.size(); bi++) {
            if (invariant[bi]) {
                hoisted.push_back(std::move(body_nodes[bi]));
                EINSUMS_LOG_INFO("LoopInvariantHoisting: hoisting '{}' out of loop '{}'", hoisted.back().label, nodes[loop_idx].label);
                _num_hoisted++;
            } else {
                remaining.push_back(std::move(body_nodes[bi]));
            }
        }

        if (hoisted.empty())
            continue;

        // Update body to only contain remaining (non-hoisted) nodes
        body_nodes = std::move(remaining);

        // Collect hoisted nodes to insert after the main loop completes
        // (inserting during iteration would invalidate pointers/iterators)
        for (auto &h : hoisted) {
            nodes.push_back(std::move(h));
        }
        // Note: hoisted nodes are appended at the end; topological_sort
        // will place them correctly before the loop node on next sort.
    }

    if (_num_hoisted > 0) {
        // Nodes were appended; need topological_sort to place them correctly
        graph.topological_sort();
    }

    return _num_hoisted > 0;
}

} // namespace einsums::compute_graph::passes
