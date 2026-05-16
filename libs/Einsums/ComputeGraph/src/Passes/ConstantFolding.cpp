//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_set>
#include <vector>

namespace einsums::compute_graph::passes {

bool ConstantFolding::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        _num_folded = 0;
        return false;
    }

    // Build the set of all tensors that are written by any node
    std::unordered_set<TensorId> written_tensors;
    for (auto const &node : nodes) {
        for (auto tid : node.outputs) {
            written_tensors.insert(tid);
        }
    }

    // A node is foldable if ALL its inputs are NOT written by any node
    // (i.e., they are external constants) AND it's not a control flow node.
    // We iterate in topological order and propagate: once a node is folded,
    // its outputs become constants too.

    std::unordered_set<TensorId> constant_tensors;
    // Initially, only graph-owned intermediates (is_intermediate=true) that are
    // never written by any node are treated as constant. User-owned tensors
    // (is_intermediate=false) are NOT assumed constant because they may change
    // between loop iterations or between successive execute() calls.
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (written_tensors.find(tid) == written_tensors.end() && handle.is_intermediate) {
            constant_tensors.insert(tid);
        }
    }

    _num_folded = 0;
    std::vector<bool> folded(nodes.size(), false);

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto &node = nodes[idx];

        // Skip control flow, memory management, I/O, communication, allocation, and user-defined nodes.
        // These have side effects and should never be folded.
        if (node.kind == OpKind::Conditional || node.kind == OpKind::Loop || node.kind == OpKind::Alloc || node.kind == OpKind::Free ||
            node.kind == OpKind::DiskRead || node.kind == OpKind::DiskWrite || node.kind == OpKind::Custom ||
            node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost || node.kind == OpKind::Allreduce ||
            node.kind == OpKind::Broadcast || node.kind == OpKind::Allgather || node.kind == OpKind::Scatter ||
            node.kind == OpKind::Barrier || node.kind == OpKind::Materialize || node.kind == OpKind::Initialize) {
            continue;
        }

        // Check if all inputs are constant
        bool all_inputs_constant = true;
        for (auto tid : node.inputs) {
            if (constant_tensors.find(tid) == constant_tensors.end()) {
                all_inputs_constant = false;
                break;
            }
        }

        if (!all_inputs_constant) {
            continue;
        }

        // This node's inputs are all constant — execute it now and replace with no-op
        EINSUMS_LOG_INFO("ConstantFolding: folding node {} ({})", node.id, node.label);
        node.execute();

        // Replace executor with no-op
        node.execute = []() {};

        // Mark its outputs as constant (they won't change on replay)
        for (auto tid : node.outputs) {
            constant_tensors.insert(tid);
        }

        folded[idx] = true;
        _num_folded++;
    }

    if (_num_folded > 0) {
        graph.mark_sorted();
    }

    return _num_folded > 0;
}

} // namespace einsums::compute_graph::passes
