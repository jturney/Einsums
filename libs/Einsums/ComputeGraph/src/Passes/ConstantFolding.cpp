//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

std::vector<std::string> ConstantFolding::explain() const {
    if (_num_folded == 0) {
        return {};
    }
    return {fmt::format("ConstantFolding: folded {} node(s) to constants", _num_folded)};
}

void ConstantFolding::reset_stats() {
    _num_folded = 0;
}

bool ConstantFolding::run(Graph &graph) {
    PassCounter const folded_count{_num_folded};
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        return false;
    }

    // Every tensor some node writes a value into, through any alias of its buffer.
    //
    // Lifecycle nodes are not writers, and skipping them is what makes this pass able to fire at
    // all: an eagerly created graph-owned tensor gets an Alloc node whose output is that tensor,
    // so counting Alloc made every such tensor non-constant. EscapeAnalysis counts value writers
    // only, and it counts them per buffer: a tensor written only through a view of it was once
    // taken for constant here, and a later reader was folded before the write it depended on.
    auto const writers = EscapeAnalysis::over(graph);

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
        if (writers.writer_count(tid) == 0 && handle.is_intermediate) {
            constant_tensors.insert(tid);
        }
    }

    // A node may only be folded if every tensor it touches has real backing
    // data *right now*, folding executes the node at pass time and bakes
    // the result. ConstantFolding runs before the MaterializationPass, and
    // Materialize nodes only allocate at graph-execution time, so a deferred
    // (shell) tensor has no storage during this pass. This matters
    // especially inside loop bodies, whose workspace tensors are deferred:
    // without this guard, recursing into a body and executing a node that
    // reads/writes a shell tensor would crash. Eager tensors
    // (create_*_tensor) are Materialized from the start and fold normally.
    auto all_tensors_materialized = [&](Node const &node) {
        auto materialized = [&](TensorId tid) {
            auto const *handle = graph.find_tensor(tid);
            return handle != nullptr && handle->alloc_state == AllocState::Materialized;
        };
        return std::ranges::all_of(node.inputs, materialized) && std::ranges::all_of(node.outputs, materialized);
    };

    std::vector<bool> folded(nodes.size(), false);

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto &node = nodes[idx];

        // Skip control flow, memory management, I/O, communication, allocation, and user-defined nodes.
        // These have side effects and should never be folded.
        if (is_infrastructure(node.kind) || is_lifecycle(node.kind) || node.kind == OpKind::Custom) {
            continue;
        }

        // A node whose effect or operand lives in the ParamTable is never a
        // constant. WriteParam carries no tensor inputs at all, so the test
        // below is vacuously satisfied and folding swaps its per-iteration
        // write for a no-op, pinning every downstream slice at the value the
        // pass happened to evaluate. A runtime-bound View is the same mistake
        // from the reading side: its parent really is constant, its slice is
        // not.
        if (node.kind == OpKind::WriteParam || has_runtime_view_bounds(node)) {
            note_skip("node's effect or slice bounds live in the parameter table, not in its tensor operands",
                      fmt::format("node {} ({})", node.id, node.label));
            continue;
        }

        if (!std::ranges::all_of(node.inputs, [&](TensorId tid) { return constant_tensors.contains(tid); })) {
            continue;
        }

        // Don't execute a node whose tensors aren't materialized yet (see above).
        if (!all_tensors_materialized(node)) {
            continue;
        }

        // This node's inputs are all constant, execute it now and replace with no-op
        EINSUMS_LOG_INFO("ConstantFolding: folding node {} ({})", node.id, node.label);
        report(2, fmt::format("fold node {} ({}): all inputs constant, evaluated at compile time", node.id, node.label));
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

    if (folded_count.moved()) {
        graph.mark_sorted();
        report(1, fmt::format("folded {} constant node(s)", _num_folded));
    }

    return folded_count.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
