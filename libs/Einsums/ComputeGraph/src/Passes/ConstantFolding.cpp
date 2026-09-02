//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

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

    // Every tensor some node WRITES A VALUE INTO.
    //
    // Lifecycle nodes are skipped, and skipping them is what makes this pass
    // able to fire at all. An eagerly created graph-owned tensor gets an Alloc
    // node whose output is that tensor, so counting Alloc as a writer made
    // every such tensor non-constant, and a deferred one has no Alloc but is
    // not materialized, which the guard further down rejects. Between them the
    // two creation paths left no tensor this pass could ever call constant,
    // which is why nothing in the tree had ever been observed to fold.
    //
    // `is_lifecycle` already says this is the rule: it documents its members as
    // producing no value of their own and says passes resolving readers and
    // writers skip them. This one was hand-rolling the set instead.
    std::unordered_set<TensorId> written_tensors;
    for (auto const &node : nodes) {
        if (is_lifecycle(node.kind)) {
            continue;
        }
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
            auto it = graph.tensors_map().find(tid);
            return it != graph.tensors_map().end() && it->second.alloc_state == AllocState::Materialized;
        };
        for (auto tid : node.inputs) {
            if (!materialized(tid)) {
                return false;
            }
        }
        for (auto tid : node.outputs) {
            if (!materialized(tid)) {
                return false;
            }
        }
        return true;
    };

    std::vector<bool> folded(nodes.size(), false);

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto &node = nodes[idx];

        // Skip control flow, memory management, I/O, communication, allocation, and user-defined nodes.
        // These have side effects and should never be folded.
        if (is_control_flow(node.kind) || node.kind == OpKind::Alloc || node.kind == OpKind::Free || node.kind == OpKind::DiskRead ||
            node.kind == OpKind::DiskWrite || node.kind == OpKind::Custom || node.kind == OpKind::HostToDevice ||
            node.kind == OpKind::DeviceToHost || node.kind == OpKind::Allreduce || node.kind == OpKind::Broadcast ||
            node.kind == OpKind::Allgather || node.kind == OpKind::Scatter || node.kind == OpKind::Barrier ||
            node.kind == OpKind::Materialize || node.kind == OpKind::Initialize) {
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

        // Don't execute a node whose tensors aren't materialized yet (see above).
        if (!all_tensors_materialized(node)) {
            continue;
        }

        // This node's inputs are all constant, execute it now and replace with no-op
        EINSUMS_LOG_INFO("ConstantFolding: folding node {} ({})", node.id, node.label);
        report(2, fmt::format("fold node {} ({}) — all inputs constant, evaluated at compile time", node.id, node.label));
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
