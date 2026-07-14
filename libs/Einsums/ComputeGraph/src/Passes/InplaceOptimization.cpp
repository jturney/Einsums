//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace einsums::compute_graph::passes {

namespace {

bool is_lifecycle(OpKind kind) {
    return kind == OpKind::Alloc || kind == OpKind::Free || kind == OpKind::Materialize || kind == OpKind::Initialize;
}

/// Consumers whose output may safely alias a (dying) input: element-aligned
/// elementwise ops, where out[i] depends only on the inputs' element i, so
/// writing through the aliased buffer never corrupts a value still to be
/// read. Contractions (Einsum/Gemm/BatchedGemm) and permutes must NOT alias
/// output with input and stay excluded. The pure-overwrite requirement
/// (output not read) is checked separately via the out-tensor-as-input
/// recording convention: an op that reads its destination lists it in
/// Node::inputs.
bool elementwise_alias_safe(OpKind kind) {
    return kind == OpKind::DirectProduct || kind == OpKind::DirectDivision || kind == OpKind::Axpby;
}

struct UseInfo {
    size_t writes{0};
    size_t reads{0};
    size_t writer_idx{SIZE_MAX}; ///< only meaningful when writes == 1
};

/// One storage merge: consumer node `node_idx` writes `dst` while reading
/// `src` for the last time; `dst` will reuse `src`'s buffer.
struct MergePlan {
    size_t   node_idx;
    TensorId dst; ///< output tensor merged away
    TensorId src; ///< dying input whose storage is reused
};

/// Find the first sound merge in `graph`, or nullopt. Called repeatedly
/// until quiescent because each applied merge deletes lifecycle nodes and
/// rewrites ids (graphs are small and merges are rare, so the restart is
/// cheaper than maintaining incremental state).
std::optional<MergePlan> find_merge(Graph &graph) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Control-flow bodies reference parent tensors without listing them in
    // the parent nodes' plain input/output lists (only effective_io sees
    // them), so plain use-counts under-count and a merge could corrupt a
    // tensor a body still reads. Bodies are processed on their own recursion
    // level; the parent level is skipped when control flow is present.
    for (auto const &node : nodes) {
        if (node.kind == OpKind::Loop || node.kind == OpKind::Conditional) {
            return std::nullopt;
        }
        // GPU placement swaps buffers behind slots (device shadows); storage
        // merging is host-only for now.
        if (node.target == Target::GPU || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            return std::nullopt;
        }
    }

    std::unordered_map<TensorId, UseInfo> uses;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (is_lifecycle(nodes[idx].kind)) {
            continue;
        }
        for (auto tid : nodes[idx].outputs) {
            auto &u = uses[tid];
            u.writes++;
            u.writer_idx = idx;
        }
        for (auto tid : nodes[idx].inputs) {
            uses[tid].reads++;
        }
    }

    // Tensors somebody views: their storage must not be repurposed.
    std::unordered_set<TensorId> view_targets;
    for (auto const &[tid, handle] : tensors) {
        if (handle.aliases != 0) {
            view_targets.insert(handle.aliases);
        }
    }

    auto mergeable_intermediate = [&](TensorId tid) {
        auto it = tensors.find(tid);
        return it != tensors.end() && it->second.is_intermediate && it->second.aliases == 0 && !view_targets.contains(tid);
    };

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        Node const &node = nodes[idx];
        if (!elementwise_alias_safe(node.kind) || node.outputs.size() != 1) {
            continue;
        }

        TensorId const dst = node.outputs[0];

        // Pure overwrite: an op that reads its destination lists it as an
        // input (the gemm/direct_product recording convention).
        if (std::ranges::find(node.inputs, dst) != node.inputs.end()) {
            continue;
        }
        if (!mergeable_intermediate(dst)) {
            continue;
        }
        auto const dst_use = uses[dst];
        if (dst_use.writes != 1) {
            continue;
        }
        // An Initialize on dst would be dead after the merge but deleting it
        // is only sound because of the pure-overwrite check above; keep v1
        // simple and skip such candidates instead.
        bool dst_has_init = false;
        for (auto const &n : nodes) {
            if (n.kind == OpKind::Initialize && std::ranges::find(n.outputs, dst) != n.outputs.end()) {
                dst_has_init = true;
                break;
            }
        }
        if (dst_has_init) {
            continue;
        }

        auto const &dst_handle = tensors.at(dst);

        for (auto const src : node.inputs) {
            if (src == dst || !mergeable_intermediate(src)) {
                continue;
            }
            auto const src_use = uses[src];
            // src must die here: one producer, and this node is its only reader.
            if (src_use.writes != 1 || src_use.reads != 1) {
                continue;
            }
            auto const &src_handle = tensors.at(src);
            if (src_handle.dims != dst_handle.dims || src_handle.total_bytes() != dst_handle.total_bytes()) {
                continue;
            }

            return MergePlan{.node_idx = idx, .dst = dst, .src = src};
        }
    }

    return std::nullopt;
}

/// Apply a merge: `dst` disappears from the graph metadata (every reference
/// from the consumer onward becomes `src`), its lifecycle nodes are removed,
/// and its executor slot is durably redirected at `src`'s storage so baked
/// lambdas follow.
void apply_merge(Graph &graph, MergePlan const &plan, size_t &num_merged) {
    auto &nodes = graph.nodes();

    for (auto &node : nodes) {
        std::ranges::replace(node.inputs, plan.dst, plan.src);
        std::ranges::replace(node.outputs, plan.dst, plan.src);
    }

    // dst's Alloc/Materialize (now rewritten to src) duplicate src's own
    // lifecycle; drop the duplicates that originally belonged to dst. After
    // the rewrite they are indistinguishable by id, so match by label.
    auto const &dst_name = graph.tensor(plan.dst).name;
    std::erase_if(nodes, [&](Node const &n) {
        if (n.kind != OpKind::Alloc && n.kind != OpKind::Materialize && n.kind != OpKind::Free) {
            return false;
        }
        return n.label == fmt::format("alloc({})", dst_name) || n.label == fmt::format("materialize({})", dst_name) ||
               n.label == fmt::format("free({})", dst_name);
    });

    graph.redirect_slot(plan.dst, plan.src);
    graph.mark_sorted(); // order-preserving rewrite; position-keyed deps are stale
    num_merged++;
}

void process(Graph &graph, size_t &num_candidates, size_t &num_merged) {
    // Candidate census (kept for introspection parity with the old
    // analysis-only behavior).
    {
        graph.topological_sort();
        std::unordered_map<TensorId, UseInfo> uses;
        for (auto const &node : graph.nodes()) {
            if (is_lifecycle(node.kind)) {
                continue;
            }
            for (auto tid : node.outputs) {
                uses[tid].writes++;
            }
            for (auto tid : node.inputs) {
                uses[tid].reads++;
            }
        }
        for (auto const &[tid, handle] : graph.tensors_map()) {
            if (handle.is_intermediate && uses[tid].writes == 1 && uses[tid].reads == 1) {
                num_candidates++;
            }
        }
    }

    while (auto plan = find_merge(graph)) {
        auto const &src_name = graph.tensor(plan->src).name;
        auto const &dst_name = graph.tensor(plan->dst).name;
        EINSUMS_LOG_INFO("InplaceOptimization: '{}' reuses the storage of dying '{}' ({} bytes saved)", dst_name, src_name,
                         graph.tensor(plan->dst).total_bytes());
        apply_merge(graph, *plan, num_merged);
    }

    graph.for_each_subgraph([&](Graph &sub) { process(sub, num_candidates, num_merged); });
}

} // namespace

bool InplaceOptimization::run(Graph &graph) {
    _num_candidates = 0;
    _num_merged     = 0;

    process(graph, _num_candidates, _num_merged);

    if (_num_merged > 0) {
        report(1, fmt::format("merged {} output buffer(s) into dying elementwise inputs ({} candidate(s) found)", _num_merged,
                              _num_candidates));
    } else if (_num_candidates > 0) {
        report(1, fmt::format("found {} in-place candidate tensor(s), none safely mergeable", _num_candidates));
    }

    return _num_merged > 0;
}

} // namespace einsums::compute_graph::passes
