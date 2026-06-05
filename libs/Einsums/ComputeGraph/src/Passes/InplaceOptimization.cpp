//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/Logging.hpp>

#include <unordered_map>

namespace einsums::compute_graph::passes {

namespace {

// Count single-producer/single-consumer intermediate candidates in one
// graph, adding to @p candidates.
void count_candidates(Graph &graph, size_t &candidates) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    std::unordered_map<TensorId, size_t> write_count;
    std::unordered_map<TensorId, size_t> read_count;

    for (auto const &node : nodes) {
        // Alloc/Free nodes are lifetime markers, not real writers/readers
        if (node.kind == OpKind::Alloc || node.kind == OpKind::Free)
            continue;

        for (auto tid : node.outputs) {
            write_count[tid]++;
        }
        for (auto tid : node.inputs) {
            read_count[tid]++;
        }
    }

    for (auto const &[tid, handle] : tensors) {
        if (!handle.is_intermediate)
            continue;

        auto wc = write_count.count(tid) ? write_count[tid] : 0;
        auto rc = read_count.count(tid) ? read_count[tid] : 0;

        if (wc == 1 && rc == 1) {
            candidates++;
            EINSUMS_LOG_INFO("InplaceOptimization: candidate tensor '{}' (id={}, {} bytes) — "
                             "single producer, single consumer",
                             handle.name, tid, handle.total_bytes());
        }
    }
}

void accumulate(Graph &graph, size_t &candidates) {
    count_candidates(graph, candidates);
    graph.for_each_subgraph([&](Graph &sub) { accumulate(sub, candidates); });
}

} // namespace

bool InplaceOptimization::run(Graph &graph) {
    // Aggregate candidate count over the whole graph tree so body-resident
    // intermediates are considered too. recurse_into_subgraphs() stays
    // false — we aggregate here rather than being re-run per sub-graph.
    _num_candidates = 0;
    accumulate(graph, _num_candidates);

    if (_num_candidates > 0) {
        report(1, fmt::format("found {} in-place candidate tensor(s) (single producer + single consumer)", _num_candidates));
    }

    // Analysis only — does not modify the graph
    return false;
}

} // namespace einsums::compute_graph::passes
