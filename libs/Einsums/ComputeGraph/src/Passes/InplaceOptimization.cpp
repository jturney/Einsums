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

bool InplaceOptimization::run(Graph &graph) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();
    _num_candidates     = 0;

    // Count writers and readers for each tensor
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

    // Find candidates: intermediate tensors with exactly 1 writer and 1 reader
    for (auto const &[tid, handle] : tensors) {
        if (!handle.is_intermediate)
            continue;

        auto wc = write_count.count(tid) ? write_count[tid] : 0;
        auto rc = read_count.count(tid) ? read_count[tid] : 0;

        if (wc == 1 && rc == 1) {
            _num_candidates++;
            EINSUMS_LOG_INFO("InplaceOptimization: candidate tensor '{}' (id={}, {} bytes) — "
                             "single producer, single consumer",
                             handle.name, tid, handle.total_bytes());
        }
    }

    // Analysis only — does not modify the graph
    return false;
}

} // namespace einsums::compute_graph::passes
