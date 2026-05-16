//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Platform.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationInsertion.hpp>
#include <Einsums/Logging.hpp>

#include <ranges>

namespace einsums::compute_graph::passes {

bool CommunicationInsertion::run(Graph &graph) {
    _num_inserted = 0;

    // On single rank (mock), no communication is needed.
    if (comm::world_size() <= 1)
        return false;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Walk nodes and identify where allreduce is needed.
    // Heuristic: if a compute node reads from a distributed (non-replicated) tensor
    // and writes to a replicated output, the output holds a partial sum that needs
    // allreduce to combine results across ranks.

    struct Insertion {
        size_t position; // Insert AFTER this node index
        Node   node;
    };
    std::vector<Insertion> insertions;

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        auto const &node = nodes[idx];

        // Skip infrastructure nodes — process all compute nodes (Einsum, Scale, Axpy, Permute, etc.)
        // BatchedGemm is intentionally not listed: it arrives here only if a user
        // mixes GEMMBatching with the distribution pipeline, which is unsupported
        // (see libs/Einsums/ComputeGraph/docs/gemm_batching.rst). On non-distributed
        // inputs the has_distributed_input check below short-circuits.
        if (node.kind == OpKind::Materialize || node.kind == OpKind::Initialize || node.kind == OpKind::Allreduce ||
            node.kind == OpKind::Broadcast || node.kind == OpKind::Allgather || node.kind == OpKind::Scatter ||
            node.kind == OpKind::Barrier || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost ||
            node.kind == OpKind::DiskRead || node.kind == OpKind::DiskWrite || node.kind == OpKind::Loop ||
            node.kind == OpKind::Conditional)
            continue;

        // Check if any input is distributed (non-replicated)
        bool has_distributed_input = false;
        for (auto tid : node.inputs) {
            auto it = tensors.find(tid);
            if (it != tensors.end() && it->second.is_distributed && !it->second.is_replicated) {
                has_distributed_input = true;
                break;
            }
        }

        if (!has_distributed_input)
            continue;

        // For each replicated output, insert an allreduce
        for (auto tid : node.outputs) {
            auto it = tensors.find(tid);
            if (it == tensors.end())
                continue;

            auto const &handle = it->second;

            // Output is replicated (or not marked distributed) → partial sum needs allreduce
            if (!handle.is_distributed || handle.is_replicated) {
                Node ar_node;
                ar_node.kind    = OpKind::Allreduce;
                ar_node.label   = fmt::format("allreduce({})", handle.name);
                ar_node.inputs  = {tid};
                ar_node.outputs = {tid}; // In-place

                CommDescriptor desc;
                desc.tensor_id  = tid;
                desc.size_bytes = handle.total_bytes();
                ar_node.op_data = desc;

                // Use the type-erased allreduce function from the handle
                auto ar_fn      = handle.allreduce_sum_fn;
                ar_node.execute = [ar_fn, name = handle.name]() {
                    if (ar_fn) {
                        ar_fn();
                        EINSUMS_LOG_DEBUG("allreduce({}) completed", name);
                    }
                };

                insertions.push_back({.position = idx, .node = std::move(ar_node)});
                _num_inserted++;

                EINSUMS_LOG_INFO("CommunicationInsertion: inserting allreduce({}) after node '{}' ({} bytes)", handle.name, node.label,
                                 handle.total_bytes());
            }
        }
    }

    if (insertions.empty())
        return false;

    // Insert in reverse order so positions don't shift
    for (auto &insertion : std::views::reverse(insertions)) {
        nodes.insert(nodes.begin() + static_cast<ptrdiff_t>(insertion.position + 1), std::move(insertion.node));
    }

    graph.mark_sorted();
    return true;
}

} // namespace einsums::compute_graph::passes
