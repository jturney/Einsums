//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <memory>
#include <variant>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

void CommunicationScheduling::reset_stats() {
    _num_scheduled = 0;
}

bool CommunicationScheduling::transform(Graph &graph) {
    PassCounter const scheduled{_num_scheduled};
    if (comm::world_size() <= 1)
        return false;

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Transform synchronous Allreduce nodes into async two-phase nodes.
    // The DataflowExecutor runs async_start immediately (non-blocking iallreduce),
    // schedules independent work, then runs async_finish (wait) before any node
    // that reads the allreduced tensor.
    //
    // This enables overlapping communication with computation when the graph
    // has independent work available after the allreduce.

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Node const &node = nodes[i];
        if (node.kind != OpKind::Allreduce)
            continue;

        // Skip nodes that already have async phases
        if (node.async_start && node.async_finish)
            continue;

        if (!admits(graph, node)) {
            continue;
        }

        // Get the tensor handle for the allreduced tensor
        auto const *desc = node.op_data.get_if<CommDescriptor>();
        if (!desc)
            continue;

        auto it = tensors.find(desc->tensor_id);
        if (it == tensors.end())
            continue;

        auto const &handle = it->second;
        if (!handle.iallreduce_sum_fn)
            continue;

        // Split into async_start (iallreduce) and async_finish (wait)
        auto iallreduce_fn = handle.iallreduce_sum_fn;
        auto request       = std::make_shared<comm::Request>();

        // The DataflowExecutor runs the two halves when both are set; the serial executors run the
        // synchronous allreduce the node keeps, which is the same reduction without the overlap.
        graph.edit(i, [&](NodeEdit &e) {
            e->async_start  = [iallreduce_fn, request]() { *request = iallreduce_fn(); };
            e->async_finish = [request, name = handle.name]() {
                request->wait();
                EINSUMS_LOG_DEBUG("CommunicationScheduling: async allreduce({}) completed", name);
            };
        });

        _num_scheduled++;

        EINSUMS_LOG_INFO("CommunicationScheduling: converted allreduce({}) to async (iallreduce + wait)", handle.name);
        report(2, fmt::format("split Allreduce('{}') into async iallreduce + wait for compute overlap", handle.name));
    }

    return scheduled.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
