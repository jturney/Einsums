//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/Logging.hpp>

#include <memory>
#include <variant>

namespace einsums::compute_graph::passes {

bool CommunicationScheduling::run(Graph &graph) {
    _num_scheduled = 0;

    if (comm::world_size() <= 1)
        return false;

    auto       &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Transform synchronous Allreduce nodes into async two-phase nodes.
    // The DataflowExecutor runs async_start immediately (non-blocking iallreduce),
    // schedules independent work, then runs async_finish (wait) before any node
    // that reads the allreduced tensor.
    //
    // This enables overlapping communication with computation when the graph
    // has independent work available after the allreduce.

    bool modified = false;

    for (auto &node : nodes) {
        if (node.kind != OpKind::Allreduce)
            continue;

        // Skip nodes that already have async phases
        if (node.async_start && node.async_finish)
            continue;

        // Get the tensor handle for the allreduced tensor
        auto const *desc = std::get_if<CommDescriptor>(&node.op_data);
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

        node.async_start = [iallreduce_fn, request]() { *request = iallreduce_fn(); };

        node.async_finish = [request, name = handle.name]() {
            request->wait();
            EINSUMS_LOG_DEBUG("CommunicationScheduling: async allreduce({}) completed", name);
        };

        // Clear the synchronous execute — the DataflowExecutor uses
        // async_start/async_finish instead when both are set.
        node.execute = nullptr;

        _num_scheduled++;
        modified = true;

        EINSUMS_LOG_INFO("CommunicationScheduling: converted allreduce({}) to async (iallreduce + wait)", handle.name);
    }

    return modified;
}

} // namespace einsums::compute_graph::passes
