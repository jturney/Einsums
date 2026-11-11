//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/StreamAssignment.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

bool StreamAssignment::transform(Graph &graph) {
    PassCounter const assigned{_stats.num_assigned};
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        Node const &node       = graph.nodes()[i];
        int         new_stream = 0; // default: compute stream

        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            new_stream = 1; // transfer stream
        }

        if (node.stream_id != new_stream) {
            NodeSchedule schedule = schedule_of(node);
            schedule.stream_id    = new_stream;
            graph.set_schedule(i, schedule);
            _stats.num_assigned++;
        }
    }

    if (assigned.moved()) {
        EINSUMS_LOG_INFO("StreamAssignment: assigned {} nodes to transfer stream", _stats.num_assigned);
        report(1, fmt::format("assigned {} node(s) to the transfer stream for overlap", _stats.num_assigned));
    }

    return assigned.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
