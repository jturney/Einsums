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

void StreamAssignment::reset_stats() {
    _num_assigned = 0;
}

bool StreamAssignment::run(Graph &graph) {
    PassCounter const assigned{_num_assigned};
    for (auto &node : graph.nodes()) {
        int new_stream = 0; // default: compute stream

        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            new_stream = 1; // transfer stream
        }

        if (node.stream_id != new_stream) {
            node.stream_id = new_stream;
            _num_assigned++;
        }
    }

    if (assigned.moved()) {
        EINSUMS_LOG_INFO("StreamAssignment: assigned {} nodes to transfer stream", _num_assigned);
        report(1, fmt::format("assigned {} node(s) to the transfer stream for overlap", _num_assigned));
    }

    return assigned.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
