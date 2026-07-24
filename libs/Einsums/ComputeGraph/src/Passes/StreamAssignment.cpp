//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/StreamAssignment.hpp>
#include <Einsums/Logging.hpp>

namespace einsums::compute_graph::passes {

void StreamAssignment::reset_stats() {
    _num_assigned = 0;
}

bool StreamAssignment::run(Graph &graph) {
    // Per-apply counters: compare against entry values, not zero. The
    // recursive driver calls run() once per subgraph and reset_stats() runs
    // only once per apply, so `_num_x > 0` would report this graph as
    // modified whenever ANY earlier subgraph changed something.
    size_t const num_assigned_at_entry = _num_assigned;
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

    if (_num_assigned > num_assigned_at_entry) {
        EINSUMS_LOG_INFO("StreamAssignment: assigned {} nodes to transfer stream", _num_assigned);
        report(1, fmt::format("assigned {} node(s) to the transfer stream for overlap", _num_assigned));
    }

    return _num_assigned > num_assigned_at_entry;
}

} // namespace einsums::compute_graph::passes
