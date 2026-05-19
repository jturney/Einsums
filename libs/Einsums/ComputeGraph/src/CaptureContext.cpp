//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile.hpp>

namespace einsums::compute_graph {

CaptureContext &CaptureContext::current() {
    thread_local CaptureContext instance;
    return instance;
}

void CaptureContext::begin_capture(Graph &graph) {
    if (_capturing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext: already capturing. Nested captures are not supported.");
    }
    profile::Profiler::instance().push(fmt::format("ComputeGraph::capture({})", graph.name()));
    _graph     = &graph;
    _capturing = true;
    _ptr_to_id.clear();
}

void CaptureContext::end_capture() {
    if (!_capturing) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "CaptureContext: not currently capturing.");
    }

    // Reset capture state and balance the profiler push unconditionally —
    // before running validation/finalization that can throw. If validation
    // fails the caller still sees the exception, but the next `with
    // cg.capture(...)` is no longer locked out by a stale `_capturing` flag.
    Graph *g   = _graph;
    _capturing = false;
    _graph     = nullptr;
    _ptr_to_id.clear();
    profile::Profiler::instance().pop();

    g->topological_sort();
    g->validate_shapes_at_capture();
    profile::annotate("num_nodes", static_cast<int64_t>(g->num_nodes()));
    profile::annotate("num_tensors", static_cast<int64_t>(g->num_tensors()));
    register_graph(g);
}

} // namespace einsums::compute_graph
