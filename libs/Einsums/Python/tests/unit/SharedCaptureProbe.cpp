//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A stage module that emits into the caller's graph: the C++ half of the
// shared-graph contract, which is decision 2 of the hybrid framework.
//
// Separate from StageModuleProbe.cpp on purpose. That file is compiled at two
// language levels to test the fingerprint refusal, which means it has to stay
// on a narrow ledge of minimal includes; pulling ComputeGraph into it made the
// C++23 build fail to dlopen on a template instantiation the C++20 library
// never emitted, and a module that cannot load cannot be refused, so the skew
// test silently stopped running. This file is built once, at the project's own
// standard, and is free to include whatever it needs.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Python/StageModule.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;
namespace cg = einsums::compute_graph;

namespace {

/// Emit one einsum node into whatever capture is already running.
///
/// The whole of the shared-graph contract on this side, and the point is what
/// it does NOT do: no Graph of its own, no CaptureGuard, no execute. `cg::` ops
/// record into the ambient CaptureContext when there is one and run eagerly
/// when there is not, so a stage written this way joins the caller's graph
/// without knowing whether it has one.
///
/// That property is why the capture context has to be thread-local state
/// inside ONE libEinsums. A stage module bound to a second copy would find its
/// own empty context here, emit into a graph nobody executes, and return having
/// quietly computed nothing - which is the failure einsums.sealed exists to
/// make impossible.
void stage_shared_capture_einsum(einsums::RuntimeTensor<double> const &A, einsums::RuntimeTensor<double> const &B,
                                 einsums::RuntimeTensor<double> *C) {
    cg::einsum("ik;kj->ij", C, A, B);
}

} // namespace

PYBIND11_MODULE(einsums_shared_capture_probe, m) {
    EINSUMS_STAGE_MODULE(m, "einsums_shared_capture_probe");

    m.def("stage_shared_capture_einsum", &stage_shared_capture_einsum, py::arg("A"), py::arg("B"), py::arg("C"),
          "Emit one einsum node into the ambient capture, if any.");

    // Thread-local test introspection, not bound anywhere else. Lets the test
    // assert the C++-emitted node took the intended kernel route rather than
    // silently falling back to the generic loop - a fallback still computes the
    // right answer, so the result alone cannot detect it.
    m.def(
        "last_dispatch_route", []() { return std::string(cg::dispatch::last_dispatch_route()); },
        "Kernel route the most recent string einsum on this thread selected.");
}
