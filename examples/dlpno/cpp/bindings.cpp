//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// The stage module: the handshake, and one entry point per stage.
///
/// Everything here is what `python -m einsums.stages promote` will generate.
/// It is hand-written for M3 so the generator has something to be diffed
/// against, and so the parts that are awkward are discovered before a tool is
/// built to emit them.

#include <Einsums/Python/StageModule.hpp>

#include <dlpno/Casters.hpp>
#include <dlpno/PnoOverlaps.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dlpno_stages, m) {
    m.doc() = "C++ backends for the DLPNO stages.";

    // Must come first: einsums.stages.load_stage_module refuses to read
    // anything out of a module that has not registered against the same
    // libEinsums it is bound to.
    EINSUMS_STAGE_MODULE(m, "dlpno_stages");

    // PnoOverlaps is returned, so it is a real bound type. CouplingPlan is
    // only ever an argument and crosses through the caster in Casters.hpp,
    // which is why it needs no class_ here.
    py::class_<dlpno::PnoOverlaps>(m, "PnoOverlaps")
        .def_readonly("S_cls", &dlpno::PnoOverlaps::S_cls)
        .def_readonly("S_T", &dlpno::PnoOverlaps::S_T);

    // The stage entry point. The stage_ prefix is what load_stage_module scans
    // for: the module states which of its symbols are stages rather than
    // leaving the loader to guess from everything callable.
    //
    // One line, and its signature is the C++ API verbatim. That is the shape
    // promote should emit; the twelve-field read this used to be lived here
    // only because the caster had not been written yet.
    m.def("stage_compute_pno_overlaps", &dlpno::compute_pno_overlaps, py::arg("X_pno"), py::arg("S_pao"), py::arg("lmopair_to_paos"),
          py::arg("n_pno"), py::arg("bucket_of"), py::arg("bucket_dims"), py::arg("plan"),
          "Overlaps between every coupled pair of PNO bases, scaled by the Fock element.");
}
