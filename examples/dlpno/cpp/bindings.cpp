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

    // The contract types cross as plain aggregates. They are bound rather than
    // converted from Python dicts so that a field added on one side and not the
    // other is a TypeError at the call, not a silent KeyError deep inside.
    py::class_<dlpno::CouplingPlan>(m, "CouplingPlan")
        .def(py::init<>())
        .def_readwrite("classes", &dlpno::CouplingPlan::classes)
        .def_readwrite("slots_pair", &dlpno::CouplingPlan::slots_pair)
        .def_readwrite("slots_partner", &dlpno::CouplingPlan::slots_partner)
        .def_readwrite("inv_perm", &dlpno::CouplingPlan::inv_perm)
        .def_readwrite("pair", &dlpno::CouplingPlan::pair)
        .def_readwrite("partner", &dlpno::CouplingPlan::partner)
        .def_readwrite("cls", &dlpno::CouplingPlan::cls)
        .def_readwrite("dest_slot", &dlpno::CouplingPlan::dest_slot)
        .def_readwrite("src_slot", &dlpno::CouplingPlan::src_slot)
        .def_readwrite("sign", &dlpno::CouplingPlan::sign)
        .def_readwrite("factor", &dlpno::CouplingPlan::factor)
        .def_readwrite("coupled_pairs", &dlpno::CouplingPlan::coupled_pairs);

    py::class_<dlpno::PnoOverlaps>(m, "PnoOverlaps")
        .def_readonly("S_cls", &dlpno::PnoOverlaps::S_cls)
        .def_readonly("S_T", &dlpno::PnoOverlaps::S_T);

    // The stage entry point. The stage_ prefix is what load_stage_module scans
    // for: the module states which of its symbols are stages rather than
    // leaving the loader to guess from everything callable.
    m.def(
        "stage_compute_pno_overlaps",
        [](std::vector<einsums::RuntimeTensor<double>> const &X_pno, einsums::RuntimeTensor<double> const &S_pao,
           std::vector<std::vector<std::int64_t>> const &lmopair_to_paos, std::vector<std::int64_t> const &n_pno,
           std::vector<std::int64_t> const &bucket_of, std::vector<std::int64_t> const &bucket_dims, py::object const &plan) {
            // The plan arrives as the Python @contract dataclass, not as the
            // bound C++ aggregate: the planner is Python and building a
            // CouplingPlan there would mean the Python backend depended on this
            // module existing. Read the fields off it once, here.
            dlpno::CouplingPlan p;
            p.classes       = plan.attr("classes").cast<decltype(p.classes)>();
            p.slots_pair    = plan.attr("slots_pair").cast<decltype(p.slots_pair)>();
            p.slots_partner = plan.attr("slots_partner").cast<decltype(p.slots_partner)>();
            p.inv_perm      = plan.attr("inv_perm").cast<decltype(p.inv_perm)>();
            p.pair          = plan.attr("pair").cast<decltype(p.pair)>();
            p.partner       = plan.attr("partner").cast<decltype(p.partner)>();
            p.cls           = plan.attr("cls").cast<decltype(p.cls)>();
            p.dest_slot     = plan.attr("dest_slot").cast<decltype(p.dest_slot)>();
            p.src_slot      = plan.attr("src_slot").cast<decltype(p.src_slot)>();
            p.sign          = plan.attr("sign").cast<decltype(p.sign)>();
            p.factor        = plan.attr("factor").cast<decltype(p.factor)>();
            p.coupled_pairs = plan.attr("coupled_pairs").cast<decltype(p.coupled_pairs)>();

            return dlpno::compute_pno_overlaps(X_pno, S_pao, lmopair_to_paos, n_pno, bucket_of, bucket_dims, p);
        },
        py::arg("X_pno"), py::arg("S_pao"), py::arg("lmopair_to_paos"), py::arg("n_pno"), py::arg("bucket_of"), py::arg("bucket_dims"),
        py::arg("plan"), "Overlaps between every coupled pair of PNO bases, scaled by the Fock element.");
}
