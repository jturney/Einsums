//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// einsums._core entry point.
//
// This file is not auto-generated. It owns the runtime startup/shutdown
// boilerplate and the PYBIND11_MODULE block. The list of per-module
// register functions is the only piece that varies with the enabled module
// set, and that's pulled in from a tiny auto-generated header,
// ``Einsums/Python/Detail/PyEinsumsModules.hpp``, that ``einsums_finalize_pybind``
// writes at config time. Splitting the static and generated halves keeps this
// file linted, formatted, and IDE-indexed like ordinary C++ sources.

#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>

// Auto-generated header: declares ``apiary_register_<Module>(py::module_ &)``
// for every enabled module and defines the inline aggregator
// ``apiary_register_all(py::module_ &)`` that calls them in turn.
#include <Einsums/Python/Detail/PyEinsumsModules.hpp>

namespace py = pybind11;

namespace {

// Translate the user's einsums.rc settings into the argv vector that
// einsums::initialize expects. Unknown / None entries are skipped so the
// C++ runtime falls back to its built-in defaults. The exact CLI flag
// spellings are runtime-internal to keep this mapping in lockstep with
// libs/Einsums/Runtime configuration.
//
// Per-category block layout mirrors the runtime's cl::OptionCategory
// groupings. See libs/Einsums/RuntimeConfiguration/src/RuntimeConfiguration.cpp
// and the per-module InitModule.cpp files in BufferAllocator, Tensor, and so on.
std::vector<std::string> argv_from_rc(py::module_ const &rc) {
    std::vector<std::string> argv;
    argv.emplace_back("einsums-python");

    // Helper: append a string-valued ``--einsums:<flag>=<value>`` if the
    // attribute is not None. Captures rc + argv by reference.
    auto opt_string = [&](char const *attr, char const *flag) {
        py::object const v = rc.attr(attr);
        if (!v.is(py::none())) {
            argv.emplace_back(std::string{"--einsums:"} + flag + "=" + v.cast<std::string>());
        }
    };
    // Helper: append an integer-valued ``--einsums:<flag>=<int>`` if the
    // attribute is not None.
    auto opt_int = [&](char const *attr, char const *flag) {
        py::object const v = rc.attr(attr);
        if (!v.is(py::none())) {
            argv.emplace_back(std::string{"--einsums:"} + flag + "=" + std::to_string(v.cast<int>()));
        }
    };
    // Helper: append a presence-only ``--einsums:<flag>`` when the attribute
    // is set to a truthy value. Mirrors how the runtime's cl::Flag options
    // ignore any value and just check for the flag's presence.
    auto opt_flag = [&](char const *attr, char const *flag) {
        py::object const v = rc.attr(attr);
        if (!v.is(py::none()) && v.cast<bool>()) {
            argv.emplace_back(std::string{"--einsums:"} + flag);
        }
    };
    // Helper: append a ``--einsums:<flag>=<true|false>`` for cl::Opt<bool>
    // options that take an explicit value. These are rare; profile:no-append
    // is one.
    auto opt_bool_value = [&](char const *attr, char const *flag) {
        py::object const v = rc.attr(attr);
        if (!v.is(py::none())) {
            argv.emplace_back(std::string{"--einsums:"} + flag + "=" + (v.cast<bool>() ? "true" : "false"));
        }
    };

    // Buffer Allocator
    {
        opt_string("buffer_size", "buffer-size");
        opt_string("work_buffer_size", "work-buffer-size");
    }

    // ComputeGraph Passes
    {
        opt_string("pass_disable", "pass:disable");
        opt_flag("pass_analyze", "pass:analyze");
        opt_flag("pass_verbose", "pass:verbose");
    }

    // Debug
    {
        opt_flag("debug_no_install_signal_handlers", "debug:no-install-signal-handlers");
        opt_flag("debug_no_attach_debugger", "debug:no-attach-debugger");
        opt_flag("debug_no_diagnostics_on_terminate", "debug:no-diagnostics-on-terminate");
    }

    // HPTT
    { opt_string("hptt_selection_method", "hptt:selection-method"); }

    // Logging
    {
        // Log level is a LogLevel enum on the Python side; extract .value
        // for the integer the runtime parser expects.
        py::object const loglevel = rc.attr("log_level");
        if (!loglevel.is(py::none())) {
            int const lv = loglevel.attr("value").cast<int>();
            argv.emplace_back("--einsums:log:level=" + std::to_string(lv));
        }

        opt_string("log_destination", "log:destination");
        opt_string("log_format", "log:format");
    }

    // Profile
    {
        opt_flag("profile_no_report", "profile:no-report");
        opt_string("profile_filename", "profile:filename");
        opt_bool_value("profile_no_append", "profile:no-append");
        opt_flag("profile_detailed", "profile:detailed");
        opt_string("profile_save", "profile:save");
        opt_int("profile_port", "profile:port");
        opt_flag("profile_wait_for_viewer", "profile:wait-for-viewer");
    }

    // Tensor Options
    {
        opt_string("scratch_dir", "scratch-dir");
        opt_string("hdf5_file_name", "hdf5-file-name");
        opt_flag("no_delete_hdf5_files", "no-delete-hdf5-files");
    }

    return argv;
}

// einsums has no CLI flag for thread count; OpenMP picks up OMP_NUM_THREADS
// from the environment on first parallel region. Honor rc.threads by
// setting that env var before einsums::initialize fires off any worker pool
// setup. No-op when rc.threads is None.
void apply_threads_from_rc(py::module_ const &rc) {
    py::object const v = rc.attr("threads");
    if (v.is(py::none())) {
        return;
    }
    int const n = v.cast<int>();
    setenv("OMP_NUM_THREADS", std::to_string(n).c_str(), /*overwrite=*/1);
}

// Py_AtExit hook. Runs at interpreter teardown, which is later than module
// destruction. The GIL is dropped and Python objects are gone, so this is
// C++-only. Do not touch py::* state here.
extern "C" void einsums_python_atexit() {
    if (einsums::is_running()) {
        einsums::finalize();
    }
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Einsums Python bindings (autogenerated; see einsums_finalize_pybind).";

    // NOTE: importing _core only registers the bindings. It deliberately
    // does not start the runtime. Runtime startup is deferred to
    // _initialize_from_rc() below, called lazily on the first real use from
    // the Python package; see einsums/__init__.py::_ensure_initialized. This
    // decouples "the extension is loaded", which gives `import einsums` or an
    // external module receiving an einsums object the registered types, from
    // "the runtime is up", which must wait until after einsums.rc is configured.
    m.def(
        "_initialize_from_rc",
        []() {
            // Idempotent: no-op once the runtime is already running. Reads the
            // user's einsums.rc settings, falling back to defaults if rc was
            // never touched. einsums.rc is a pure-Python sibling; importing it
            // does not recurse back into _core.
            if (einsums::is_running()) {
                return;
            }
            py::module_ const rc = py::module_::import("einsums.rc");
            apply_threads_from_rc(rc);                              // OMP_NUM_THREADS env var
            std::vector<std::string> const argv = argv_from_rc(rc); // CLI flags
            einsums::initialize(argv);
            Py_AtExit(&einsums_python_atexit);
        },
        "Start the einsums runtime from einsums.rc if it is not already running.");

    m.def(
        "_is_initialized", []() { return einsums::is_running(); }, "Returns True if the einsums runtime has been initialized.");

    apiary_register_all(m);
}
