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

#include <Einsums/Config/ABI.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>
#include <Einsums/Utilities/SetEnv.hpp>

#include <cstdint>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // std::vector<std::string> caster for the cli_options argument
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
// Per-category block layout mirrors the runtime's option categories. Each
// spelling below matches a descriptor in the owning module's Options.hpp:
// Einsums/Runtime/Options.hpp, Einsums/Profile/Options.hpp, and so on.
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
    // Helper: append a presence-only flag. A flag carries its meaning in its
    // presence, so ``False`` cannot be expressed by leaving it out - that is
    // what ``None`` means. It asks for the negation instead, spelled by the
    // runtime's own rule so the two cannot drift apart.
    auto opt_flag = [&](char const *attr, char const *flag) {
        py::object const v = rc.attr(attr);
        if (v.is(py::none())) {
            return;
        }
        std::string const name = std::string{"einsums:"} + flag;
        argv.emplace_back("--" + (v.cast<bool>() ? name : einsums::cl::derive_negated_name(name)));
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
        opt_flag("debug_install_signal_handlers", "debug:install-signal-handlers");
        opt_flag("debug_attach_debugger", "debug:attach-debugger");
        opt_flag("debug_diagnostics_on_terminate", "debug:diagnostics-on-terminate");
        opt_flag("debug_crash_handler", "debug:crash-handler");
        opt_string("debug_crash_dump_dir", "debug:crash-dump-dir");
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
        opt_flag("profile_report", "profile:report");
        opt_string("profile_filename", "profile:filename");
        opt_flag("profile_append", "profile:append");
        opt_flag("profile_detailed", "profile:detailed");
        opt_string("profile_save", "profile:save");
        opt_int("profile_port", "profile:port");
        opt_flag("profile_wait_for_viewer", "profile:wait-for-viewer");
    }

    // Tensor Options
    {
        opt_string("scratch_dir", "scratch-dir");
        opt_string("hdf5_file_name", "hdf5-file-name");
        opt_flag("delete_hdf5_files", "delete-hdf5-files");
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
    einsums::set_env_var("OMP_NUM_THREADS", std::to_string(n));
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
        [](std::vector<std::string> const &cli_options) {
            // Idempotent: no-op once the runtime is already running. Reads the
            // user's einsums.rc settings, falling back to defaults if rc was
            // never touched. einsums.rc is a pure-Python sibling; importing it
            // does not recurse back into _core.
            if (einsums::is_running()) {
                return;
            }
            py::module_ const rc = py::module_::import("einsums.rc");
            apply_threads_from_rc(rc);                        // OMP_NUM_THREADS env var
            std::vector<std::string> argv = argv_from_rc(rc); // CLI flags
            // ``--einsums:*`` flags the Python package claimed from the command
            // line at import (einsums.cli_options). Appended AFTER the
            // rc-derived flags so a flag given explicitly on the command line
            // wins over the same setting made through rc.
            argv.insert(argv.end(), cli_options.begin(), cli_options.end());
            einsums::initialize(argv);
            Py_AtExit(&einsums_python_atexit);
        },
        py::arg("cli_options") = std::vector<std::string>{},
        "Start the einsums runtime from einsums.rc if it is not already running.\n"
        "cli_options: extra ``--einsums:*`` flags, appended after the rc-derived\n"
        "ones so they take precedence.");

    m.def(
        "_is_initialized", []() { return einsums::is_running(); }, "Returns True if the einsums runtime has been initialized.");

    // --- sealed worlds ------------------------------------------------------
    //
    // Which libEinsums did THIS extension bind to? A compiled stage module
    // exports the same attribute, and the loader refuses it when the two
    // disagree. Exposed as a plain int rather than a capsule so the comparison,
    // and the error message when it fails, are both trivial in Python.
    //
    // Bound by hand here rather than annotated in <Einsums/Config/ABI.hpp>:
    // that would mean turning on apiary codegen for Config, the lowest module
    // in the tree, to publish four accessors.
    m.attr("__einsums_world__") = reinterpret_cast<std::uintptr_t>(einsums::sealed::world().identity);

    m.def(
        "_world_info",
        []() {
            einsums::sealed::WorldInfo const &w = einsums::sealed::world();
            py::dict                          d;
            d["identity"]           = reinterpret_cast<std::uintptr_t>(w.identity);
            d["config_fingerprint"] = w.config_fingerprint;
            d["layout_fingerprint"] = w.layout_fingerprint;
            d["version"]            = std::string(w.version_string);
            d["version_major"]      = w.version_major;
            d["version_minor"]      = w.version_minor;
            d["version_patch"]      = w.version_patch;
            d["git_commit"]         = std::string(w.git_commit);
            d["compiler"]           = std::string(w.compiler_id);
            d["compiler_major"]     = w.compiler_major;
            d["library_path"]       = std::string(w.library_path);
            return d;
        },
        "Identity and build fingerprint of the libEinsums this extension is bound to.\n"
        "Compared against a stage module's own record to refuse a cross-world load.");

    m.def(
        "_mapped_einsums_libraries", []() { return einsums::sealed::mapped_einsums_libraries(); },
        "Paths of every libEinsums mapped into this process.\n"
        "More than one entry means more than one world; empty means the platform\n"
        "offers no way to enumerate loaded images.");

    m.def(
        "_register_stage_module", [](std::string const &name) { return einsums::sealed::register_stage_module(name.c_str()); },
        py::arg("name"), "Record a stage module against THIS libEinsums. Returns False if already recorded.");

    m.def(
        "_stage_module_registered", [](std::string const &name) { return einsums::sealed::stage_module_registered(name.c_str()); },
        py::arg("name"),
        "Whether a stage module registered against THIS libEinsums.\n"
        "False for a module that bound to a different copy: its registration went\n"
        "into that copy's table, which is what makes this a cross-world check.");

    m.def(
        "_registered_stage_modules", []() { return einsums::sealed::registered_stage_modules(); },
        "Every stage module recorded against THIS libEinsums, in load order.\n"
        "Worth printing when a handshake fails: it names the modules that DID reach\n"
        "this world, against the one that did not.");

    apiary_register_all(m);
}
