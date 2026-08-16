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
#include <Einsums/Options/Parse.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>
#include <Einsums/Utilities/SetEnv.hpp>

#include <cstdint>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h> // std::vector<std::string> caster for the cli_options argument
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Auto-generated header: declares ``apiary_register_<Module>(py::module_ &)``
// for every enabled module and defines the inline aggregator
// ``apiary_register_all(py::module_ &)`` that calls them in turn.
#include <Einsums/Python/Detail/PyEinsumsModules.hpp>

namespace py = pybind11;

namespace {

// The Python attribute name an option's command-line name maps to: drop a
// leading ``einsums:``, then turn every ``:`` and ``-`` into an underscore.
// So ``einsums:profile:report`` is ``profile_report`` and ``einsums:row-major``
// is ``row_major``. It is the only relationship between the two spellings, and
// einsums/rc.py is generated from the same rule.
std::string rc_attribute_name(std::string_view long_name) {
    constexpr std::string_view prefix = "einsums:";

    std::string_view rest = long_name;
    if (rest.starts_with(prefix)) {
        rest.remove_prefix(prefix.size());
    }

    std::string out(rest);
    for (char &c : out) {
        if (c == ':' || c == '-') {
            c = '_';
        }
    }
    return out;
}

// An integer setting, whether it was written as a plain int or as one of
// rc.py's enums (``LogLevel.INFO``). An Enum member carries the number in
// ``.value``; a plain int has no such attribute.
std::int64_t rc_int_value(py::object const &v) {
    py::object const n = py::hasattr(v, "value") ? v.attr("value") : v;
    return n.cast<std::int64_t>();
}

// Every field in rc.py carries a type annotation and nothing else in the module
// does, so the annotations are exactly the option surface plus ``threads``. A
// name the registry does not claim is therefore a field describing an option
// that no longer exists - the drift this whole path was written to prevent - so
// it is refused loudly rather than skipped in silence, which is what the
// hand-written mapping used to do.
void reject_unknown_rc_fields(py::module_ const &rc, std::set<std::string> const &claimed) {
    if (!py::hasattr(rc, "__annotations__")) {
        return;
    }
    py::dict const annotations = rc.attr("__annotations__");

    std::string orphans;
    for (auto const item : annotations) {
        auto const name = item.first.cast<std::string>();
        // ``threads`` is not an option: it has no flag and routes through
        // OMP_NUM_THREADS, see apply_threads_from_rc.
        if (name == "threads" || name.starts_with("_") || claimed.contains(name)) {
            continue;
        }
        if (!orphans.empty()) {
            orphans += ", ";
        }
        orphans += name;
    }

    if (!orphans.empty()) {
        throw std::runtime_error("einsums.rc declares field(s) no registered option claims: " + orphans +
                                 ". rc.py is generated from the option descriptors; regenerate it "
                                 "(build the PyEinsums target) or remove the stale field.");
    }
}

// Translate the user's einsums.rc settings into the argv vector that
// einsums::initialize expects.
//
// There is no table here on purpose. The registry already knows every option's
// name, kind, and value type, so this walks it and asks rc for the matching
// attribute; a new descriptor reaches Python without anyone editing this file,
// and a spelling cannot drift because there is only one of it.
std::vector<std::string> argv_from_rc(py::module_ const &rc) {
    std::vector<std::string> argv;
    argv.emplace_back("einsums-python");

    std::set<std::string> claimed;

    for (auto const &opt : einsums::cl::registered_options()) {
        std::string const attr = rc_attribute_name(opt.name);
        claimed.insert(attr);

        if (!py::hasattr(rc, attr.c_str())) {
            continue;
        }
        py::object const v = rc.attr(attr.c_str());
        // ``None`` is "leave the runtime default alone", which is what an
        // absent flag means, so nothing is emitted.
        if (v.is_none()) {
            continue;
        }

        // A flag carries its meaning in its presence, so ``False`` cannot be
        // expressed by leaving it out. It asks for the negation instead,
        // spelled by the runtime's own rule so the two cannot drift apart.
        if (opt.kind == einsums::cl::OptionKind::Flag) {
            argv.emplace_back("--" + (v.cast<bool>() ? opt.name : opt.negated_name));
            continue;
        }

        std::string rendered;
        switch (opt.type) {
        case einsums::cl::OptionType::String:
            rendered = v.cast<std::string>();
            break;
        case einsums::cl::OptionType::Int:
            rendered = std::to_string(rc_int_value(v));
            break;
        case einsums::cl::OptionType::Double:
            rendered = std::to_string(v.cast<double>());
            break;
        case einsums::cl::OptionType::Bool:
            rendered = v.cast<bool>() ? "true" : "false";
            break;
        }
        // The ``--name=value`` form keeps a value with spaces in it one argv
        // element, which the two-token form would not.
        argv.emplace_back("--" + opt.name + "=" + rendered);
    }

    reject_unknown_rc_fields(rc, claimed);

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

    // The registry as data, so a test can hold einsums/rc.py up against the
    // descriptors it was generated from. rc.py is generated from the headers
    // by a static parse and this comes from the running registry, so comparing
    // the two catches a stale generated file as well as a drifted one.
    m.def(
        "_registered_options",
        []() {
            py::list out;
            for (auto const &opt : einsums::cl::registered_options()) {
                py::dict d;
                d["name"]             = opt.name;
                d["key"]              = opt.key;
                d["attribute"]        = rc_attribute_name(opt.name);
                d["negated_name"]     = opt.negated_name;
                d["help"]             = opt.help;
                d["category"]         = opt.category;
                d["value_name"]       = opt.value_name;
                d["kind"]             = opt.kind == einsums::cl::OptionKind::Flag ? "flag" : "value";
                d["computed_default"] = opt.computed_default;
                switch (opt.type) {
                case einsums::cl::OptionType::Bool:
                    d["type"]    = "bool";
                    d["default"] = opt.default_bool;
                    break;
                case einsums::cl::OptionType::Int:
                    d["type"]    = "int";
                    d["default"] = opt.default_int;
                    break;
                case einsums::cl::OptionType::Double:
                    d["type"]    = "float";
                    d["default"] = opt.default_double;
                    break;
                case einsums::cl::OptionType::String:
                    d["type"]    = "str";
                    d["default"] = opt.default_string;
                    break;
                }
                out.append(std::move(d));
            }
            return out;
        },
        "Every option a descriptor registered, as dicts: name, key, the einsums.rc\n"
        "attribute it maps to, kind, value type, and declared default. The generated\n"
        "``--no-`` twin of a flag is named by ``negated_name`` rather than listed.");

    m.def(
        "_argv_from_rc", []() { return argv_from_rc(py::module_::import("einsums.rc")); },
        "The argv einsums.rc's current settings would start the runtime with.\n"
        "Exposed so a test can check that every field reaches a real flag; calling it\n"
        "has no effect on a runtime that is already up.");

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
