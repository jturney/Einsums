//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "IR.hpp"

namespace einsums::pybind {

// Selects which Python-binding library the emitter writes against.
//   * Pybind11 — the default, battle-tested target.
//   * Nanobind — newer, ~3-5× faster compile/binary/runtime. Mostly
//     API-compatible with pybind11 but with renamed enums
//     (``nb::rv_policy`` vs ``py::return_value_policy``), a different
//     module macro (``NB_MODULE``), and a split STL header layout.
//     Nanobind has no direct ``def_buffer`` analog — buffer protocol
//     directives are dropped under the nanobind target.
enum class Target : std::uint8_t { Pybind11, Nanobind };

// Configuration for a single emit pass.
//
// The emitter has two output shapes:
//   * Standalone module — emit ``PYBIND11_MODULE(<module_name>, m) { ... }``
//     (or ``NB_MODULE``) producing a self-contained ``.so`` importable
//     by name. Used by the standalone fixtures and goldens.
//   * Register function — emit ``void <register_function_name>(<ns>::module_ &m)
//     { ... }`` which the aggregator main calls from a single
//     module-macro block. Used by the einsums_add_module autogen path
//     so every module ends up under one ``import einsums``.
struct EmitOptions {
    std::string              module_name;            // module-macro arg (standalone form)
    std::string              register_function_name; // register-function symbol (aggregator form)
    std::string              source_path_for_format; // path used to discover .clang-format
    std::vector<std::string> source_includes;        // headers the bindings refer to (emitted as #include "...")
    Target                   target = Target::Pybind11;
};

// Emit C++ binding code for `module_` and return it as a string. Output
// is post-clang-format using the project's .clang-format if one is found
// via `opts.source_path_for_format`; otherwise LLVM style is used.
std::string emit(Module const &module_, EmitOptions const &opts);

} // namespace einsums::pybind
