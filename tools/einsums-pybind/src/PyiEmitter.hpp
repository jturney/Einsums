//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// .pyi stub emitter. Walks the IR and produces a Python type-stub file
// suitable for pyright / mypy. Consumes the precomputed views populated
// by Visitor + post-IR passes:
//
//   * Visitor                : py_type / return_py_type / default_value_py
//   * PythonOverloads pass   : BoundFunction.python_overloads — dispatcher kind
//                              and overload grouping (no re-derivation)
//   * Properties pass        : BoundClass.properties — @getter/@setter merge
//   * BoundEntityCommon      : submodule routing
//
// MVP scope: emits a single .pyi covering every entity in the module
// (no per-submodule splitting yet — that's an aggregation step in CMake).
// Resolves bound class types in parameter/return positions to their
// Python identifiers; falls back to ``Any`` for unresolved names so
// pyright at least flags them rather than silently mistyping.

#include <string>

#include "IR.hpp"

namespace einsums::pybind {

struct PyiOptions {
    /// Optional comment header inserted at the top (e.g. "// generated
    /// from XYZ.hpp"). Empty by default.
    std::string banner;
};

/// Render the module IR as a Python stub file.
std::string emit_pyi(Module const &module_, PyiOptions const &opts = {});

} // namespace einsums::pybind
