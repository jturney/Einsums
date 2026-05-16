//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Post-IR pass that decides how each templated free function's raw
// instantiation list collapses into Python-facing entries. Mirrors the
// grouping logic the pybind11 emitter expects:
//
//   * @instantiate_as overloads sharing a Python name AND argument
//     signature (only the return type varies) collapse into one
//     dispatcher taking a ``dtype="..."`` kwarg.
//   * 2^N instantiations from EINSUMS_PYBIND_INSTANTIATE_BOOLS collapse
//     into one dispatcher with N bool kwargs (kw_only).
//   * Otherwise each instantiation is its own pybind11 overload (or the
//     sole entry, in the non-template case).
//
// Both the pybind11 C++ emitter and the .pyi stub emitter consume the
// resulting `BoundFunction.python_overloads` view so the merge rules
// don't need to be reimplemented across the two backends.

#include <string>
#include <vector>

#include "IR.hpp"

namespace einsums::pybind {

// Compute and assign `f.python_overloads`. Idempotent — calling twice
// on the same function clears the previous result first. Calls to this
// must happen after the Visitor has finished populating instantiations.
void compute_python_overloads(BoundFunction &f);

// Convenience: walk every function in `module_` and run
// compute_python_overloads on it.
void compute_python_overloads(Module &module_);

// ── Helpers exposed for reuse by the emitter ──────────────────────────

// Split a per-instantiation comma-joined string into individual values,
// respecting `<>` nesting. e.g. ``"float, 2"`` -> ``{"float", "2"}``.
std::vector<std::string> split_instantiation_args(std::string const &combo);

// Map a C++ scalar type to its accepted dtype-string aliases. Returns
// empty when the type isn't a recognized dtype (the dispatcher path
// only triggers for known dtypes).
std::vector<std::string> dtype_aliases_for(std::string const &cpp_type);

// Pick the default dtype string for a dispatcher group. Numpy
// convention favors ``float64`` (``double``) when present; otherwise
// the first instance's first alias.
std::string pick_default_dtype(std::vector<std::string> const &dtype_values_in_order);

} // namespace einsums::pybind
