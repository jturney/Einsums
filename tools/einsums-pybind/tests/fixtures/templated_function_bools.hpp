//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase C.8 fixture: leading bool template parameters lifted into Python
// kwargs via EINSUMS_PYBIND_TEMPLATE_KWARGS, with per-dtype slices supplied
// by EINSUMS_PYBIND_INSTANTIATE_BOOLS. Each BOOLS directive expands
// internally to 2^N INSTANTIATE_AS lines covering every false/true combo
// in lexicographic order; same-tail instantiations collapse into one
// m.def with a runtime if-chain dispatcher.

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

template <typename T>
struct Box {
    T value;
};

/// Apply a flagged transform to a value plus a Box payload.
template <bool TransA, bool TransB, typename T>
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_TEMPLATE_KWARGS("trans_a", "trans_b") EINSUMS_PYBIND_INSTANTIATE_BOOLS("apply", float)
    EINSUMS_PYBIND_INSTANTIATE_BOOLS("apply", double) T apply(T const x, Box<T> &b);

/// Single-bool void-returning variant.
template <bool Conjugate, typename T>
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_TEMPLATE_KWARGS("conjugate") EINSUMS_PYBIND_INSTANTIATE_BOOLS("scale_inplace", float)
    EINSUMS_PYBIND_INSTANTIATE_BOOLS("scale_inplace", double) void scale_inplace(Box<T> &b, T const factor);

} // namespace einsums::fixture
