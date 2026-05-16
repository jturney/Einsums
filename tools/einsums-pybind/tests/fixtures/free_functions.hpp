//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase-2 fixture: free functions in a namespace. Exercises BoundFunction
// emission, parameter capture (including a default arg), and the
// release_gil + rvp directive plumbing.

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

/// Sum of two integers.
EINSUMS_PYBIND_EXPOSE int add(int a, int b);

/// Sum with default scale factor.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RVP(move) int scaled_add(int a, int b, int scale = 1);

/// A computation that should release the GIL while running.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RELEASE_GIL double heavy_compute(double seed);

// Unannotated — must NOT appear in the IR.
double should_not_appear();

} // namespace einsums::fixture
