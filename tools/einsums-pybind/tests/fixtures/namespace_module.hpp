//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase C.13 fixture: EINSUMS_PYBIND_MODULE on a namespace declaration is
// inherited by every annotated entity inside the namespace. Per-entity
// overrides win.

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

namespace EINSUMS_PYBIND_MODULE("graph") graph {

/// Inherits the enclosing namespace's module — binds into ``graph``.
EINSUMS_PYBIND_EXPOSE int inherited(int x);

/// Per-entity override binds into ``graph.ops`` instead.
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph.ops") int overridden(int x);

} // namespace EINSUMS_PYBIND_MODULE("graph")graph

/// Outside the annotated namespace — binds at the top level.
EINSUMS_PYBIND_EXPOSE int top_level(int x);

} // namespace einsums::fixture
