//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase-2 fixture: a class with constructors, methods, fields, operator
// overload, getter/setter pair, and one hidden inherited member. Exercises
// scope tracking and the BoundClass / BoundMethod / BoundField paths.

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

class EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RENAME("PyShape") EINSUMS_PYBIND_NOCOPY Shape {
  public:
    /// Default-construct an empty shape.
    EINSUMS_PYBIND_EXPOSE Shape();

    /// Build a shape from a single dimension.
    EINSUMS_PYBIND_EXPOSE explicit Shape(int dim);

    /// Number of dimensions.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_RVP(reference_internal) int rank() const;

    /// Equality across two shapes.
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_OPERATOR("__eq__") bool operator==(Shape const &other) const;

    /// Read-only access to the contained dim, exposed as a Python property.
    EINSUMS_PYBIND_GETTER("dim") int get_dim() const;

    /// Mutator paired with the getter above.
    EINSUMS_PYBIND_SETTER("dim") void set_dim(int value);

    EINSUMS_PYBIND_EXPOSE int public_field;

    EINSUMS_PYBIND_HIDE void internal_helper();
};

} // namespace einsums::fixture
