//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Member-template fixture: INSTANTIATE_MEMBER_AS pins a templated
// method's own template parameter to a concrete type for one Python
// binding. Multiple directives stack — same-arg-signature ones merge
// into a dtype dispatcher exactly like free-function INSTANTIATE_AS.

#pragma once

#include "Einsums/Python/Annotations.hpp"

namespace einsums::fixture {

/// Holder class with a member-template method. Each
/// INSTANTIATE_MEMBER_AS line on the method emits one Python binding
/// with the named template arg pinned to a concrete type.
class EINSUMS_PYBIND_EXPOSE Workspace {
  public:
    EINSUMS_PYBIND_EXPOSE Workspace();

    /// Allocate a slot keyed by an integer id, returning the new
    /// slot's value. Two INSTANTIATE_MEMBER_AS lines share an argument
    /// signature, differing only in the return type — same dtype-
    /// dispatcher rules apply: collapse into one Python def taking
    /// ``dtype="..."``.
    template <typename U>
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS("allocate", U = float)
        EINSUMS_PYBIND_INSTANTIATE_MEMBER_AS("allocate", U = double) U allocate(int slot_id);
};

} // namespace einsums::fixture
