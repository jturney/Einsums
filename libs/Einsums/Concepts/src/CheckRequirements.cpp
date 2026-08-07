//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <list>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

static_assert(Container<std::vector<int>>);
static_assert(Container<std::array<int, 10>>);
static_assert(Container<std::list<int>>);
static_assert(!Container<std::initializer_list<int>>); // initializer_list does not have difference_type, so it is not a container.

EINSUMS_NAMESPACE_END()