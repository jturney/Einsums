//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdio>
#include <ostream>
#include <type_traits>

namespace einsums {

template <typename T>
concept IsFilePointer = std::is_same_v<T, FILE *>;

template <typename T>
concept IsOstream = std::is_base_of_v<std::ostream, T>;

template <typename T>
concept FileOrOstream = IsFilePointer<T> || IsOstream<T>;

} // namespace einsums