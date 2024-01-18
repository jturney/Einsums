//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace einsums::detail {

template <typename Action, typename Enable = void>
struct is_action : std::false_type {};

template <typename T>
inline constexpr bool is_action_v = false;

template <typename Action>
struct is_bound_action : std::false_type {};

template <typename T>
inline constexpr bool is_bound_action_v = false;

} // namespace einsums::detail