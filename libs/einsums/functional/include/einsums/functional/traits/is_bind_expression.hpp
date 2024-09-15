//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <functional>

namespace einsums::detail {

template <typename T>
struct is_bind_expression : std::is_bind_expression<T> {};

template <typename T>
struct is_bind_expression<T const> : std::is_bind_expression<T> {};

template <typename T>
inline constexpr bool is_bind_expression_v = is_bind_expression<T>::value;

} // namespace einsums::detail