//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <type_traits>

namespace einsums {

// From: https://stackoverflow.com/questions/29671643/checking-type-of-parameter-pack-using-enable-if
namespace detail {

template <bool...>
struct bool_pack;

template <bool... bs>
using AllTrue = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

} // namespace detail

template <class R, class... Ts>
using AreAllConvertible = detail::AllTrue<std::is_convertible_v<Ts, R>...>;

template <typename R, class... Ts>
concept AllConvertible = detail::AllTrue<std::is_convertible_v<Ts, R>...>::value;

} // namespace einsums