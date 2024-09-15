//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace einsums::detail {
///////////////////////////////////////////////////////////////////////////
template <typename TD>
struct decay_unwrap_impl {
    using type = TD;
};

template <typename X>
struct decay_unwrap_impl<::std::reference_wrapper<X>> {
    using type = X &;
};

template <typename T>
struct decay_unwrap : detail::decay_unwrap_impl<std::decay_t<T>> {};

template <typename T>
using decay_unwrap_t = typename decay_unwrap<T>::type;
} // namespace einsums::detail
