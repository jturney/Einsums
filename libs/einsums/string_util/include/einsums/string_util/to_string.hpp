//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/string_util/bad_lexical_cast.hpp>

#include <fmt/format.h>

#include <string>
#include <type_traits>

namespace einsums::string_util {

namespace detail {

template <typename T, typename Enable = void>
struct to_string_impl {
    static std::string call(T const &value) { return fmt::format("{}", value); }
};

template <typename T>
struct to_string_impl<T, std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>> {
    static std::string call(T const &value) { return std::to_string(value); }
};

} // namespace detail

template <typename T>
std::string to_string(T const &v) {
    try {
        return detail::to_string_impl<T>::call(v);
    } catch (...) {
        return detail::throw_bad_lexical_cast<T, std::string>();
    }
}

} // namespace einsums::string_util