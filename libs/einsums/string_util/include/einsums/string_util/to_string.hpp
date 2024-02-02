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

namespace einsums::detail {

template <typename T, typename Enable = void>
struct to_string_impl {
    static auto call(T const &value) -> std::string { return fmt::format("{}", value); }
};

template <typename T>
struct to_string_impl<T, std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>> {
    static auto call(T const &value) -> std::string { return std::to_string(value); }
};

template <typename T>
auto to_string(T const &v) -> std::string {
    try {
        return to_string_impl<T>::call(v);
    } catch (...) {
        return throw_bad_lexical_cast<T, std::string>();
    }
}

} // namespace einsums::detail
