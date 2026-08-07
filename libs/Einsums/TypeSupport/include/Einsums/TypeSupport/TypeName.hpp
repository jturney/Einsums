//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TypeSupport/StringLiteral.hpp>

#include <string_view>
#include <utility>

#if __has_include(<source_location>)
#    include <source_location>
#endif

EINSUMS_NAMESPACE_BEGIN()

namespace detail {
template <typename T>
consteval auto get_type_name_string_view() {
#if __cpp_lib_source_location >= 201907L
    auto const func_name = std::string_view{std::source_location::current().function_name()};
#elif defined(_MSC_VER)
    auto const func_name = std::string_view {
        __FUNCSIG__
    }
#else
    auto const func_name = std::string_view{__PRETTY_FUNCTION__};
#endif

#if defined(__clang__) || defined(__GNUC__)
    auto const split = func_name.substr(0, func_name.size() - 1);
    return split.substr(split.find("T = ") + 4);
#elif defined(_MSC_VER)
    auto split = func_name.substr(0, func_name.size() - 7);
    split      = split.substr(split.find("get_type_name_str_view<") + 23);
    auto pos   = split.find(" ");
    if (pos != std::string_view::npos) {
        return split.substr(pos + 1);
    }
    return split;
#else
    static_assert(false, "You are using an unsupported compiler. Please use GCC, Clang "
                         "or MSVC or explicity tag your structs using 'Tag' or 'Name'.");
#endif
}

} // namespace detail

/**
 * @brief Turns a type into a string representing its name.
 *
 * @return The string representation of the passed type.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
consteval auto type_name() {
    // static_assert(detail::get_type_name_string_view<int>() == "int", "Expected 'int', got something else.");
    constexpr auto name       = detail::get_type_name_string_view<T>();
    auto const     to_str_lit = [&]<auto... Ns>(std::index_sequence<Ns...>) { return StringLiteral<sizeof...(Ns) + 1>{name[Ns]...}; };
    return to_str_lit(std::make_index_sequence<name.size()>{});
}

EINSUMS_NAMESPACE_END()