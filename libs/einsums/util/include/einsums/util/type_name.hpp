//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <array>
#include <cstdint>

namespace einsums::util {

//
// Taken from https://stackoverflow.com/posts/59522794/revisions
//
namespace detail {
template <typename T>
constexpr auto raw_type_name() -> const auto & {
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

struct raw_type_name_format {
    std::size_t leading_junk = 0, trailing_junk = 0;
};

// Returns `false` on failure.
inline constexpr auto get_raw_type_name_format(raw_type_name_format *format) -> bool {
    const auto &str = raw_type_name<int>();
    for (std::size_t i = 0;; i++) {
        if (str[i] == 'i' && str[i + 1] == 'n' && str[i + 2] == 't') {
            if (format) {
                format->leading_junk  = i;
                format->trailing_junk = sizeof(str) - i - 3 - 1; // `3` is the length of "int", `1` is the space for the null terminator.
            }
            return true;
        }
    }
    return false;
}

inline static constexpr raw_type_name_format format = [] {
    static_assert(get_raw_type_name_format(nullptr), "Unable to figure out how to generate type names on this compiler.");
    raw_type_name_format format;
    get_raw_type_name_format(&format);
    return format;
}();

// Returns the type name in a `std::array<char, N>` (null-terminated).
template <typename T>
[[nodiscard]] constexpr auto cexpr_type_name() {
    constexpr std::size_t len = sizeof(raw_type_name<T>()) - format.leading_junk - format.trailing_junk;
    std::array<char, len> name{};
    for (std::size_t i = 0; i < len - 1; i++)
        name[i] = raw_type_name<T>()[i + format.leading_junk];
    return name;
}

} // namespace detail

template <typename T>
[[nodiscard]] auto type_name() -> const char * {
    static constexpr auto name = detail::cexpr_type_name<T>();
    return name.data();
}
template <typename T>
[[nodiscard]] auto type_name(const T &) -> const char * {
    return type_name<T>();
}

} // namespace einsums::util