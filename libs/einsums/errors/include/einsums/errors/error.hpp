// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <fmt/format.h>

#include <string>
#include <system_error>

namespace einsums {

enum class error : std::uint16_t {
    /// The operation was successful
    success = 0,
    /// The operation failed but not in an expected way
    no_success,
    unknown_error,
    bad_parameter,

    last_error,

    system_error_flag = 0x4000L,
    error_upper_bound = 0x7fffL
};

namespace detail {
constexpr const char *const error_names[] = {"success", "no_success", "unknown_error", "bad_parameter"};

inline auto error_code_has_system_error(int e) -> bool {
    return e & static_cast<int>(error::system_error_flag);
}

} // namespace detail
} // namespace einsums

namespace fmt {

template <>
struct formatter<einsums::error> : formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::error e, FormatContext &ctx) const {
        int e_int = static_cast<int>(e);
        if (e_int >= static_cast<int>(einsums::error::success) && e_int < static_cast<int>(einsums::error::last_error)) {
            return formatter<std::string>::format(einsums::detail::error_names[e_int], ctx);
        }
        return formatter<std::string>::format(fmt::format("invalid einsums::error ({})", e_int), ctx);
    }
};

} // namespace fmt

namespace std {

template <>
struct is_error_code_enum<einsums::error> {
    static const bool value = true;
};

template <>
struct is_error_condition_enum<einsums::error> {
    static const bool value = true;
};

} // namespace std