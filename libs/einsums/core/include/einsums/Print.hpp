//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/config/ExportDefinitions.hpp"

#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/format.h"
// #include "fmt/ranges.h"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace print {

/** Adds spaces to the global indentation counter. */
void EINSUMS_EXPORT indent();
/** Removes spaces from the global indentation counter. */
void EINSUMS_EXPORT deindent();

/** Returns the current indentation level. */
auto EINSUMS_EXPORT current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param onoff If true, print thread id for main and child threads, otherwise just print for child threads.
 *
 *
 */
void EINSUMS_EXPORT always_print_thread_id(bool onoff);

/**
 * @brief Silences all output.
 *
 * @param onoff If true, output is suppressed, otherwise printing is allowed.
 */
void EINSUMS_EXPORT suppress_output(bool onoff);

struct Indent {
    Indent() { indent(); }
    ~Indent() { deindent(); }
};

} // namespace print

namespace detail {
void EINSUMS_EXPORT println(std::string const &oss);
}

//
// Taken from https://stackoverflow.com/posts/59522794/revisions
//
namespace detail {
template <typename T>
constexpr auto RawTypeName() -> auto const & {
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

struct RawTypeNameFormat {
    std::size_t leading_junk = 0, trailing_junk = 0;
};

// Returns `false` on failure.
inline constexpr auto GetRawTypeNameFormat(RawTypeNameFormat *format) -> bool {
    auto const &str = RawTypeName<int>();
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

inline static constexpr RawTypeNameFormat format = [] {
    static_assert(GetRawTypeNameFormat(nullptr), "Unable to figure out how to generate type names on this compiler.");
    RawTypeNameFormat format;
    GetRawTypeNameFormat(&format);
    return format;
}();
} // namespace detail

// Returns the type name in a `std::array<char, N>` (null-terminated).
template <typename T>
[[nodiscard]] constexpr auto CexprTypeName() {
    constexpr std::size_t len = sizeof(detail::RawTypeName<T>()) - detail::format.leading_junk - detail::format.trailing_junk;
    std::array<char, len> name{};
    for (std::size_t i = 0; i < len - 1; i++)
        name[i] = detail::RawTypeName<T>()[i + detail::format.leading_junk];
    return name;
}

template <typename T>
[[nodiscard]] auto type_name() -> char const * {
    static constexpr auto name = CexprTypeName<T>();
    return name.data();
}
template <typename T>
[[nodiscard]] auto type_name(T const &) -> char const * {
    return type_name<T>();
}
namespace detail {

template <typename Tuple, std::size_t N>
struct TuplePrinter {
    static void print(std::ostream &os, Tuple const &t) {
        TuplePrinter<Tuple, N - 1>::print(os, t);
        os << ", (" << type_name<decltype(std::get<N - 1>(t))>() << ")" << std::get<N - 1>(t);
    }
};

template <typename Tuple>
struct TuplePrinter<Tuple, 1> {
    static void print(std::ostream &os, Tuple const &t) { os << "(" << type_name<decltype(std::get<0>(t))>() << ")" << std::get<0>(t); }
};

template <typename Tuple, std::size_t N>
struct TuplePrinterNoType {
    static void print(std::ostream &os, Tuple const &t) {
        TuplePrinterNoType<Tuple, N - 1>::print(os, t);
        os << ", " << std::get<N - 1>(t);
    }
};

template <typename Tuple>
struct TuplePrinterNoType<Tuple, 1> {
    static void print(std::ostream &os, Tuple const &t) { os << std::get<0>(t); }
};

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple(std::tuple<Args...> const &) -> std::string {
    return {"()"};
}

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
auto print_tuple_no_type(std::tuple<Args...> const &) -> std::string {
    return {"()"};
}

} // namespace detail

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple(std::tuple<Args...> const &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinter<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
auto print_tuple_no_type(std::tuple<Args...> const &t) -> std::string {
    std::ostringstream out;
    out << "(";
    detail::TuplePrinterNoType<decltype(t), sizeof...(Args)>::print(out, t);
    out << ")";
    return out.str();
}

inline auto print_tuple_no_type(std::tuple<> const &) -> std::string {
    std::ostringstream out;
    out << "( )";
    return out.str();
}

using fmt::bg;       // NOLINT
using fmt::color;    // NOLINT
using fmt::emphasis; // NOLINT
using fmt::fg;       // NOLINT

template <typename... Ts>
void println(std::string_view const &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::println(s);
}

template <typename... Ts>
void println(fmt::text_style const &style, std::string_view const &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    detail::println(s);
}

inline void println(std::string const &format) {
    detail::println(format);
}

inline void println(fmt::text_style const &style, std::string_view const &format) {
    std::string const s = fmt::format(style, format);
    detail::println(s);
}

inline void println() {
    detail::println("\n");
}

template <typename... Ts>
inline void println_abort(std::string_view const &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
inline void println_warn(std::string_view const &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}
