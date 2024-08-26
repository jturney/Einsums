// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#if defined(EINSUMS_HAVE_CPPTRACE)
#    include <cpptrace/cpptrace.hpp>
#endif

namespace einsums {

namespace print {

/// Add spaces to the global indentation counter.
void EINSUMS_EXPORT indent();

/// Removes spaces from the global indentation counter.
void EINSUMS_EXPORT deindent();

/// Returns the current indentation level.
auto EINSUMS_EXPORT current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param onoff If true, print thread id for main and child threads, otherwise just print for child threads.
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
void EINSUMS_EXPORT println(const std::string &oss);
void EINSUMS_EXPORT fprintln(std::FILE *fp, const std::string &oss);
void EINSUMS_EXPORT fprintln(std::ostream &os, const std::string &oss);
} // namespace detail

using fmt::bg;
using fmt::color;
using fmt::emphasis;
using fmt::fg;

template <typename... Ts>
void println(const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::println(s);
}

template <typename... Ts>
void println(const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, fmt::runtime(format), ts...);
    detail::println(s);
}

inline void println(const fmt::text_style &style, const std::string_view &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::println(s);
}

inline void println() {
    detail::println("\n");
}

template <typename... Ts>
void fprintln(std::FILE *fp, const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::FILE *fp, const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, format, ts...);
    } else {
        s = fmt::format(format, ts...);
    }
    detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp, const std::string &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::FILE *fp, const fmt::text_style &style, const std::string_view &format) {
    std::string s;
    if (fp == stdout || fp == stderr) {
        s = fmt::format(style, fmt::runtime(format));
    } else {
        s = format;
    }
    detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp) {
    detail::fprintln(fp, "\n");
}

template <typename... Ts>
void fprintln(std::ostream &fp, const std::string_view &f, const Ts... ts) {
    std::string s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::ostream &fp, const fmt::text_style &style, const std::string_view &format, const Ts... ts) {
    std::string s = fmt::format(style, format, ts...);
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp, const std::string &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::ostream &fp, const fmt::text_style &style, const std::string_view &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp) {
    ::einsums::detail::fprintln(fp, "\n");
}

template <typename... Ts>
inline void println_abort(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace().print();
#endif

    std::abort();
}

template <typename... Ts>
inline void println_warn(const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace(0, 3).print();
#endif
}

template <typename... Ts>
inline void fprintln_abort(std::FILE *fp, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(fp, message, ts...);

    std::abort();
}

template <typename... Ts>
inline void fprintln_warn(std::FILE *fp, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(fp, message, ts...);
}

template <typename... Ts>
inline void fprintln_abort(std::ostream &os, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(os, bg(fmt::color::red) | fg(fmt::color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
inline void fprintln_warn(std::ostream &os, const std::string_view &format, const Ts... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(os, bg(fmt::color::yellow) | fg(fmt::color::black), message, ts...);
}

} // namespace einsums