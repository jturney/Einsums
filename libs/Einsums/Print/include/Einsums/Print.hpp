//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/File.hpp>

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <string_view>

namespace einsums {
namespace print {

/// Callback type for forwarding println output to an external sink (e.g. profiler TCP server).
using OutputSinkCallback = std::function<void(std::string const &)>;

/// Register a callback that receives every println/fprintln(stdout/stderr) line (ANSI-stripped).
void EINSUMS_EXPORT set_output_sink(OutputSinkCallback cb);

/// Remove the output sink callback.
void EINSUMS_EXPORT clear_output_sink();

/**
 * Add spaces to the global indentation counter.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT indent();

/**
 * Removes spaces from the global indentation counter.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT deindent();

/**
 * Gets the current indentation level.
 *
 * @return The current indentation level.
 *
 * @versionadded{1.0.0}
 */
auto EINSUMS_EXPORT current_indent_level() -> int;

/**
 * @brief Controls whether a line header is printed for the main thread or not.
 *
 * @param[in] onoff If true, print thread id for main and child threads, otherwise just print for child threads.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT always_print_thread_id(bool onoff);

/**
 * @brief Silences all output.
 *
 * @param[in] onoff If true, output is suppressed, otherwise printing is allowed.
 *
 * @versionadded{1.0.0}
 */
void EINSUMS_EXPORT suppress_output(bool onoff);

/**
 * @brief Controls indentation.
 *
 * @versionadded{1.0.0}
 */
struct Indent {
    Indent() { indent(); }
    ~Indent() { deindent(); }
};

/**
 * @struct ordinal
 *
 * Wraps an integer. When the ordinal is printed, it will add the appropriate ordinal
 * suffix to that integer.
 *
 * @tparam IntType The integer type the ordinal will wrap.
 *
 * @versionadded{1.0.0}
 */
template <std::integral IntType>
struct ordinal {
    constexpr ordinal() = default;

    /**
     * Copy constructor.
     *
     * @param[in] other The ordinal to copy.
     *
     * @versionadded{1.0.0}
     */
    constexpr ordinal(ordinal<IntType> const &other) = default;

    /**
     * Move constructor.
     *
     * @param[inout] other The ordinal to move. On exit, this may be invalidated.
     *
     * @versionadded{1.0.0}
     */
    constexpr ordinal(ordinal<IntType> &&other) noexcept : val_{std::move(other.val_)} {}

    /**
     * Cast constructor.
     *
     * @param[in] value The integer to wrap.
     *
     * @versionadded{1.0.0}
     */
    constexpr ordinal(IntType const &value) : val_{value} {}

    /**
     * Destructor.
     *
     * @versionadded{1.0.0}
     */
    constexpr ~ordinal() = default;

    /**
     * Copy assignment.
     *
     * @param[in] other The ordinal to copy.
     *
     * @versionadded{1.0.0}
     */
    constexpr ordinal<IntType> &operator=(ordinal<IntType> const &other) = default;

    /**
     * Move assignment.
     *
     * @param[inout] other The ordinal to move. On exit, this may be invalidated.
     *
     * @versionadded{1.0.0}
     */
    constexpr ordinal<IntType> &operator=(ordinal<IntType> &&other) noexcept {
        val_ = std::move(other.val_);
        return *this;
    }

    /**
     * Cast to underlying integer type.
     */
    constexpr operator IntType() const { return val_; }

    /**
     * Cast to reference of the underlying integer type.
     *
     * @return A reference to the storage member.
     *
     * @versionadded{1.0.0}
     */
    constexpr operator IntType &() { return val_; }

    /**
     * Cast to arbitrary type.
     *
     * @versionadded{1.0.0}
     */
    template <typename T>
    constexpr operator T() const {
        return T(val_);
    }

#ifndef DOXYGEN
#    define OPERATOR(OP)                                                                                                                   \
        template <std::integral OtherType>                                                                                                 \
        constexpr ordinal<IntType> &operator OP##=(const ordinal<OtherType> &other) {                                                      \
            val_ OP## = other.val_;                                                                                                        \
            return *this;                                                                                                                  \
        }                                                                                                                                  \
        template <std::integral OtherType>                                                                                                 \
        constexpr ordinal<IntType> &operator OP##=(const OtherType &other) {                                                               \
            val_ OP## = other;                                                                                                             \
            return *this;                                                                                                                  \
        }

    OPERATOR(+)
    OPERATOR(-)
    OPERATOR(*)
    OPERATOR(/)
    OPERATOR(%)
    OPERATOR(<<)
    OPERATOR(>>)
    OPERATOR(^)
    OPERATOR(&)
    OPERATOR(|)
#    undef OPERATOR

    template <std::integral OtherType>
    constexpr auto operator<=>(ordinal<OtherType> const &other) const {
        return val_ <=> other.val_;
    }

    template <std::integral OtherType>
    constexpr auto operator<=>(OtherType const &other) const {
        return val_ <=> other;
    }

    template <std::integral OtherType>
    constexpr bool operator==(ordinal<OtherType> const &other) const {
        return val_ == other.val_;
    }

    template <std::integral OtherType>
    constexpr bool operator==(OtherType const &other) const {
        return val_ == other;
    }
#endif

  private:
    /**
     * @var val_
     *
     * The value wrapped by the ordinal.
     *
     * @versionadded{1.0.0}
     */
    IntType val_{0};
};

/**
 * Creates a literal ordinal.
 *
 * @param value The value to wrap.
 *
 * @versionadded{2.0.0}
 */
constexpr ordinal<unsigned long long int> operator""_th(unsigned long long int value) {
    return {value};
}

} // namespace print

/// \cond NOINTERNAL
namespace detail {
void EINSUMS_EXPORT println(std::string const &str);
void EINSUMS_EXPORT fprintln(std::FILE *fp, std::string const &str);
void EINSUMS_EXPORT fprintln(std::ostream &os, std::string const &str);

bool EINSUMS_EXPORT is_terminal(std::ostream const &os);
bool EINSUMS_EXPORT is_terminal(FILE *os);
} // namespace detail
/// \endcond NOINTERNAL

using fmt::bg;
using fmt::color;
using fmt::emphasis;
using fmt::fg;

#ifdef DOXYGEN
/**
 * Prints something to standard output. A new line is emmitted after the print is done.
 *
 * @param[in] args The arguments passed to the printer.
 *
 * @versionadded{1.0.0}
 */
template <typename... Ts>
void println(Ts... args) {
    ;
}

/**
 * Prints something to a file pointer or output stream. A new line is emmitted after the print is done.
 *
 * @param[inout] out The output stream or file.
 * @param[in] args The arguments passed to the printer.
 *
 * @versionadded{1.0.0}
 */
template <typename OutType, typename... Ts>
void fprintln(OutType out, Ts... args) {
    ;
}
#endif

#ifndef DOXYGEN
template <typename... Ts>
void println(std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::println(s);
}

template <typename... Ts>
void println(fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string const s = fmt::format(style, fmt::runtime(format), ts...);
    detail::println(s);
}

inline void println(fmt::text_style const &style, std::string_view const &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::println(s);
}

inline void println() {
    detail::println("\n");
}

template <typename... Ts>
void fprintln(std::FILE *fp, std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::FILE *fp, fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string s;
    if (detail::is_terminal(fp)) {
        s = fmt::format(style, fmt::runtime(format), ts...);
    } else {
        s = fmt::format(fmt::runtime(format), ts...);
    }
    detail::fprintln(fp, s);
}

inline void fprintln(std::FILE *fp, std::string const &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::FILE *fp, fmt::text_style const &style, std::string_view const &format) {
    std::string s;
    if (detail::is_terminal(fp)) {
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
void fprintln(std::ostream &fp, std::string_view const &f, Ts const... ts) {
    std::string const s = fmt::format(fmt::runtime(f), ts...);
    detail::fprintln(fp, s);
}

template <typename... Ts>
void fprintln(std::ostream &fp, fmt::text_style const &style, std::string_view const &format, Ts const... ts) {
    std::string const s = fmt::format(style, fmt::runtime(format), ts...);
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp, std::string const &format) {
    detail::fprintln(fp, format);
}

inline void fprintln(std::ostream &fp, fmt::text_style const &style, std::string_view const &format) {
    std::string const s = fmt::format(style, fmt::runtime(format));
    detail::fprintln(fp, s);
}

inline void fprintln(std::ostream &fp) {
    detail::fprintln(fp, "\n");
}
#endif

/**
 * Calls println to generate an error message, then aborts.
 *
 * @versionadded{1.0.0}
 */

template <typename... Ts>
void println_abort(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    println(bg(color::red) | fg(color::white), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace().print();
#endif

    std::abort();
}

/**
 * Calls println to generate a warning message.
 *
 * @versionadded{1.0.0}
 */
template <typename... Ts>
void println_warn(std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    println(bg(color::yellow) | fg(color::black), message, ts...);

#if defined(EINSUMS_HAVE_CPPTRACE)
    cpptrace::generate_trace(0, 3).print();
#endif
}

/**
 * Calls fprintln to generate an error message, then aborts.
 *
 * @versionadded{1.0.0}
 */
template <typename... Ts>
void fprintln_abort(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(fp, message, ts...);

    std::abort();
}

/**
 * Calls fprintln to generate a warning message.
 *
 * @versionadded{1.0.0}
 */
template <typename... Ts>
void fprintln_warn(std::FILE *fp, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(fp, message, ts...);
}

#ifndef DOXYGEN
template <typename... Ts>
void fprintln_abort(std::ostream &os, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("ERROR: ") + format.data();
    fprintln(os, bg(color::red) | fg(color::white), message, ts...);

    std::abort();
}

template <typename... Ts>
void fprintln_warn(std::ostream &os, std::string_view const &format, Ts const... ts) {
    std::string message = std::string("WARNING: ") + format.data();
    fprintln(os, bg(color::yellow) | fg(color::black), message, ts...);
}
#endif

} // namespace einsums

#if !defined(DOXYGEN)

template <std::integral IntType>
struct fmt::formatter<einsums::print::ordinal<IntType>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return _fmt.parse(ctx);
    }

    template <typename FormatContext>
    auto format(einsums::print::ordinal<IntType> const &value, FormatContext &ctx) const {
        ctx.advance_to(_fmt.format((IntType)value, ctx));
        auto suffix = get_suffix(value);
        return fmt::format_to(ctx.out(), "{}", suffix);
    }

  protected:
    constexpr inline char const *get_suffix(IntType value) const {
        IntType const hundreds = (value % 100 + 100) % 100;

        if (hundreds > 20 || hundreds < 10) {
            switch (hundreds % 10) {
            case 1:
                return "st";
            case 2:
                return "nd";
            case 3:
                return "rd";
            default:
                return "th";
            }
        } else {
            return "th";
        }
    }

  private:
    fmt::formatter<IntType> _fmt;
};

#endif