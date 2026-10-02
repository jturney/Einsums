//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <string_view>

// The shared vocabulary: value sources, occurrence rules and numeric bounds. Standard headers only.

EINSUMS_NAMESPACE_BEGIN(cl)

namespace detail {
/// Dependent false, so a static_assert in an uninstantiated template body does
/// not fire until the template is actually used.
template <typename...>
inline constexpr bool always_false = false;
} // namespace detail

/**
 * @brief Where an option's current value came from.
 *
 * In increasing precedence, the order the parser applies them. @ref Source::None means never set.
 *
 * @versionadded{2.0.0}
 */
enum struct Source : std::uint8_t {
    None,        ///< No value has been assigned yet.
    Default,     ///< The Default(...) supplied by the programmer.
    ConfigFile,  ///< A config file or config map entry.
    Environment, ///< An environment variable.
    CommandLine  ///< An argument on the command line.
};

/// Human-readable name for a value source, for diagnostics and help text.
constexpr std::string_view to_string(Source s) noexcept {
    switch (s) {
    case Source::Default:
        return "default";
    case Source::ConfigFile:
        return "config file";
    case Source::Environment:
        return "environment";
    case Source::CommandLine:
        return "command line";
    default:
        return "unset";
    }
}

struct ParseResult {
    bool ok        = true;
    int  exit_code = 0;
};

/// Whether an option appears in `--help` at all.
enum struct Visibility : std::uint8_t { Normal, Hidden };

/// How many times an option may, or must, be given.
enum struct Occurrence : std::uint8_t { Optional, Required, ZeroOrMore, OneOrMore };

/// Whether an option takes a value, and whether it may go without one.
enum struct ValueExpected : std::uint8_t { ValueDisallowed, ValueOptional, ValueRequired };

/// How an option is spelled on the command line.
enum struct OptionKind : std::uint8_t {
    Value, ///< Takes a value: `--einsums:log:level 3`.
    Flag   ///< Presence carries the meaning; a `--no-` twin is generated for it.
};

/// The value type an option holds. A descriptor fixes it at declaration; the
/// enumeration API hands it to callers who only see an option as data.
enum struct OptionType : std::uint8_t { Bool, Int, Double, String };

/// Constructor tag marking an option positional rather than named.
struct Positional {};

// -------------------------- Range ----------------------------------------- //

/**
 * @brief Inclusive bounds applied to a numeric option after parsing.
 *
 * Integral and floating bounds are kept separately, so neither loses precision.
 */
struct Range {
    long long   int_min  = (std::numeric_limits<long long>::min)();
    long long   int_max  = (std::numeric_limits<long long>::max)();
    long double real_min = -std::numeric_limits<long double>::infinity();
    long double real_max = std::numeric_limits<long double>::infinity();
};

/// Bounds for an integral option, e.g. `RangeBetween(1, 256)`.
template <std::integral T>
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr Range RangeBetween(T min_v, T max_v) {
    return Range{.int_min  = static_cast<long long>(min_v),
                 .int_max  = static_cast<long long>(max_v),
                 .real_min = static_cast<long double>(min_v),
                 .real_max = static_cast<long double>(max_v)};
}

/// Bounds for a floating point option, e.g. `RangeBetween(0.0, 1.0)`.
template <std::floating_point T>
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr Range RangeBetween(T min_v, T max_v) {
    return Range{.int_min  = (std::numeric_limits<long long>::min)(),
                 .int_max  = (std::numeric_limits<long long>::max)(),
                 .real_min = static_cast<long double>(min_v),
                 .real_max = static_cast<long double>(max_v)};
}

EINSUMS_NAMESPACE_END(cl)
