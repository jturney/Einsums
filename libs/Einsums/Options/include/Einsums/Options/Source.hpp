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

/*
 * The vocabulary every other header in this module builds on: where a value
 * came from, how an option may appear, and the numeric bounds a value must
 * satisfy. Nothing here allocates, formats, or parses, so a translation unit
 * that only reads options pays for standard library headers and nothing else.
 */

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
 * The enumerators are ordered by increasing precedence: a later source is
 * allowed to overwrite an earlier one, and the parser applies them in exactly
 * this order. @ref Source::None means the option was never touched at all.
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

/// Constructor tag marking an option positional rather than named.
struct Positional {};

// -------------------------- Range ----------------------------------------- //

/**
 * @brief Inclusive bounds applied to a numeric option after parsing.
 *
 * Integral and floating point bounds are stored separately so that neither
 * kind of check has to round-trip through the other's representation: an
 * int64_t bound near the limits of its range stays exact, and a fractional
 * bound is not truncated.
 *
 * Bounds live here rather than alongside the option types because a
 * @ref ConfigOption carries them too, and the descriptor header is meant to
 * stay free of the declaration machinery.
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
