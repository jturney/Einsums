//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/string_utils/trim.hpp>

#include <fmt/format.h>

#include <array>
#include <sstream>

namespace einsums {

#define DEFINE_STRUCT(Name, UnderlyingType)                                                                                                \
    template <std::size_t Rank>                                                                                                            \
    struct Name : std::array<std::int64_t, Rank> {                                                                                         \
        template <typename... Args>                                                                                                        \
        constexpr explicit Name(Args... args) : std::array<std::int64_t, Rank>{static_cast<std::int64_t>(args)...} {                       \
        }                                                                                                                                  \
    };                                                                                                                                     \
    template <typename... Args>                                                                                                            \
    Name(Args... args)->Name<sizeof...(Args)> /**/

DEFINE_STRUCT(dim, std::int64_t);
DEFINE_STRUCT(stride, std::size_t);
DEFINE_STRUCT(offset, std::size_t);
DEFINE_STRUCT(count, std::size_t);
DEFINE_STRUCT(chunk, std::int64_t);

struct range : std::array<std::int64_t, 2> {
    template <typename... Args>
    constexpr explicit range(Args... args) : std::array<std::int64_t, 2>{static_cast<std::int64_t>(args)...} {}
};

struct all_t {};
static struct all_t all; // NOLINT

#undef DEFINE_STRUCT

} // namespace einsums

template <size_t Rank>
struct fmt::formatter<einsums::dim<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const einsums::dim<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;

        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Dim{{{}}}", einsums::string_utils::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::stride<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const einsums::stride<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Stride{{{}}}", einsums::string_utils::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::count<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const einsums::count<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Count{{{}}}", einsums::string_utils::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct fmt::formatter<einsums::offset<Rank>> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const einsums::offset<Rank> &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Offset{{{}}}", einsums::string_utils::rtrim_copy(oss.str()));
    }
};

template <>
struct fmt::formatter<einsums::range> {
    constexpr auto parse(format_parse_context &ctx) -> format_parse_context::iterator {
        // Parse the presentation format and store it in the formatter:

        auto it = ctx.begin(), end = ctx.end();

        // Check if reached the end of the range:
        if (it != end && *it != '}')
            report_error("invalid format");

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const einsums::range &dim, format_context &ctx) const -> format_context::iterator {
        // ctx.out() is an output iterator to write to.
        return fmt::format_to(ctx.out(), "Range{{{}, {}}}", dim[0], dim[1]);
    }
};
