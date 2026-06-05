//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/StringUtil/Trim.hpp>

#include <fmt/format.h>

#include <array>
#include <concepts>
#include <sstream>
#include <type_traits>

namespace einsums {

/**
 * @brief Sentinel value for ``Tensor::Rank`` on tensor types whose rank is
 *        not known at compile time (e.g. ``GeneralRuntimeTensor``).
 *
 * Concepts that compare ``Rank`` across tensor types (``IsSameRankV``,
 * ``SameRank``, ``SameUnderlyingAndRank``) treat any operand whose
 * ``Rank == dynamic_rank`` as a wildcard — no compile-time mismatch can
 * be inferred against it. Runtime-rank checks remain the caller's
 * responsibility.
 */
inline constexpr int dynamic_rank = -1;

/**
 * @def DEFINE_STRUCT
 *
 * @brief Convenience macro for creating a type derived from a std::array.
 *
 * @versionadded{1.0.0}
 */
#define DEFINE_STRUCT(Name, UnderlyingType)                                                                                                \
    template <std::size_t Rank>                                                                                                            \
    struct Name : std::array<std::int64_t, Rank> {                                                                                         \
        template <typename... Args>                                                                                                        \
        constexpr explicit Name(Args... args) : std::array<std::int64_t, Rank>{static_cast<std::int64_t>(args)...} {                       \
        }                                                                                                                                  \
    };                                                                                                                                     \
    template <typename... Args>                                                                                                            \
    Name(Args... args)->Name<sizeof...(Args)>

/**
 * @struct Dim
 *
 * @brief Holds a list of dimensions in an array.
 *
 * @versionadded{1.0.0}
 */
template <std ::size_t Rank>
struct Dim : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Dim(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    /**
     * Construct a dimension array and fill it with values.
     *
     * @param[in] start An iterator to the start of the data.
     * @param[in] end An iterator to the end of the data.
     *
     * @versionadded{2.0.0}
     */
    template <std::input_iterator Iterator>
    constexpr Dim(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }

        while (this_it != this->end()) {
            *this_it = std::int64_t{0};
            this_it++;
        }
    }
};

/**
 * @struct Stride
 *
 * @brief Holds a list of strides in an array.
 *
 * @versionadded{1.0.0}
 */
template <std ::size_t Rank>
struct Stride : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Stride(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    /**
     * Construct a stride array from an iterator. If more elements are provided than the rank, then the extra elements are ignored.
     * If fewer elements are provided than the rank, then the array will be backfilled with zeros.
     *
     * @param[in] start The starting iterator.
     * @param[in] end The ending iterator.
     *
     * @versionadded{2.0.0}
     */
    template <std::input_iterator Iterator>
    constexpr Stride(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }

        while (this_it != this->end()) {
            *this_it = std::int64_t{0};
            this_it++;
        }
    }
};

/**
 * @struct Offset
 *
 * @brief Holds a list of offsets in an array.
 *
 * @versionadded{1.0.0}
 */
template <std ::size_t Rank>
struct Offset : std ::array<std ::int64_t, Rank> {
    using std::array<std::int64_t, Rank>::array;
    /**
     * @brief Aggregate constructor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Args>
        requires(std::is_integral_v<std::remove_cvref_t<Args>> && ...)
    constexpr explicit Offset(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}

    /**
     * Construct an offset array from an iterator. If more elements are provided than the rank, then the extra elements are ignored.
     * If fewer elements are provided than the rank, then the array will be backfilled with zeros.
     *
     * @param[in] start The starting iterator.
     * @param[in] end The ending iterator.
     *
     * @versionadded{2.0.0}
     */
    template <std::input_iterator Iterator>
    constexpr Offset(Iterator start, Iterator end) {
        auto this_it  = this->begin();
        auto other_it = start;

        while (this_it != this->end() && other_it != end) {
            *this_it = static_cast<std::int64_t>(*other_it);
            this_it++;
            other_it++;
        }

        while (this_it != this->end()) {
            *this_it = std::int64_t{0};
            this_it++;
        }
    }
};

/**
 * @struct Count
 *
 * @brief Holds a list of counts in an array.
 *
 * @versionadded{1.0.0}
 */
template <std ::size_t Rank>
struct Count : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Args>
    constexpr explicit Count(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}
};

/**
 * @struct Chunk
 *
 * @brief Holds a list of chunks in an array.
 *
 * @versionadded{1.0.0}
 */
template <std ::size_t Rank>
struct Chunk : std ::array<std ::int64_t, Rank> {
    /**
     * @brief Aggregate constructor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Args>
    constexpr explicit Chunk(Args... args) : std ::array<std ::int64_t, Rank>{static_cast<std ::int64_t>(args)...} {}
};

template <typename... Args>
Dim(Args... args) -> Dim<sizeof...(Args)>;

template <typename... Args>
Stride(Args... args) -> Stride<sizeof...(Args)>;

template <typename... Args>
Offset(Args... args) -> Offset<sizeof...(Args)>;

template <typename... Args>
Count(Args... args) -> Count<sizeof...(Args)>;

template <typename... Args>
Chunk(Args... args) -> Chunk<sizeof...(Args)>;

/**
 * @struct Range
 *
 * Holds two values: a starting value and an ending value.
 *
 * @versionadded{1.0.0}
 */
struct Range : std::array<std::int64_t, 2> {
    constexpr Range() = default;

    /**
     * Initialize a range.
     *
     * @versionadded{1.0.0}
     */
    template <std::integral First, std::integral Second>
    constexpr explicit Range(First first, Second second)
        : std::array<std::int64_t, 2>{static_cast<std::int64_t>(first), static_cast<std::int64_t>(second)} {}

    /**
     * Check if the range can be treated as a single value if its entries are the same.
     * If it is removable and the entries are the same, then the rank of the tensor view created with this range will
     * have a lower rank than the parent tensor. If it is not removable, or the entries are different, then the rank
     * of the view will be the same as the rank of the parent.
     *
     * @versionadded{2.0.0}
     */
    [[nodiscard]] bool is_removable() const noexcept { return _is_removable; }

  protected:
    /**
     * Holds whether the range is removable.
     *
     * @versionadded{2.0.0}
     */
    bool _is_removable{false};
};

/**
 * @struct RemovableRange
 *
 * Holds two values: a starting value and an ending value. It will be treated as a single value if the start and end are the same.
 * The usefulness may not be immediately apparent. This class is mostly used in the Python compatibility layer, but as an example,
 * look at this code segment.
 *
 * @code
 * RuntimeTensor<double> A{"A", 3, 3, 3}; // A is a rank-3 tensor.
 *
 * auto A_view_1 = A(Range{0, 1}, Range{0, 1}, Range{0, 0});
 * auto A_view_2 = A(Range{0, 1}, Range{0, 1}, 0);
 * auto A_view_3 = A(Range{0, 1}, Range{0, 1}, RemovableRange{0, 0});
 *
 * std::vector<Range> indices{Range{0, 1}, Range{0, 1}, RemovableRange{0, 0}};
 *
 * auto A_view_4 = A(indices);
 *
 * EINSUMS_ASSERT(A_view_1.rank() == 3);
 * EINSUMS_ASSERT(A_view_2.rank() == 2);
 * EINSUMS_ASSERT(A_view_3.rank() == 2);
 * EINSUMS_ASSERT(A_view_4.rank() == 2);
 * @endcode
 *
 * In the first three examples, the RemovableRange is not really needed. It can be replaced with a single value index. However, in the
 * fourth example, we can't use a single value because the vector can only hold ranges. Since RemovableRange extends Range, we can use it in
 * this vector where it will be treated as if it were a single index. If the elements in the range are not the same, such as
 * <tt>RemovableRange{0, 1}</tt>, then it behaves exactly the same as a regular Range. The removable part is a hint to the functions that
 * look for them that they can be removed if needed.
 *
 * @versionadded{2.0.0}
 */
struct RemovableRange : Range {
    /**
     * Initialize a range.
     *
     * @versionadded{2.0.0}
     */
    template <std::integral First, std::integral Second>
    constexpr explicit RemovableRange(First first, Second second) : Range{first, second} {
        this->_is_removable = true;
    }
};

/**
 * Type that indicates that all elements along a dimension should be included in a view.
 *
 * @versionadded{1.0.0}
 */
struct AllT {};

/**
 * Implementation of AllT .
 *
 * @versionadded{1.0.0}
 */
static struct AllT All; // NOLINT

#undef DEFINE_STRUCT

} // namespace einsums

namespace fmt {
template <size_t Rank>
struct formatter<einsums::Dim<Rank>> {
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
    auto format(einsums::Dim<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;

        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Dim{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct formatter<einsums::Stride<Rank>> {
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
    auto format(einsums::Stride<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Stride{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct formatter<einsums::Count<Rank>> {
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
    auto format(einsums::Count<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Count{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <size_t Rank>
struct formatter<einsums::Offset<Rank>> {
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
    auto format(einsums::Offset<Rank> const &dim, format_context &ctx) const -> format_context::iterator {
        std::ostringstream oss;
        for (size_t i = 0; i < Rank; i++) {
            oss << dim[i] << " ";
        }
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Offset{{{}}}", einsums::string_util::rtrim_copy(oss.str()));
    }
};

template <>
struct formatter<einsums::Range> {
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
    auto format(einsums::Range const &dim, format_context &ctx) const -> format_context::iterator {
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "Range{{{}, {}}}", dim[0], dim[1]);
    }
};
} // namespace fmt
