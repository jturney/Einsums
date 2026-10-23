//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(testing)

/// A deterministic test value for logical element @p k of a tensor drawn with @p seed.
///
/// Never zero and never tiny, so a test may divide by it: a real value lies in [0.25, 1.25] in magnitude
/// with either sign, and an integer in [1, 9] with either sign. A complex value has both parts so drawn.
template <typename T>
T layout_value(unsigned seed, std::size_t k) {
    auto const draw = [&](std::uint64_t salt) {
        std::uint64_t h = (static_cast<std::uint64_t>(seed) << 32U) ^ (k * 0x9E3779B97F4A7C15ULL) ^ salt;
        h ^= h >> 33U;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33U;
        return h;
    };
    auto const real = [&](std::uint64_t salt) {
        std::uint64_t const h    = draw(salt);
        double const        sign = (h & 1U) != 0 ? -1.0 : 1.0;
        if constexpr (std::is_integral_v<T>) {
            return sign * static_cast<double>(1U + (h >> 1U) % 9U);
        } else {
            return sign * (0.25 + static_cast<double>((h >> 11U) & 0xFFFFFU) / static_cast<double>(0xFFFFFU));
        }
    };
    if constexpr (IsComplexV<T>) {
        using R = RemoveComplexT<T>;
        return T(static_cast<R>(real(1)), static_cast<R>(real(2)));
    } else {
        return static_cast<T>(real(1));
    }
}

/// One memory layout of a tensor, in a buffer of its own.
///
/// Every layout made by @ref layouts holds the same logical values, so an operation's result can be
/// compared across layouts and against a reference computed from @ref layout_value. Buffer elements
/// the layout does not reach hold @ref sentinel, so a write through a wrong stride shows in
/// @ref gaps_untouched.
template <typename T>
struct Layout {
    std::string              name;
    std::vector<std::size_t> dims, strides;
    std::vector<T>           buffer;

    static T sentinel() { return T(77); }

    /// A tensor implementation over the buffer. Make it after the layout has stopped moving.
    einsums::detail::TensorImpl<T> impl() { return {buffer.data(), dims, strides}; }

    /// The element at multi-index @p index.
    T &at(std::vector<std::size_t> const &index) {
        std::size_t offset = 0;
        for (std::size_t d = 0; d < dims.size(); d++) {
            offset += index[d] * strides[d];
        }
        return buffer[offset];
    }

    /// Whether every buffer element outside the layout still holds the sentinel.
    [[nodiscard]] bool gaps_untouched() const {
        std::size_t const reached   = std::accumulate(dims.begin(), dims.end(), std::size_t{1}, std::multiplies<>());
        auto const        sentinels = static_cast<std::size_t>(std::count(buffer.begin(), buffer.end(), sentinel()));
        return sentinels == buffer.size() - reached;
    }
};

/// Calls @p f with every multi-index of @p dims, the first index fastest. Nothing for an empty extent;
/// once, with an empty index, for rank 0.
template <typename F>
void for_each_index(std::vector<std::size_t> const &dims, F &&f) {
    for (std::size_t const d : dims) {
        if (d == 0) {
            return;
        }
    }
    std::vector<std::size_t> index(dims.size(), 0);
    while (true) {
        f(static_cast<std::vector<std::size_t> const &>(index));
        std::size_t d = 0;
        for (; d < dims.size(); d++) {
            if (++index[d] < dims[d]) {
                break;
            }
            index[d] = 0;
        }
        if (d == dims.size()) {
            return;
        }
    }
}

/// The logical index of @p index within @p dims, the first index fastest: the k of @ref layout_value.
inline std::size_t logical_index(std::vector<std::size_t> const &dims, std::vector<std::size_t> const &index) {
    std::size_t k = 0, step = 1;
    for (std::size_t d = 0; d < dims.size(); d++) {
        k += index[d] * step;
        step *= dims[d];
    }
    return k;
}

/// Every layout of a tensor with extents @p dims, each filled with @ref layout_value draws from @p seed.
///
/// Dense column- and row-major; each with a gap after every axis, so the innermost axis is unit
/// stride but the whole is not one vector; each with every stride doubled, so no axis is unit stride;
/// and, from rank 3, a dense layout fastest along the middle axis, matching neither order. Rank 0 and
/// rank 1 have fewer distinct layouts and get only those.
template <typename T>
std::vector<Layout<T>> layouts(std::vector<std::size_t> const &dims, unsigned seed) {
    std::size_t const rank = dims.size();

    // Axes from fastest to slowest, the step each axis's stride takes past the one before it, and an
    // overall scale on every stride.
    auto const make = [&](std::string name, std::vector<std::size_t> const &order, std::size_t gap, std::size_t scale) {
        Layout<T>   layout{std::move(name), dims, std::vector<std::size_t>(rank), {}};
        std::size_t stride = 1;
        for (std::size_t const axis : order) {
            layout.strides[axis] = stride * scale;
            stride *= dims[axis] + gap;
        }
        std::size_t extent = 1;
        for (std::size_t d = 0; d < rank; d++) {
            extent += dims[d] == 0 ? 0 : (dims[d] - 1) * layout.strides[d];
        }
        layout.buffer.assign(extent, Layout<T>::sentinel());
        for_each_index(
            dims, [&](std::vector<std::size_t> const &index) { layout.at(index) = layout_value<T>(seed, logical_index(dims, index)); });
        return layout;
    };

    std::vector<std::size_t> column(rank), row(rank);
    std::iota(column.begin(), column.end(), std::size_t{0});
    std::ranges::reverse_copy(column, row.begin());

    std::vector<Layout<T>> out;
    out.push_back(make("column-major", column, 0, 1));
    if (rank == 0) {
        return out;
    }
    out.push_back(make("strided", column, 0, 2));
    if (rank == 1) {
        return out;
    }
    out.push_back(make("row-major", row, 0, 1));
    out.push_back(make("padded column-major", column, 1, 1));
    out.push_back(make("padded row-major", row, 1, 1));
    out.push_back(make("strided row-major", row, 0, 2));
    if (rank >= 3) {
        std::vector<std::size_t> middle{1, 0};
        for (std::size_t d = 2; d < rank; d++) {
            middle.push_back(d);
        }
        out.push_back(make("middle-fastest", middle, 0, 1));
    }
    return out;
}

/// Shapes that reach every layout path: a rank-3 shape with distinct extents, extent-1 axes in front
/// and in the middle, ranks 2, 1 and 0, and an empty extent.
inline std::vector<std::vector<std::size_t>> const &layout_shapes() {
    static std::vector<std::vector<std::size_t>> const shapes = {{3, 4, 5}, {1, 4, 5}, {3, 1, 5}, {4, 3}, {5}, {}, {3, 0, 2}};
    return shapes;
}

/// @p dims as text, for test output.
inline std::string shape_name(std::vector<std::size_t> const &dims) {
    std::string out = "(";
    for (std::size_t d = 0; d < dims.size(); d++) {
        out += (d == 0 ? "" : ", ") + std::to_string(dims[d]);
    }
    return out + ")";
}

/// Whether @p got matches @p want: exactly for an integer type, otherwise to @p ulps units of rounding
/// relative to @p scale, which is the magnitude the result was computed from.
template <typename T>
bool layout_close(T got, T want, double scale = 1.0, double ulps = 8.0) {
    if constexpr (std::is_integral_v<T>) {
        return got == want;
    } else {
        return std::abs(got - want) <= ulps * std::numeric_limits<RemoveComplexT<T>>::epsilon() * (scale + std::abs(want));
    }
}

namespace detail {
template <std::size_t I, typename All, typename F, typename... Chosen>
void combine_layouts(All const &all, F &f, std::vector<std::size_t> const &dims, Chosen const &...chosen) {
    if constexpr (I == std::tuple_size_v<All>) {
        f(dims, Chosen(chosen)...);
    } else {
        for (auto const &layout : std::get<I>(all)) {
            combine_layouts<I + 1>(all, f, dims, chosen..., layout);
        }
    }
}
} // namespace detail

/// Calls @p f(dims, layout...) with a fresh copy of every combination of layouts, one operand of each
/// of @p Ts, for every shape in @ref layout_shapes. Operand i is drawn with seed i + 1.
template <typename... Ts, typename F>
void for_each_layout_combination(F &&f) {
    for (auto const &dims : layout_shapes()) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            auto const all = std::make_tuple(layouts<Ts>(dims, static_cast<unsigned>(I + 1))...);
            detail::combine_layouts<0>(all, f, dims);
        }(std::index_sequence_for<Ts...>{});
    }
}

EINSUMS_NAMESPACE_END(testing)
