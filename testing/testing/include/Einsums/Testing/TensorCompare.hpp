//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TensorCompare.hpp
/// @brief Element-wise comparison of whole tensors for tests.
///
/// Kept out of Testing.hpp so a test that never compares tensors does not pay for TensorImpl.

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <complex>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

EINSUMS_NAMESPACE_BEGIN(testing)

/**
 * @brief How far a value may be from its reference: numpy.isclose's rule.
 *
 * An element passes when ``|got - want| <= atol + rtol * |want|``. Both fields are named at every
 * call site, so the two numbers cannot be swapped unnoticed:
 * ``{.rtol = 1e-12, .atol = 1e-12}`` for a bound relative to the value that still holds near zero,
 * ``{.rtol = 0, .atol = 1e-11}`` for a purely absolute one.
 */
struct Tolerance {
    double rtol;
    double atol;
};

/// What @ref compare_tensors found. Converts to true when the tensors are close.
struct TensorComparison {
    /// Empty when the ranks and extents agree; otherwise how they differ, and no element was checked.
    std::string shape_mismatch;
    /// Elements outside their bound, NaN included.
    std::size_t failures{0};
    std::size_t checked{0};
    /// The first element outside its bound: its index, both values and the bound it missed.
    std::string first_failure;

    [[nodiscard]] bool close() const { return shape_mismatch.empty() && failures == 0; }
    explicit           operator bool() const { return close(); }

    [[nodiscard]] std::string describe() const {
        if (!shape_mismatch.empty()) {
            return shape_mismatch;
        }
        if (failures == 0) {
            return fmt::format("all {} elements within tolerance", checked);
        }
        return fmt::format("{} of {} elements out of tolerance; the first {}", failures, checked, first_failure);
    }
};

namespace detail {

/// Visit every element of a TensorImpl in logical, column-major index order, giving its flat
/// position in that order, its index and the element. Walks the strides, so a view is visited
/// correctly.
template <typename T, typename F>
void for_each_element(einsums::detail::TensorImpl<T> const &t, F &&visit) {
    std::size_t const        rank = t.rank();
    std::size_t const        n    = t.size();
    std::vector<std::size_t> index(rank, 0);
    for (std::size_t flat = 0; flat < n; ++flat) {
        std::ptrdiff_t offset = 0;
        for (std::size_t d = 0; d < rank; ++d) {
            offset += static_cast<std::ptrdiff_t>(index[d]) * static_cast<std::ptrdiff_t>(t.stride(static_cast<int>(d)));
        }
        visit(flat, index, t.data()[offset]);
        for (std::size_t d = 0; d < rank; ++d) {
            if (++index[d] < t.dim(static_cast<int>(d))) {
                break;
            }
            index[d] = 0;
        }
    }
}

/// A value widened for comparison: complex<double> if complex, else double.
template <typename T>
auto widen(T value) {
    if constexpr (IsComplexV<T>) {
        return std::complex<double>(value);
    } else {
        return static_cast<double>(value);
    }
}

} // namespace detail

/// Compare @p got against @p want element by element, without asserting. See @ref require_tensors_close.
template <typename TG, typename TW>
TensorComparison compare_tensors(einsums::detail::TensorImpl<TG> const &got, einsums::detail::TensorImpl<TW> const &want, Tolerance tol) {
    TensorComparison result;
    bool             same_shape = got.rank() == want.rank();
    for (std::size_t d = 0; same_shape && d < got.rank(); ++d) {
        same_shape = got.dim(static_cast<int>(d)) == want.dim(static_cast<int>(d));
    }
    if (!same_shape) {
        result.shape_mismatch = fmt::format("got extents ({}) but want ({})", fmt::join(got.dims(), ", "), fmt::join(want.dims(), ", "));
        return result;
    }

    std::vector<decltype(detail::widen(TW{}))> expected;
    expected.reserve(want.size());
    detail::for_each_element(want, [&](std::size_t, auto const &, TW const &w) { expected.push_back(detail::widen(w)); });

    detail::for_each_element(got, [&](std::size_t flat, std::vector<std::size_t> const &index, TG const &g) {
        auto const   w     = expected[flat];
        double const diff  = std::abs(detail::widen(g) - w);
        double const bound = tol.atol + tol.rtol * std::abs(w);
        ++result.checked;
        if (!(diff <= bound)) { // written this way round so a NaN on either side fails
            if (result.failures++ == 0) {
                result.first_failure = fmt::format("is at index ({}): got {} want {}, |diff| {:.3e} > bound {:.3e}", fmt::join(index, ", "),
                                                   detail::widen(g), w, diff, bound);
            }
        }
    });
    return result;
}

/**
 * @brief Require @p got to have @p want's extents and every element within @p tol of it.
 *
 * One assertion for the whole tensor, whose message names the first element out of tolerance and
 * how many missed, rather than one assertion per element.
 */
template <typename TG, typename TW>
void require_tensors_close(einsums::detail::TensorImpl<TG> const &got, einsums::detail::TensorImpl<TW> const &want, Tolerance tol) {
    auto const comparison = compare_tensors(got, want, tol);
    INFO(comparison.describe());
    REQUIRE(comparison.close());
}

/// @ref compare_tensors for any tensor types that expose their TensorImpl through ``impl()``.
template <typename GotType, typename WantType>
    requires requires(GotType const &g, WantType const &w) {
        g.impl();
        w.impl();
    }
TensorComparison compare_tensors(GotType const &got, WantType const &want, Tolerance tol) {
    return compare_tensors(got.impl(), want.impl(), tol);
}

/// @ref require_tensors_close for any tensor types that expose their TensorImpl through ``impl()``.
template <typename GotType, typename WantType>
    requires requires(GotType const &g, WantType const &w) {
        g.impl();
        w.impl();
    }
void require_tensors_close(GotType const &got, WantType const &want, Tolerance tol) {
    require_tensors_close(got.impl(), want.impl(), tol);
}

/**
 * @brief A tensor's element bytes, in logical index order, for a bit-identity comparison.
 *
 * Walks the strides, so a view gives its own elements rather than whatever lies after its first
 * one in memory. Two tensors compare equal exactly when every element is bitwise identical, which
 * tells -0.0 from 0.0 and one NaN payload from another where ``==`` would not.
 */
template <typename T>
std::vector<unsigned char> bytes_of(einsums::detail::TensorImpl<T> const &t) {
    std::vector<unsigned char> out(t.size() * sizeof(T));
    detail::for_each_element(
        t, [&](std::size_t flat, auto const &, T const &value) { std::memcpy(out.data() + flat * sizeof(T), &value, sizeof(T)); });
    return out;
}

/// @ref bytes_of for any tensor type that exposes its TensorImpl through ``impl()``.
template <typename TensorType>
    requires requires(TensorType const &t) { t.impl(); }
std::vector<unsigned char> bytes_of(TensorType const &t) {
    return bytes_of(t.impl());
}

EINSUMS_NAMESPACE_END(testing)
