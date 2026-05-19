//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <cmath>
#include <complex>
#include <limits>

// Three-tier compensated accumulator helpers used by the BLAS-bypass dot
// and DiskAlgebra dot paths.
//
// Each value is funnelled into one of three running sums based on its
// magnitude (``big_sum`` if scaled-down avoids overflow, ``small_sum`` if
// scaled-up avoids underflow, ``medium_sum`` otherwise). ``combine_accum``
// merges the three tiers into a single result at the end. This avoids the
// catastrophic cancellation that a naive single-accumulator reduction
// would hit when summing across many orders of magnitude.

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename T>
inline void add_scale(T value, T &big_sum, T &medium_sum, T &small_sum, bool &not_big, bool /*ignore*/ = false) {
    constexpr T sfmin  = std::numeric_limits<T>::min();
    constexpr T small  = 1 / std::numeric_limits<T>::max();
    constexpr T smlnum = (small > sfmin) ? small * (1 + std::numeric_limits<T>::epsilon()) : sfmin;
    constexpr T bignum = 1 / smlnum;

    auto ax = std::abs(value);

    if (ax > bignum) {
        big_sum += value * smlnum;
        not_big = false;
    } else if (ax < smlnum) {
        if (not_big) {
            small_sum += value * bignum;
        }
    } else {
        medium_sum += value;
    }
}

template <typename T>
inline void add_scale(std::complex<T> value, std::complex<T> &big_sum, std::complex<T> &medium_sum, std::complex<T> &small_sum,
                      bool &not_big_re, bool &not_big_im) {
    // This is allowed and guaranteed by the C++ standard.
    auto big_array    = reinterpret_cast<T(&)[2]>(big_sum);
    auto medium_array = reinterpret_cast<T(&)[2]>(medium_sum);
    auto small_array  = reinterpret_cast<T(&)[2]>(small_sum);
    add_scale(value.real(), big_array[0], medium_array[0], small_array[0], not_big_re);
    add_scale(value.imag(), big_array[1], medium_array[1], small_array[1], not_big_im);
}

template <typename T>
inline T combine_accum(T big_sum, T medium_sum, T small_sum) {
    constexpr T sfmin  = std::numeric_limits<T>::min();
    constexpr T small  = 1 / std::numeric_limits<T>::max();
    constexpr T smlnum = (small > sfmin) ? small * (1 + std::numeric_limits<T>::epsilon()) : sfmin;
    constexpr T bignum = 1 / smlnum;

    if (big_sum > T{0.0}) {
        if (std::abs(medium_sum) > T{0.0} || std::isnan(medium_sum)) {
            big_sum += medium_sum * smlnum;
        }
        return big_sum / smlnum;
    } else if (std::abs(small_sum) > T{0.0}) {
        if (std::abs(medium_sum) > T{0.0} || std::isnan(medium_sum)) {
            medium_sum += small_sum * bignum;
            return medium_sum;
        } else {
            return small_sum * bignum;
        }
    } else {
        return medium_sum;
    }
}

template <typename T>
inline std::complex<T> combine_accum(std::complex<T> big_sum, std::complex<T> medium_sum, std::complex<T> small_sum) {
    return std::complex<T>{combine_accum(big_sum.real(), medium_sum.real(), small_sum.real()),
                           combine_accum(big_sum.imag(), medium_sum.imag(), small_sum.imag())};
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums
