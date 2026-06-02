//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#pragma once

#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <list>
#include <vector>

#include "HPTTTypes.hpp"

namespace hptt {

/**
 * Find the conjugate of the given type.
 */
template <typename floatType>
static floatType conj(floatType x) {
    return std::conj(x);
}

/// @copydoc conj<floatType>(floatType)
template <>
float conj(float x) {
    return x;
}

/// @copydoc conj<floatType>(floatType)
template <>
double conj(double x) {
    return x;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
/// @copydoc conj<floatType>(floatType)
template <>
einsums::simd::half_t conj(einsums::simd::half_t x) {
    return x;
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
/// @copydoc conj<floatType>(floatType)
template <>
einsums::simd::bfloat16_t conj(einsums::simd::bfloat16_t x) {
    return x;
}
#endif

/**
 * Get the value that we consider to be zero.
 */
template <typename floatType>
static double get_zero_threshold();

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<double>() {
    return 1e-16;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<DoubleComplex>() {
    return 1e-16;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<float>() {
    return 1e-6;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<FloatComplex>() {
    return 1e-6;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<einsums::simd::half_t>() {
    // FP16 has ~3 decimal digits; 1e-3 captures "effectively zero".
    return 1e-3;
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double get_zero_threshold<einsums::simd::bfloat16_t>() {
    // BF16 has ~2 decimal digits; loose threshold matches the format's resolution.
    return 1e-2;
}
#endif

/**
 * @todo Figure this out.
 */
void trash_cache(double *A, double *B, size_t n);

/**
 * Check whether a vector contains an item.
 */
template <typename t>
int has_item(std::vector<t> const &vec, t value) {
    return (std::find(vec.begin(), vec.end(), value) != vec.end());
}

/// Log the contents of a vector.
template <typename t>
void print_vector(std::vector<t> const &vec, char const *label) {
    EINSUMS_LOG_DEBUG("HPTT: {}: [{}]", label, fmt::join(vec, ", "));
}

/// Log the contents of a list.
template <typename t>
void print_vector(std::list<t> const &vec, char const *label) {
    EINSUMS_LOG_DEBUG("HPTT: {}: [{}]", label, fmt::join(vec, ", "));
}

/**
 * Gets the prime factors of a number. If zero or one is passed, the prime
 * factors output will be empty.
 *
 * @param n The number to factor.
 * @param primeFactors The list of factors.
 */
template <typename T>
void get_prime_factors(T n, std::list<T> &primeFactors);

template <>
void get_prime_factors(std::uint8_t n, std::list<std::uint8_t> &primeFactors);
template <>
void get_prime_factors(std::uint16_t n, std::list<std::uint16_t> &primeFactors);
template <>
void get_prime_factors(std::uint32_t n, std::list<std::uint32_t> &primeFactors);

template <>
void get_prime_factors(std::int8_t n, std::list<std::int8_t> &primeFactors);
template <>
void get_prime_factors(std::int16_t n, std::list<std::int16_t> &primeFactors);
template <>
void get_prime_factors(std::int32_t n, std::list<std::int32_t> &primeFactors);

/**
 * Find where a value is in an array.
 */
template <typename t>
int find_pos(t value, std::vector<t> const &array) {
    for (int i = 0; i < array.size(); ++i)
        if (array[i] == value)
            return i;
    return -1;
}

/**
 * Find where a value is in an array.
 *
 * @param value The value to find.
 * @param array The array of values.
 * @param n The maximum index to check.
 *
 * @return The position of the value, or -1 if not found.
 */
int find_pos(int value, int const *array, int n);

/**
 * Calculate the factorial of a number. Can only take factorials of numbers up to but not
 * including 21.
 *
 * @param n The number to take the factorial of.
 *
 * @return The factorial of that number.
 *
 * @throws std::overflow_error Throws this if the result would be too high to represent.
 */
std::uint64_t factorial(std::uint8_t n);

/**
 * Reorders parameters to swap between row-major and column-major forms.
 */
void account_for_row_major(size_t const *sizeA, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
                           size_t const *offsetB, int const *perm, size_t *tmpSizeA, size_t *tmpOuterSizeA, size_t *tmpouterSizeB,
                           size_t *tmpOffsetA, size_t *tmpOffsetB, int *tmpPerm, int const dim, bool const useRowMajor);
} // namespace hptt
