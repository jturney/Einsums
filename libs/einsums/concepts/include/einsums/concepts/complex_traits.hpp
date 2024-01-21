//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <complex>
#include <type_traits>

namespace einsums {

template <typename>
inline constexpr bool IsComplexV = false;

template <typename T>
inline constexpr bool IsComplexV<std::complex<T>> =
    std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, long double>>;

template <typename T>
struct IsComplex : std::bool_constant<IsComplexV<T>> {};

/**
 * @brief C++20 concept for requiring datatype T be complex.
 *
 * @tparam T the datatype to analyze
 */
template <typename T>
concept Complex = IsComplexV<T>;

template <typename T>
struct ComplexType {
    using Type = T;
};

template <typename T>
struct ComplexType<std::complex<T>> {
    using Type = T;
};

template <typename T>
using RemoveComplexT = typename ComplexType<T>::Type;

template <typename T>
struct AddComplex {
    using Type = std::complex<T>;
};

template <typename T>
struct AddComplex<std::complex<T>> {
    using Type = std::complex<T>;
};

template <typename T>
using AddComplexT = typename AddComplex<T>::Type;

} // namespace einsums