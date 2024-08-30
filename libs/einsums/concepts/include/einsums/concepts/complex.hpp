//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/concepts/tensor.hpp>

#include <complex>
#include <type_traits>

namespace einsums {

template <typename>
inline constexpr bool is_complex_v = false;

template <typename T>
inline constexpr bool is_complex_v<std::complex<T>> =
    std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, long double>>;

template <typename T>
concept IsComplex = requires(T t) { requires std::same_as<T, std::complex<typename T::value_type>>; };

template <typename T>
concept IsComplexTensor = requires(T t) {
    requires std::same_as<typename T::value_type, std::complex<typename T::value_type::value_type>>;
    requires TensorConcept<T>;
};

template <typename T>
concept Complex = IsComplex<T> || IsComplexTensor<T>;

template <typename T>
concept NotComplex = !Complex<T>;

namespace detail {
template <typename T>
struct remove_complex {
    using type = T;
};

template <typename T>
struct remove_complex<std::complex<T>> {
    using type = T;
};

} // namespace detail

template <typename T>
using remove_complex_t = typename detail::remove_complex<T>::type;

namespace detail {
template <typename T>
struct add_complex {
    using type = std::complex<T>;
};

template <typename T>
struct add_complex<std::complex<T>> {
    using type = std::complex<T>;
};
} // namespace detail

template <typename T>
using add_complex_t = typename detail::add_complex<T>::type;

} // namespace einsums