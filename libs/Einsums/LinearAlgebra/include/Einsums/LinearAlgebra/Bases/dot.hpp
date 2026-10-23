//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/LinearAlgebra/Bases/compensated_sum.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <complex>
#include <tuple>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN()
namespace linear_algebra {
namespace detail {

/// The sum over every element of @p term applied to the operands' elements.
///
/// The operands are walked in stretches (see einsums::detail::strided_for_each_run). Each stretch is
/// one call to @p vector, when it returns a value for the stretch and the stretch is long enough, or a
/// parallel reduction of @p term; the stretches' sums combine through ScaledSum.
template <typename R, typename Term, typename Vector, typename... Ts>
R impl_sum_of_products(Term const &term, Vector const &vector, einsums::detail::TensorImpl<Ts> const &...ops) {
    ScaledSum<R>                       total;
    einsums::detail::ParallelThreshold par;
    size_t const                       total_size = std::get<0>(std::forward_as_tuple(ops...)).size();
    einsums::detail::strided_for_each_run(
        std::get<0>(std::forward_as_tuple(ops...)).dims(),
        [&](size_t n, auto const &...r) {
            if constexpr (!std::is_same_v<decltype(vector(n, r...)), void>) {
                if (einsums::detail::use_vector_routine(n, total_size, r...)) {
                    total.add(vector(n, r...));
                    return;
                }
            }
            // A short stretch sums without a parallel region, which costs more to enter than it saves.
            R sum{0.0};
            if (n >= par()) {
                EINSUMS_OMP_PRAGMA(parallel for reduction(+: sum))
                for (size_t i = 0; i < n; i++) {
                    sum += term(r.data[i * r.inc]...);
                }
            } else {
                for (size_t i = 0; i < n; i++) {
                    sum += term(r.data[i * r.inc]...);
                }
            }
            total.add(sum);
        },
        einsums::detail::strided(ops)...);
    return total.total();
}

/// A stand-in for "no vector routine": impl_sum_of_products sees its void result and always loops.
inline constexpr auto no_vector_routine = [](size_t, auto const &...) {};

template <typename T, typename TOther>
BiggestTypeT<T, TOther> impl_dot(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    WAGGLE_ZONE_FUNC();
    einsums::detail::require_same_shape(in, out, "dot");

    using R         = BiggestTypeT<T, TOther>;
    auto const term = [](T const &a, TOther const &b) -> R { return a * b; };
    if constexpr (std::is_same_v<T, TOther> && blas::IsBlasableV<T>) {
        return impl_sum_of_products<R>(
            term, [](size_t n, auto const &a, auto const &b) { return blas::dot(n, a.data, a.inc, b.data, b.inc); }, in, out);
    } else {
        return impl_sum_of_products<R>(term, no_vector_routine, in, out);
    }
}

template <typename T, typename TOther>
BiggestTypeT<T, TOther> impl_true_dot(einsums::detail::TensorImpl<T> const &in, einsums::detail::TensorImpl<TOther> const &out) {
    WAGGLE_ZONE_FUNC();

    if constexpr (!IsComplexV<T>) {
        return impl_dot(in, out);
    } else {
        einsums::detail::require_same_shape(in, out, "dot");

        using R         = BiggestTypeT<T, TOther>;
        auto const term = [](T const &a, TOther const &b) -> R { return std::conj(a) * b; };
        if constexpr (std::is_same_v<T, TOther> && blas::IsBlasableV<T>) {
            return impl_sum_of_products<R>(
                term, [](size_t n, auto const &a, auto const &b) { return blas::dotc(n, a.data, a.inc, b.data, b.inc); }, in, out);
        } else {
            return impl_sum_of_products<R>(term, no_vector_routine, in, out);
        }
    }
}

template <typename A, typename B, typename C>
BiggestTypeT<A, B, C> impl_dot(einsums::detail::TensorImpl<A> const &a, einsums::detail::TensorImpl<B> const &b,
                               einsums::detail::TensorImpl<C> const &c) {
    WAGGLE_ZONE_FUNC();
    einsums::detail::require_same_shape(a, b, "dot");
    einsums::detail::require_same_shape(a, c, "dot");

    using R = BiggestTypeT<A, B, C>;
    return impl_sum_of_products<R>([](A const &x, B const &y, C const &z) -> R { return x * y * z; }, no_vector_routine, a, b, c);
}

extern template EINSUMS_EXTERN_TEMPLATE_EXPORT float  impl_dot<float, float>(einsums::detail::TensorImpl<float> const &a,
                                                                             einsums::detail::TensorImpl<float> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT double impl_dot<double, double>(einsums::detail::TensorImpl<double> const &a,
                                                                               einsums::detail::TensorImpl<double> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT        std::complex<float>
impl_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                   einsums::detail::TensorImpl<std::complex<float>> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT std::complex<double>
impl_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                     einsums::detail::TensorImpl<std::complex<double>> const &b);

extern template EINSUMS_EXTERN_TEMPLATE_EXPORT float  impl_true_dot<float, float>(einsums::detail::TensorImpl<float> const &a,
                                                                                  einsums::detail::TensorImpl<float> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT double impl_true_dot<double, double>(einsums::detail::TensorImpl<double> const &a,
                                                                                    einsums::detail::TensorImpl<double> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT        std::complex<float>
impl_true_dot<std::complex<float>, std::complex<float>>(einsums::detail::TensorImpl<std::complex<float>> const &a,
                                                        einsums::detail::TensorImpl<std::complex<float>> const &b);
extern template EINSUMS_EXTERN_TEMPLATE_EXPORT std::complex<double>
impl_true_dot<std::complex<double>, std::complex<double>>(einsums::detail::TensorImpl<std::complex<double>> const &a,
                                                          einsums::detail::TensorImpl<std::complex<double>> const &b);
} // namespace detail
} // namespace linear_algebra
EINSUMS_NAMESPACE_END()
