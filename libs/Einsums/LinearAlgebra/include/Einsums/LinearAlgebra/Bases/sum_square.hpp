//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/StridedForEach.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <complex>

EINSUMS_NAMESPACE_BEGIN()
namespace linear_algebra {
namespace detail {

/// lassq for a type the vendor has no lassq for: add sum |x_i|^2 to scale^2 * sumsq. The result is
/// left as scale = 1 and sumsq = the total, since such a type need not have a square root that
/// round-trips. Each call adds to what it was given, so a strided walk can call it once per run.
template <typename T>
void impl_sum_square_unblased(size_t n, T const *x, size_t inc, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    using Real = RemoveComplexT<T>;
    Real added{};
    EINSUMS_OMP_PRAGMA(parallel for reduction(+: added))
    for (size_t i = 0; i < n; i++) {
        if constexpr (IsComplexV<T>) {
            added += std::norm(x[i * inc]);
        } else {
            added += x[i * inc] * x[i * inc];
        }
    }
    *sumsq = *scale * *scale * *sumsq + added;
    *scale = Real{1};
}

template <typename T>
void impl_sum_square(einsums::detail::TensorImpl<T> const &in, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    WAGGLE_ZONE_FUNC();

    // Each stretch adds to the scale and sum the last one left.
    einsums::detail::strided_for_each_run(
        in.dims(),
        [&](size_t n, einsums::detail::RunOperand<T const> r) {
            if constexpr (blas::IsBlasableV<T>) {
                blas::lassq(n, r.data, r.inc, scale, sumsq);
            } else {
                impl_sum_square_unblased(n, r.data, r.inc, scale, sumsq);
            }
        },
        einsums::detail::strided(in));
}

} // namespace detail
} // namespace linear_algebra
EINSUMS_NAMESPACE_END()
