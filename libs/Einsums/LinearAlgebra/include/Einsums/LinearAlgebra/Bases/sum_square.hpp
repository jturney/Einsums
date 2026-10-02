//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>
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
void impl_sum_square_contiguous(einsums::detail::TensorImpl<T> const &a, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::lassq(a.size(), a.data(), a.get_incx(), scale, sumsq);
    } else {
        impl_sum_square_unblased(a.size(), a.data(), a.get_incx(), scale, sumsq);
    }
}

template <typename T, Container HardDims, Container OutStrides>
void impl_sum_square_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, T const *in,
                                              OutStrides const &in_strides, size_t inc_in, RemoveComplexT<T> *scale,
                                              RemoveComplexT<T> *sumsq) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::lassq(easy_size, in, inc_in, scale, sumsq);
        } else {
            impl_sum_square_unblased(easy_size, in, inc_in, scale, sumsq);
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_sum_square_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                                     scale, sumsq);
        }
    }
}

template <typename T>
void impl_sum_square(einsums::detail::TensorImpl<T> const &in, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    WAGGLE_ZONE_FUNC();

    if (in.size() == 0) {
        return;
    }

    if (in.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using lassq.");

        impl_sum_square_contiguous(in, scale, sumsq);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over lassq.");

        size_t              easy_size, hard_size, easy_rank, in_incx;
        ShapeVector<size_t> hard_dims, in_strides;

        in.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &in_incx);

        hard_dims.resize(in.rank() - easy_rank);

        // Layout flag (not a stride(0)<stride(-1) proxy, which ties on size-1
        // extents) to match query_vectorable_params' split direction.
        if (in.is_column_major()) {
            in_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i] = in.stride(i + easy_rank);
                hard_dims[i]  = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i] = in.stride(i);
                hard_dims[i]  = in.dim(i);
            }
        }

        impl_sum_square_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, scale,
                                                 sumsq);
    }
}

} // namespace detail
} // namespace linear_algebra
EINSUMS_NAMESPACE_END()