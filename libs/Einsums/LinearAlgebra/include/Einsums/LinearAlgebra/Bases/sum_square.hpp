//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename T>
void impl_sum_square_contiguous(einsums::detail::TensorImpl<T> const &a, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::lassq(a.size(), a.data(), a.get_incx(), scale, sumsq);
    } else {
        T const     *a_data = a.data();
        size_t const size = a.size(), incx = a.get_incx();
        T            sumsq_local = T{0.0};

        EINSUMS_OMP_PRAGMA(parallel for reduction(+: sumsq_local))
        for (size_t i = 0; i < size; i++) {
            sumsq_local += a_data[i * incx] * a_data[i * incx];
        }
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
            T sumsq_local = T{0.0};
            *scale        = T{1.0};
            EINSUMS_OMP_PRAGMA(parallel for reduction(+: sumsq_local))
            for (size_t i = 0; i < easy_size; i++) {
                sumsq_local += in[i * inc_in] * in[i * inc_in];
            }

            *sumsq = sumsq_local;
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
    LabeledSection0();

    if (in.size() == 0) {
        return;
    }

    if (in.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using lassq.");

        impl_sum_square_contiguous(in, scale, sumsq);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over lassq.");

        size_t               easy_size, hard_size, easy_rank, in_incx;
        BufferVector<size_t> hard_dims, in_strides;

        in.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &in_incx);

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
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
} // namespace einsums