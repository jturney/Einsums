//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums::linear_algebra::detail {

template <typename T>
void impl_ger(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
              einsums::detail::TensorImpl<T> &a) {
    if (a.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to ger need to be rank-1 tensors and the output needs to be a rank-2 tensor.");
    }
    if (x.dim(0) != a.dim(0) || y.dim(0) != a.dim(1)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the tensors passed to ger are incompatible!");
    }

    if (a.is_gemmable()) {
        if (a.is_column_major()) {
            blas::ger(a.dim(0), a.dim(1), alpha, x.data(), x.get_incx(), y.data(), y.get_incy(), a.data(), a.get_lda());
        } else {
            blas::ger(a.dim(1), a.dim(0), alpha, y.data(), y.get_incy(), x.data(), x.get_incx(), a.data(), a.get_lda());
        }
    } else {
        T       *a_data = a.data();
        T const *x_data = x.data();
        T const *y_data = y.data();

        size_t row_stride = a.stride(0), col_stride = a.stride(1), incx = x.get_incx(), incy = y.get_incy(), rows = a.dim(0),
               cols = a.dim(1);

        if (row_stride < col_stride) {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t j = 0; j < cols; j++) {
                for (size_t i = 0; i < rows; i++) {
                    a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                }
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                }
            }
        }
    }
}

template <typename T>
void impl_gerc(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
               einsums::detail::TensorImpl<T> &a) {
    if (a.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to ger need to be rank-1 tensors and the output needs to be a rank-2 tensor.");
    }
    if (x.dim(0) != a.dim(0) || y.dim(0) != a.dim(1)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the tensors passed to ger are incompatible!");
    }

    if (a.is_gemmable()) {
        if (a.is_column_major()) {
            blas::gerc(a.dim(0), a.dim(1), alpha, x.data(), x.get_incx(), y.data(), y.get_incy(), a.data(), a.get_lda());
        } else {
            blas::gerc(a.dim(1), a.dim(0), alpha, y.data(), y.get_incy(), x.data(), x.get_incx(), a.data(), a.get_lda());
        }
    } else {
        T       *a_data = a.data();
        T const *x_data = x.data();
        T const *y_data = y.data();

        size_t row_stride = a.stride(0), col_stride = a.stride(1), incx = x.get_incx(), incy = y.get_incy(), rows = a.dim(0),
               cols = a.dim(1);

        if constexpr (IsComplexV<T>) {
            if (row_stride < col_stride) {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t j = 0; j < cols; j++) {
                    for (size_t i = 0; i < rows; i++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * std::conj(y_data[j * incy]);
                    }
                }
            } else {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * std::conj(y_data[j * incy]);
                    }
                }
            }
        } else {
            if (row_stride < col_stride) {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t j = 0; j < cols; j++) {
                    for (size_t i = 0; i < rows; i++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                    }
                }
            } else {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                    }
                }
            }
        }
    }
}

} // namespace einsums::linear_algebra::detail