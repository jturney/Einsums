//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <complex>
#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(linear_algebra::detail)

/// a += alpha x y^T, or alpha x y^H when Conjugate, on any rank-2 a.
template <bool Conjugate, typename T>
void impl_ger_update(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
                     einsums::detail::TensorImpl<T> &a) {
    if (a.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(RankError, "The inputs to ger need to be rank-1 tensors and the output needs to be a rank-2 tensor.");
    }
    if (x.dim(0) != a.dim(0) || y.dim(0) != a.dim(1)) {
        EINSUMS_THROW_EXCEPTION(TensorCompatError, "The dimensions of the tensors passed to ger are incompatible!");
    }

    constexpr bool conjugate = Conjugate && IsComplexV<T>;
    size_t const   rows = a.dim(0), cols = a.dim(1), incx = x.get_incx(), incy = y.get_incy();

    if (a.is_gemmable()) {
        if (a.is_column_major()) {
            if constexpr (conjugate) {
                blas::gerc(rows, cols, alpha, x.data(), incx, y.data(), incy, a.data(), a.get_lda());
            } else {
                blas::ger(rows, cols, alpha, x.data(), incx, y.data(), incy, a.data(), a.get_lda());
            }
        } else if constexpr (conjugate) {
            // BLAS sees a row-major a as a^T = alpha conj(y) x^T, which conjugates the first vector:
            // gerc with x and y swapped would conjugate x instead. So conjugate y into scratch and
            // take the unconjugated update.
            T const        *y_data = y.data();
            BufferVector<T> y_conj(cols);
            for (size_t j = 0; j < cols; j++) {
                y_conj[j] = std::conj(y_data[j * incy]);
            }
            blas::ger(cols, rows, alpha, y_conj.data(), 1, x.data(), incx, a.data(), a.get_lda());
        } else {
            blas::ger(cols, rows, alpha, y.data(), incy, x.data(), incx, a.data(), a.get_lda());
        }
        return;
    }

    // Walk a with its smaller stride innermost. Each operand's step along the outer and inner loop
    // follows from that choice, so one loop serves both orders.
    bool const   cols_outer = a.stride(0) < a.stride(1);
    size_t const outer = cols_outer ? cols : rows, inner = cols_outer ? rows : cols;
    size_t const a_outer = cols_outer ? a.stride(1) : a.stride(0), a_inner = cols_outer ? a.stride(0) : a.stride(1);
    size_t const x_outer = cols_outer ? 0 : incx, x_inner = cols_outer ? incx : 0;
    size_t const y_outer = cols_outer ? incy : 0, y_inner = cols_outer ? 0 : incy;

    T       *a_data = a.data();
    T const *x_data = x.data();
    T const *y_data = y.data();

    EINSUMS_OMP_PRAGMA(parallel for collapse(2) if (rows * cols >= ::einsums::hardware::omp_min_parallel_elements()))
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            T const y_value = y_data[o * y_outer + i * y_inner];
            if constexpr (conjugate) {
                a_data[o * a_outer + i * a_inner] += alpha * x_data[o * x_outer + i * x_inner] * std::conj(y_value);
            } else {
                a_data[o * a_outer + i * a_inner] += alpha * x_data[o * x_outer + i * x_inner] * y_value;
            }
        }
    }
}

template <typename T>
void impl_ger(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
              einsums::detail::TensorImpl<T> &a) {
    impl_ger_update<false>(alpha, x, y, a);
}

template <typename T>
void impl_gerc(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
               einsums::detail::TensorImpl<T> &a) {
    impl_ger_update<true>(alpha, x, y, a);
}

EINSUMS_NAMESPACE_END(linear_algebra::detail)
