//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/BLASVendor.hpp>
#include <Einsums/LinearAlgebra/Bases/sum_square.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename T>
    requires(blas::IsBlasableV<T>)
auto impl_max_abs_norm_gemmable(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.is_column_major()) {
        return blas::lange('M', A.dim(0), A.dim(1), A.data(), A.get_lda(), nullptr);
    } else {
        return blas::lange('M', A.dim(1), A.dim(0), A.data(), A.get_lda(), nullptr);
    }
}

template <typename T>
    requires(blas::IsBlasableV<T>)
auto impl_one_norm_gemmable(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.is_column_major()) {
        return blas::lange('O', A.dim(0), A.dim(1), A.data(), A.get_lda(), nullptr);
    } else {
        BufferVector<RemoveComplexT<T>> work(4 * A.dim(1));
        return blas::lange('I', A.dim(1), A.dim(0), A.data(), A.get_lda(), work.data());
    }
}

template <typename T>
    requires(blas::IsBlasableV<T>)
auto impl_infinity_norm_gemmable(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.is_column_major()) {
        BufferVector<RemoveComplexT<T>> work(4 * A.dim(0));
        return blas::lange('I', A.dim(0), A.dim(1), A.data(), A.get_lda(), work.data());
    } else {
        return blas::lange('O', A.dim(1), A.dim(0), A.data(), A.get_lda(), nullptr);
    }
}

template <typename T>
    requires(blas::IsBlasableV<T>)
auto impl_frobenius_norm_gemmable(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.is_column_major()) {
        return blas::lange('F', A.dim(0), A.dim(1), A.data(), A.get_lda(), nullptr);
    } else {
        return blas::lange('F', A.dim(1), A.dim(0), A.data(), A.get_lda(), nullptr);
    }
}

template <typename T>
auto impl_max_abs_norm(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.rank() == 1) {
        RemoveComplexT<T> out{0.0};
        size_t const      n = A.dim(0), incx = A.get_incx();
        T const          *A_data = A.data();

        for (size_t i = 0; i < n; i++) {
            RemoveComplexT<T> temp = std::abs(A_data[i * incx]);
            if (temp > out) {
                out = temp;
            }
        }
        return out;
    } else {
        size_t const m = A.dim(0), n = A.dim(1), row_stride = A.stride(0), col_stride = A.stride(1);
        T const     *A_data = A.data();

        if (col_stride < row_stride) {
            RemoveComplexT<T> out{0.0};

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < m; i++) {
                RemoveComplexT<T> temp_max{0.0};
                T const          *A_base = A_data + i * row_stride;
                for (size_t j = 0; j < n; j++) {
                    auto hold = std::abs(A_base[j * col_stride]);
                    if (hold > temp_max) {
                        temp_max = hold;
                    }
                }

                EINSUMS_OMP_CRITICAL {
                    if (temp_max > out) {
                        out = temp_max;
                    }
                }
            }

            return out;
        } else {
            RemoveComplexT<T> out{0.0};

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < n; i++) {
                RemoveComplexT<T> temp_max{0.0};
                T const          *A_base = A_data + i * col_stride;
                for (size_t j = 0; j < m; j++) {
                    auto hold = std::abs(A_base[j * row_stride]);
                    if (hold > temp_max) {
                        temp_max = hold;
                    }
                }

                EINSUMS_OMP_CRITICAL {
                    if (temp_max > out) {
                        out = temp_max;
                    }
                }
            }

            return out;
        }
    }
}

template <typename T>
auto impl_infinity_norm(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.rank() == 1) {
        RemoveComplexT<T> out{0.0};
        size_t const      n = A.dim(0), incx = A.get_incx();
        T const          *A_data = A.data();

        for (size_t i = 0; i < n; i++) {
            RemoveComplexT<T> temp = std::abs(A_data[i * incx]);
            if (temp > out) {
                out = temp;
            }
        }
        return out;
    } else {
        size_t const incx = A.stride(1), row_stride = A.stride(0), rows = A.dim(0), cols = A.dim(1);

        T const *A_data = A.data();

        RemoveComplexT<T> out{0.0};

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < rows; i++) {
            auto curr = blas::sum1(cols, A_data + i * row_stride, incx);

            EINSUMS_OMP_CRITICAL {
                if (curr > out) {
                    out = curr;
                }
            }
        }

        return out;
    }
}

template <typename T>
auto impl_one_norm(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.rank() == 1) {
        RemoveComplexT<T> out{0.0};
        size_t const      n = A.dim(0), incx = A.get_incx();
        T const          *A_data = A.data();

        EINSUMS_OMP_PRAGMA(parallel for reduction(+: out))
        for (size_t i = 0; i < n; i++) {
            out += std::abs(A_data[i * incx]);
        }
        return out;
    } else {
        size_t const incx = A.stride(0), col_stride = A.stride(1), rows = A.dim(0), cols = A.dim(1);

        T const *A_data = A.data();

        RemoveComplexT<T> out{0.0};

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < cols; i++) {
            auto curr = blas::sum1(rows, A_data + i * col_stride, incx);

            EINSUMS_OMP_CRITICAL {
                if (curr > out) {
                    out = curr;
                }
            }
        }

        return out;
    }
}

template <typename T>
auto impl_frobenius_norm(einsums::detail::TensorImpl<T> const &A) -> RemoveComplexT<T> {
    if (A.rank() == 1) {
        return blas::nrm2(A.dim(0), A.data(), A.get_incx());
    } else {
        RemoveComplexT<T> sumsq{0.0}, scale{1.0};
        impl_sum_square(A, &scale, &sumsq);

        return scale * std::sqrt(sumsq);
    }
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums