//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <cstring>
#include <stdexcept>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename AType, typename XType, typename YType>
void impl_gemv_noncontiguous(char transA, YType alpha, einsums::detail::TensorImpl<AType> const &A,
                             einsums::detail::TensorImpl<XType> const &X, YType beta, einsums::detail::TensorImpl<YType> *Y) {
    LabeledSection0();

    char const   tA            = std::tolower(transA);
    size_t const a_link_stride = (tA == 'n') ? A.stride(1) : A.stride(0), a_target_stride = (tA == 'n') ? A.stride(0) : A.stride(1);
    size_t const incx = X.get_incx(), incy = Y->get_incx();
    size_t const m = A.dim(0), n = A.dim(1);
    size_t const link_dim = (tA == 'n') ? n : m, target_dim = (tA == 'n') ? m : n;

    AType const *A_data = A.data();
    XType const *X_data = X.data();
    YType       *Y_data = Y->data();

    constexpr YType zero = einsums::detail::convert<int, YType>(0);

    // Scale the output.
    blas::scal(Y->dim(0), beta, Y_data, incy);

    // Do the matrix multiplication.
    //
    // Parallelize over the output index (target) only, so each Y element is
    // owned by exactly one thread, which sums its contributions over all
    // links into a local accumulator and writes the result once. A previous
    // ``collapse(2)`` parallelized the link loop too, so multiple threads did
    // an unsynchronized read-modify-write (``Y_data[target] += ...``) on the
    // same Y element for different links, a data race that intermittently
    // dropped contributions, leaving one output element wrong run-to-run.
    if constexpr (IsComplexV<YType>) {
        if (tA == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for)
            for (size_t target = 0; target < target_dim; target++) {
                YType acc = zero;
                for (size_t link = 0; link < link_dim; link++) {
                    acc += std::conj(A_data[a_target_stride * target + a_link_stride * link]) * X_data[incx * link];
                }
                Y_data[target * incy] += alpha * acc;
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for)
            for (size_t target = 0; target < target_dim; target++) {
                YType acc = zero;
                for (size_t link = 0; link < link_dim; link++) {
                    acc += A_data[a_target_stride * target + a_link_stride * link] * X_data[incx * link];
                }
                Y_data[target * incy] += alpha * acc;
            }
        }
    } else {
        EINSUMS_OMP_PRAGMA(parallel for)
        for (size_t target = 0; target < target_dim; target++) {
            YType acc = zero;
            for (size_t link = 0; link < link_dim; link++) {
                acc += A_data[a_target_stride * target + a_link_stride * link] * X_data[incx * link];
            }
            Y_data[target * incy] += alpha * acc;
        }
    }
}

template <typename T>
void impl_gemv_contiguous(char transA, T alpha, einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &X, T beta,
                          einsums::detail::TensorImpl<T> *Y) {
    bool   colA = A.is_column_major();
    size_t m, n;
    char   tA = transA;

    if (colA) {
        m = A.dim(0);
        n = A.dim(1);
    } else {
        if (transA == 'c' || transA == 'C') {
            impl_gemv_noncontiguous(transA, alpha, A, X, beta, Y);
            return;
        }
        m = A.dim(1);
        n = A.dim(0);

        if (transA == 'n' || transA == 'N') {
            tA = 't';
        } else {
            tA = 'n';
        }
    }

    blas::gemv(tA, m, n, alpha, A.data(), A.get_lda(), X.data(), X.get_incx(), beta, Y->data(), Y->get_incx());
}

template <typename AType, typename XType, typename YType, typename AlphaType, typename BetaType>
void impl_gemv(char transA, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<XType> const &X,
               BetaType beta, einsums::detail::TensorImpl<YType> *Y) {

    // Check the parameters.
    bool tA = (std::tolower(transA) != 'n');

    auto A_m = (tA) ? A.dim(1) : A.dim(0);
    auto A_n = (tA) ? A.dim(0) : A.dim(1);

    if (!std::strchr("cntCNT", transA)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "The transpose character was invalid! Expected c, n, or t, case insensitive, got '{}'.", transA);
    }

    if (A_n != X.dim(0) || A_m != Y->dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error,
                                "The tensors passed to gemv were incompatible! Got transA: '{}', A rows: {}, A columns: {}, X size: {}, Y "
                                "size: {}. Based on transA, X size should be {} and Y size should be {}.",
                                transA, A.dim(0), A.dim(1), X.dim(0), Y->dim(0), A_n, A_m);
    }
    if constexpr (!std::is_same_v<AType, XType> || !std::is_same_v<AType, YType>) {
        impl_gemv_noncontiguous(transA, einsums::detail::convert<AlphaType, YType>(alpha), A, X,
                                einsums::detail::convert<AlphaType, YType>(beta), Y);
    } else {
        if (A.is_gemmable()) {
            impl_gemv_contiguous(transA, einsums::detail::convert<AlphaType, YType>(alpha), A, X,
                                 einsums::detail::convert<AlphaType, YType>(beta), Y);
        } else {
            impl_gemv_noncontiguous(transA, einsums::detail::convert<AlphaType, YType>(alpha), A, X,
                                    einsums::detail::convert<AlphaType, YType>(beta), Y);
        }
    }
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums