//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename AType, typename BType, typename CType>
void impl_gemm_noncontiguous(char transA, char transB, CType alpha, einsums::detail::TensorImpl<AType> const &A,
                             einsums::detail::TensorImpl<BType> const &B, CType beta, einsums::detail::TensorImpl<CType> *C) {
    LabeledSection0();

    char const   tA = std::tolower(transA), tB = std::tolower(transB);
    size_t const a_target_stride = (tA == 'n') ? A.stride(0) : A.stride(1), a_link_stride = (tA == 'n') ? A.stride(1) : A.stride(0);
    size_t const b_target_stride = (tB == 'n') ? B.stride(1) : B.stride(0), b_link_stride = (tB == 'n') ? B.stride(0) : B.stride(1);
    size_t const ca_stride = C->stride(0), cb_stride = C->stride(1);
    size_t const m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);

    AType const *A_data = A.data();
    BType const *B_data = B.data();
    CType       *C_data = C->data();

    constexpr CType zero = einsums::detail::convert<int, CType>(0);

    // Scale the output.
    if (beta == zero) {
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t ca = 0; ca < m; ca++) {
            for (size_t cb = 0; cb < n; cb++) {
                C_data[ca * ca_stride + cb * cb_stride] = zero;
            }
        }
    } else {
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t ca = 0; ca < m; ca++) {
            for (size_t cb = 0; cb < n; cb++) {
                C_data[ca * ca_stride + cb * cb_stride] *= beta;
            }
        }
    }

    // Do the matrix multiplication.
    // Loop order: ca, cb outer (parallelized), link inner (sequential reduction).
    // Each (ca, cb) pair is owned by one thread, so no race on C.
    if constexpr (IsComplexV<CType>) {
        if (tA == 'c' && tB == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t ca = 0; ca < m; ca++) {
                for (size_t cb = 0; cb < n; cb++) {
                    for (size_t link = 0; link < k; link++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * std::conj(A_data[a_target_stride * ca + a_link_stride * link]) *
                                                                   std::conj(B_data[b_target_stride * cb + b_link_stride * link]);
                    }
                }
            }
        } else if (tA == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t ca = 0; ca < m; ca++) {
                for (size_t cb = 0; cb < n; cb++) {
                    for (size_t link = 0; link < k; link++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * std::conj(A_data[a_target_stride * ca + a_link_stride * link]) *
                                                                   B_data[b_target_stride * cb + b_link_stride * link];
                    }
                }
            }
        } else if (tB == 'c') {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t ca = 0; ca < m; ca++) {
                for (size_t cb = 0; cb < n; cb++) {
                    for (size_t link = 0; link < k; link++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * A_data[a_target_stride * ca + a_link_stride * link] *
                                                                   std::conj(B_data[b_target_stride * cb + b_link_stride * link]);
                    }
                }
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t ca = 0; ca < m; ca++) {
                for (size_t cb = 0; cb < n; cb++) {
                    for (size_t link = 0; link < k; link++) {
                        C_data[ca * ca_stride + cb * cb_stride] += alpha * A_data[a_target_stride * ca + a_link_stride * link] *
                                                                   B_data[b_target_stride * cb + b_link_stride * link];
                    }
                }
            }
        }
    } else {
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t ca = 0; ca < m; ca++) {
            for (size_t cb = 0; cb < n; cb++) {
                for (size_t link = 0; link < k; link++) {
                    C_data[ca * ca_stride + cb * cb_stride] +=
                        alpha * A_data[a_target_stride * ca + a_link_stride * link] * B_data[b_target_stride * cb + b_link_stride * link];
                }
            }
        }
    }
}

template <typename T>
void impl_gemm_contiguous(char transA, char transB, T alpha, einsums::detail::TensorImpl<T> const &A,
                          einsums::detail::TensorImpl<T> const &B, T beta, einsums::detail::TensorImpl<T> *C) {
    char tA = std::tolower(transA), tB = std::tolower(transB);
    bool colA = A.is_column_major(), colB = B.is_column_major(), colC = C->is_column_major();
    bool rowA = A.is_row_major(), rowB = B.is_row_major(), rowC = C->is_row_major();

    // So many cases...
    if (colA && colB && colC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        blas::gemm(transA, transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
    } else if (rowA && colB && colC) {
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
                blas::gemm('n', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tA == 't') {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
            blas::gemm('n', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(1);
            blas::gemm('t', transB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (colA && rowB && colC) {
        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
                blas::gemm(transA, 'n', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tB == 't') {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm(transA, 'n', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm(transA, 't', m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (rowA && rowB && colC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tA = 'n';
            }
        } else if (tA == 't') {
            tA = 'n';
        } else {
            tA = 't';
        }

        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tB = 'n';
            }
        } else if (tB == 't') {
            tB = 'n';
        } else {
            tB = 't';
        }

        blas::gemm(tA, tB, m, n, k, alpha, A.data(), A.get_lda(), B.data(), B.get_lda(), beta, C->data(), C->get_lda());
    } else if (colA && colB && rowC) {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tA = 'n';
            }
        } else if (tA == 't') {
            tA = 'n';
        } else {
            tA = 't';
        }

        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
                return;
            } else {
                tB = 'n';
            }
        } else if (tB == 't') {
            tB = 'n';
        } else {
            tB = 't';
        }

        blas::gemm(tB, tA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
    } else if (rowA && colB && rowC) {
        if (tB == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
                blas::gemm('n', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tB == 't') {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm('n', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
            blas::gemm('t', transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        }
    } else if (colA && rowB && rowC) {
        if (tA == 'c') {
            if constexpr (IsComplexV<T>) {
                impl_gemm_noncontiguous(transA, transB, alpha, A, B, beta, C);
            } else {
                auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
                blas::gemm(transB, 'n', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
            }
        } else if (tA == 't') {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(0);
            blas::gemm(transB, 'n', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        } else {
            auto m = C->dim(0), n = C->dim(1), k = A.dim(1);
            blas::gemm(transB, 't', n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
        }
    } else {
        auto m = C->dim(0), n = C->dim(1), k = (tA == 'n') ? A.dim(1) : A.dim(0);
        blas::gemm(transB, transA, n, m, k, alpha, B.data(), B.get_lda(), A.data(), A.get_lda(), beta, C->data(), C->get_lda());
    }
}

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void impl_gemm(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
               einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C);

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void impl_gemm(char transA, char transB, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A,
               einsums::detail::TensorImpl<BType> const &B, BetaType beta, einsums::detail::TensorImpl<CType> *C) {
    if constexpr (!std::is_same_v<AType, BType> || !std::is_same_v<AType, CType>) {
        impl_gemm_noncontiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                einsums::detail::convert<AlphaType, CType>(beta), C);
    } else {
        if (A.is_gemmable() && B.is_gemmable() && C->is_gemmable()) {
            impl_gemm_contiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                 einsums::detail::convert<AlphaType, CType>(beta), C);
        } else {
            impl_gemm_noncontiguous(transA, transB, einsums::detail::convert<AlphaType, CType>(alpha), A, B,
                                    einsums::detail::convert<AlphaType, CType>(beta), C);
        }
    }
}

extern template EINSUMS_EXPORT void impl_gemm<float, float, float, float, float>(char transA, char transB, float alpha,
                                                                                 einsums::detail::TensorImpl<float> const &A,
                                                                                 einsums::detail::TensorImpl<float> const &B, float beta,
                                                                                 einsums::detail::TensorImpl<float> *C);
extern template EINSUMS_EXPORT void impl_gemm<double, double, double, double, double>(char transA, char transB, double alpha,
                                                                                      einsums::detail::TensorImpl<double> const &A,
                                                                                      einsums::detail::TensorImpl<double> const &B,
                                                                                      double beta, einsums::detail::TensorImpl<double> *C);
extern template EINSUMS_EXPORT void
impl_gemm<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
    char transA, char transB, std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
extern template EINSUMS_EXPORT void
impl_gemm<std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>, std::complex<double>>(
    char transA, char transB, std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);
} // namespace detail
} // namespace linear_algebra
} // namespace einsums
