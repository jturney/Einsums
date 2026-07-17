//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

extern "C" {
extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int_t *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *,
                                    float *, float *, int_t *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int_t *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *,
                                    double *, double *, int_t *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *,
                                    int_t *, std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *,
                                    int_t *);
}

#define GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)                                                                                 \
    const bool notA = (std::tolower(transa) == 'n'), notB = (std::tolower(transb) == 'n');                                                 \
    int_t      nrowa, nrowb;                                                                                                               \
                                                                                                                                           \
    if (notA) {                                                                                                                            \
        nrowa = m;                                                                                                                         \
    } else {                                                                                                                               \
        nrowa = k;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    if (notB) {                                                                                                                            \
        nrowb = k;                                                                                                                         \
    } else {                                                                                                                               \
        nrowb = n;                                                                                                                         \
    }                                                                                                                                      \
                                                                                                                                           \
    if (!notA && std::tolower(transa) != 'c' && std::tolower(transa) != 't') {                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The first argument (transA) to gemm call was invalid! Expected n, t, or c, case-insensitive, got {}.",    \
                                transa);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if (!notB && std::tolower(transb) != 'c' && std::tolower(transb) != 't') {                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The second argument (transB) to gemm call was invalid! Expected n, t, or c, case-insensitive, got {}.",   \
                                transb);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The third argument (m) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", m);  \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The fourth argument (n) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", n); \
    }                                                                                                                                      \
    if ((k) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The fifth argument (k) to gemm call was invalid! It must be greater than or equal to zero. Got {}.", k);  \
    }                                                                                                                                      \
    if ((lda) < std::max((int_t)1, nrowa)) {                                                                                               \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The eighth argument (lda) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",     \
            nrowa, lda);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldb) < std::max((int_t)1, nrowb)) {                                                                                               \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The tenth argument (ldb) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.",      \
            nrowb, ldb);                                                                                                                   \
    }                                                                                                                                      \
    if ((ldc) < std::max((int_t)1, m)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The thirteenth argument (ldc) to gemm call was invalid! It must be at least 1 and at least the number of rows ({}). Got {}.", \
            m, ldc);                                                                                                                       \
    }

void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
           float beta, float *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    if (k == 0) {
        // BLAS semantics: an empty contraction still scales the output,
        // C = beta*C (beta == 0 assigns zero, never propagating stale NaNs).
        // Handled here because empty A/B legitimately carry ld 0, which
        // GEMM_CHECK would reject even though no element is read (bug-1024).
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                if (beta == decltype(beta){0}) {
                    c[static_cast<size_t>(j) * ldc + i] = decltype(beta){0};
                } else {
                    c[static_cast<size_t>(j) * ldc + i] *= beta;
                }
            }
        }
        return;
    }

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    FC_GLOBAL(sgemm, SGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
           double beta, double *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    if (k == 0) {
        // BLAS semantics: an empty contraction still scales the output,
        // C = beta*C (beta == 0 assigns zero, never propagating stale NaNs).
        // Handled here because empty A/B legitimately carry ld 0, which
        // GEMM_CHECK would reject even though no element is read (bug-1024).
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                if (beta == decltype(beta){0}) {
                    c[static_cast<size_t>(j) * ldc + i] = decltype(beta){0};
                } else {
                    c[static_cast<size_t>(j) * ldc + i] *= beta;
                }
            }
        }
        return;
    }

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    FC_GLOBAL(dgemm, DGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    if (k == 0) {
        // BLAS semantics: an empty contraction still scales the output,
        // C = beta*C (beta == 0 assigns zero, never propagating stale NaNs).
        // Handled here because empty A/B legitimately carry ld 0, which
        // GEMM_CHECK would reject even though no element is read (bug-1024).
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                if (beta == decltype(beta){0}) {
                    c[static_cast<size_t>(j) * ldc + i] = decltype(beta){0};
                } else {
                    c[static_cast<size_t>(j) * ldc + i] *= beta;
                }
            }
        }
        return;
    }

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    FC_GLOBAL(cgemm, CGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    if (k == 0) {
        // BLAS semantics: an empty contraction still scales the output,
        // C = beta*C (beta == 0 assigns zero, never propagating stale NaNs).
        // Handled here because empty A/B legitimately carry ld 0, which
        // GEMM_CHECK would reject even though no element is read (bug-1024).
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                if (beta == decltype(beta){0}) {
                    c[static_cast<size_t>(j) * ldc + i] = decltype(beta){0};
                } else {
                    c[static_cast<size_t>(j) * ldc + i] *= beta;
                }
            }
        }
        return;
    }

    GEMM_CHECK(transa, transb, m, n, k, lda, ldb, ldc)

    FC_GLOBAL(zgemm, ZGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

} // namespace einsums::blas::vendor