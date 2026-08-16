//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include <cctype>

#include "Common.hpp"

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

extern "C" {
extern void FC_GLOBAL(sgemv, SGEMV)(char *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *, float *, float *,
                                    int_t *);
extern void FC_GLOBAL(dgemv, DGEMV)(char *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *, double *,
                                    double *, int_t *);
extern void FC_GLOBAL(cgemv, CGEMV)(char *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemv, ZGEMV)(char *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *, int_t *,
                                    std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *, int_t *);
}

#define GEMV_CHECK(transa, m, n, lda, incx, incy)                                                                                          \
    char ta = std::tolower(transa);                                                                                                        \
                                                                                                                                           \
    if (ta != 'n' && ta != 'c' && ta != 't') {                                                                                             \
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,                                                                                     \
                                "The first argument (transa) to gemv call is invalid. Expected n, t, or c, case insensitive, got {}.",     \
                                transa);                                                                                                   \
    }                                                                                                                                      \
                                                                                                                                           \
    if ((m) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error,                                                                                                             \
            "The second argument (m) to gemv call is invalid. Expected a number greater than or equal to zero, got {}.", m);               \
    }                                                                                                                                      \
    if ((n) < 0) {                                                                                                                         \
        EINSUMS_THROW_EXCEPTION(                                                                                                           \
            std::domain_error, "The third argument (n) to gemv call is invalid. Expected a number greater than or equal to zero, got {}.", \
            n);                                                                                                                            \
    }                                                                                                                                      \
    if ((lda) < std::max((int_t)1, m)) {                                                                                                   \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The sixth argument (lda) to gemv call is invalid. Expected a number at least 1 and at least {}, got {}.", \
                                m, lda);                                                                                                   \
    }                                                                                                                                      \
    if ((incx) == 0) {                                                                                                                     \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The eighth argument (incx) to gemv call is invalid. Expected a non-zero value, got zero.");               \
    }                                                                                                                                      \
    if ((incy) == 0) {                                                                                                                     \
        EINSUMS_THROW_EXCEPTION(std::domain_error,                                                                                         \
                                "The eleventh argument (incy) to gemv call is invalid. Expected a non-zero value, got zero.");             \
    }

void sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta, float *y,
           int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    FC_GLOBAL(sgemv, SGEMV)(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta, double *y,
           int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    FC_GLOBAL(dgemv, DGEMV)(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda, std::complex<float> const *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    FC_GLOBAL(cgemv, CGEMV)(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
           std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;

    GEMV_CHECK(transa, m, n, lda, incx, incy)

    FC_GLOBAL(zgemv, ZGEMV)(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

EINSUMS_NAMESPACE_END(blas::vendor)