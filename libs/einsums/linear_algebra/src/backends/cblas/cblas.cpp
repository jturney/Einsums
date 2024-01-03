//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Print.hpp"
#include "einsums/Section.hpp"
#include "einsums/_Common.hpp"
#include "fmt/format.h"

#include <exception>

#if defined(EINSUMS_HAVE_CBLAS_H)
#    include <cblas.h>
#endif

#if defined(EINSUMS_HAVE_LAPACKE_H)
#    include <lapacke.h>
#endif

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::backend::linear_algebra::cblas)

namespace {

auto transpose_to_cblas(char transpose) -> CBLAS_TRANSPOSE {
    switch (transpose) {
    case 'N':
    case 'n':
        return CblasNoTrans;
    case 'T':
    case 't':
        return CblasTrans;
    case 'C':
    case 'c':
        return CblasConjTrans;
    }
    println_warn("Unknown transpose code {}, defaulting to CblasNoTrans.", transpose);
    return CblasNoTrans;
}

} // namespace

void initialize() {
}

void finalize() {
}

void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_sgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, // NOLINT
           double beta, double *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_dgemm(CblasRowMajor, TransA, TransB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_cgemm(CblasRowMajor, TransA, TransB, m, n, k, static_cast<void *>(&alpha), static_cast<const void *>(a), lda,
                static_cast<const void *>(b), ldb, static_cast<void *>(&beta), static_cast<void *>(c), ldc);
}

void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    LabeledSection0();

    if (m == 0 || n == 0 || k == 0)
        return;

    auto TransA = transpose_to_cblas(transa);
    auto TransB = transpose_to_cblas(transb);

    cblas_zgemm(CblasRowMajor, TransA, TransB, m, n, k, static_cast<void *>(&alpha), static_cast<const void *>(a), lda,
                static_cast<const void *>(b), ldb, static_cast<void *>(&beta), static_cast<void *>(c), ldc);
}

void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_sgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_dgemv(CblasRowMajor, TransA, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_cgemv(CblasRowMajor, TransA, m, n, static_cast<const void *>(&alpha), static_cast<const void *>(a), lda, x, incx,
                static_cast<const void *>(&beta), static_cast<void *>(y), incy);
}

void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    LabeledSection0();

    if (m == 0 || n == 0)
        return;
    auto TransA = transpose_to_cblas(transa);
    if (TransA == CblasConjTrans)
        throw std::invalid_argument("einsums::backend::cblas::dgemv transa argument is invalid.");

    cblas_zgemv(CblasRowMajor, TransA, m, n, static_cast<const void *>(&alpha), static_cast<const void *>(a), lda, x, incx,
                static_cast<const void *>(&beta), static_cast<void *>(y), incy);
}

auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *, int) -> int {
    LabeledSection0();

    return LAPACKE_ssyev(LAPACK_ROW_MAJOR, job, uplo, n, a, lda, w);
}

auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *, int) -> int {
    LabeledSection0();

    return LAPACKE_dsyev(LAPACK_ROW_MAJOR, job, uplo, n, a, lda, w);
}

auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> * /*work*/, int /*lwork*/,
           float * /*rwork*/) -> int {
    LabeledSection0();

    return LAPACKE_cheev(LAPACK_ROW_MAJOR, job, uplo, n, reinterpret_cast<lapack_complex_float *>(a), lda, w);
}

auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> * /*work*/, int /*lwork*/,
           double * /*rwork*/) -> int {
    LabeledSection0();

    return LAPACKE_zheev(LAPACK_ROW_MAJOR, job, uplo, n, reinterpret_cast<lapack_complex_double *>(a), lda, w);
}

auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int {
    LabeledSection0();

    return LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int {
    LabeledSection0();

    return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int {
    LabeledSection0();

    return LAPACKE_cgesv(LAPACK_ROW_MAJOR, n, nrhs, reinterpret_cast<lapack_complex_float *>(a), lda, ipiv,
                         reinterpret_cast<lapack_complex_float *>(b), ldb);
}

auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int {
    LabeledSection0();

    return LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, nrhs, reinterpret_cast<lapack_complex_double *>(a), lda, ipiv,
                         reinterpret_cast<lapack_complex_double *>(b), ldb);
}

void sscal(int n, float alpha, float *vec, int inc) {
    LabeledSection0();

    cblas_sscal(n, alpha, vec, inc);
}

void dscal(int n, double alpha, double *vec, int inc) {
    LabeledSection0();

    cblas_dscal(n, alpha, vec, inc);
}

void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc) {
    LabeledSection0();

    cblas_cscal(n, static_cast<const void *>(&alpha), static_cast<void *>(vec), inc);
}

void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc) {
    LabeledSection0();

    cblas_zscal(n, static_cast<const void *>(&alpha), static_cast<void *>(vec), inc);
}

auto sdot(int n, const float *x, int incx, const float *y, int incy) -> double {
    LabeledSection0();

    return cblas_sdot(n, x, incx, y, incy);
}

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double {
    LabeledSection0();

    return cblas_ddot(n, x, incx, y, incy);
}

auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float> {
    LabeledSection0();

    std::complex<float> result;

    cblas_cdotu_sub(n, static_cast<const void *>(x), incx, static_cast<const void *>(y), incy, static_cast<void *>(&result));

    return result;
}

auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double> {
    LabeledSection0();

    std::complex<double> result;

    cblas_zdotu_sub(n, static_cast<const void *>(x), incx, static_cast<const void *>(y), incy, static_cast<void *>(&result));

    return result;
}

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y) {
    LabeledSection0();

    cblas_saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y) {
    LabeledSection0();

    cblas_daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y) {
    LabeledSection0();

    cblas_caxpy(n, static_cast<const void *>(&alpha_x), static_cast<const void *>(x), inc_x, static_cast<void *>(y), inc_y);
}

void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y) {
    LabeledSection0();

    cblas_zaxpy(n, static_cast<const void *>(&alpha_x), static_cast<const void *>(x), inc_x, static_cast<void *>(y), inc_y);
}

void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda) {
    LabeledSection0();

    if (m < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: m ({}) is less than zero.", m));
    } else if (n < 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: n ({}) is less than zero.", n));
    } else if (inc_x == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: inc_x ({}) is zero.", inc_x));
    } else if (inc_y == 0) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: inc_y ({}) is zero.", inc_y));
    } else if (lda < std::max(1, n)) {
        throw std::runtime_error(fmt::format("einsums::backend::cblas::dger: lda ({}) is less than max(1, n ({})).", lda, n));
    }

    cblas_dger(CblasRowMajor, m, n, alpha, y, inc_y, x, inc_x, a, lda);
}

auto dgetrf(int m, int n, double *a, int lda, int *ipiv) -> int {
    LabeledSection0();

    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv);
}

auto dgetri(int n, double *a, int lda, const int *ipiv, double *, int) -> int {
    LabeledSection0();

    return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, lda, (int *)ipiv);
}

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *) -> double {
    LabeledSection0();

    return LAPACKE_dlange(LAPACK_ROW_MAJOR, norm_type, m, n, A, lda);
}

auto sgesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *superb)
    -> int {
    LabeledSection0();
    return LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *superb)
    -> int {
    LabeledSection0();
    return LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto sgesdd(char jobz, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt) -> int {
    LabeledSection0();

    return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt) -> int {
    LabeledSection0();

    return LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu, std::complex<float> *vt,
            int ldvt) -> int {
    LabeledSection0();

    return LAPACKE_cgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_float *>(a), lda, s,
                          reinterpret_cast<lapack_complex_float *>(u), ldu, reinterpret_cast<lapack_complex_float *>(vt), ldvt);
}

auto zgesdd(char jobz, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
            std::complex<double> *vt, int ldvt) -> int {
    LabeledSection0();

    return LAPACKE_zgesdd(LAPACK_ROW_MAJOR, jobz, m, n, reinterpret_cast<lapack_complex_double *>(a), lda, s,
                          reinterpret_cast<lapack_complex_double *>(u), ldu, reinterpret_cast<lapack_complex_double *>(vt), ldvt);
}

auto sgees(char jobvs, int n, float *a, int lda, int *sdim, float *wr, float *wi, float *vs, int ldvs) -> int {
    LabeledSection0();

    return LAPACKE_sgees(LAPACK_ROW_MAJOR, jobvs, 'N', nullptr, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int {
    LabeledSection0();

    return LAPACKE_dgees(LAPACK_ROW_MAJOR, jobvs, 'N', nullptr, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto strsyl(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
            float *scale) -> int {
    LabeledSection0();

    return LAPACKE_strsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c, int ldc,
            double *scale) -> int {
    LabeledSection0();

    return LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto ctrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
            std::complex<float> *c, int ldc, float *scale) -> int {
    LabeledSection0();

    return LAPACKE_ctrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, reinterpret_cast<const lapack_complex_float *>(a), lda,
                          reinterpret_cast<const lapack_complex_float *>(b), ldb, reinterpret_cast<lapack_complex_float *>(c), ldc, scale);
}

auto ztrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
            std::complex<double> *c, int ldc, double *scale) -> int {
    LabeledSection0();

    return LAPACKE_ztrsyl(LAPACK_ROW_MAJOR, trana, tranb, isgn, m, n, reinterpret_cast<const lapack_complex_double *>(a), lda,
                          reinterpret_cast<const lapack_complex_double *>(b), ldb, reinterpret_cast<lapack_complex_double *>(c), ldc,
                          scale);
}

auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int {
    LabeledSection0();

    return LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
}

auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int {
    LabeledSection0();

    return LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);
}

auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int {
    LabeledSection0();

    return LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float *>(tau));
}

auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int {
    LabeledSection0();

    return LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double *>(tau));
}

auto sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int {
    LabeledSection0();

    return LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int {
    LabeledSection0();

    return LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, a, lda, tau);
}

auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int {
    LabeledSection0();

    return LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<const lapack_complex_float *>(tau));
}

auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int {
    LabeledSection0();

    return LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, n, k, reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<const lapack_complex_double *>(tau));
}

END_EINSUMS_NAMESPACE_CPP(einsums::backend::cblas)