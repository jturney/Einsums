//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/Blas.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "backends/cblas/cblas.hpp"
#include "backends/mkl/mkl.hpp"
#include "backends/netlib/Netlib.hpp"
#include "backends/onemkl/onemkl.hpp"
#include "backends/vendor/Vendor.hpp"

namespace einsums::blas {

void initialize() {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::initialize();
}

void finalize() {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::finalize();
}

namespace detail {
void sgemm(char transa, char transb, eint m, eint n, eint k, float alpha, float const *a, eint lda, float const *b, eint ldb, float beta,
           float *c, eint ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void dgemm(char transa, char transb, eint m, eint n, eint k, double alpha, double const *a, eint lda, double const *b, eint ldb,
           double beta, double *c, eint ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void cgemm(char transa, char transb, eint m, eint n, eint k, std::complex<float> alpha, std::complex<float> const *a, eint lda,
           std::complex<float> const *b, eint ldb, std::complex<float> beta, std::complex<float> *c, eint ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}
void zgemm(char transa, char transb, eint m, eint n, eint k, std::complex<double> alpha, std::complex<double> const *a, eint lda,
           std::complex<double> const *b, eint ldb, std::complex<double> beta, std::complex<double> *c, eint ldc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                                                                ldc);
}

void sgemv(char transa, eint m, eint n, float alpha, float const *a, eint lda, float const *x, eint incx, float beta, float *y, eint incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, eint m, eint n, double alpha, double const *a, eint lda, double const *x, eint incx, double beta, double *y,
           eint incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, eint m, eint n, std::complex<float> alpha, std::complex<float> const *a, eint lda, std::complex<float> const *x,
           eint incx, std::complex<float> beta, std::complex<float> *y, eint incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(char transa, eint m, eint n, std::complex<double> alpha, std::complex<double> const *a, eint lda, std::complex<double> const *x,
           eint incx, std::complex<double> beta, std::complex<double> *y, eint incy) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

auto ssyev(char job, char uplo, eint n, float *a, eint lda, float *w, float *work, eint lwork) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

auto dsyev(char job, char uplo, eint n, double *a, eint lda, double *w, double *work, eint lwork) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

auto sgesv(eint n, eint nrhs, float *a, eint lda, eint *ipiv, float *b, eint ldb) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(eint n, eint nrhs, double *a, eint lda, eint *ipiv, double *b, eint ldb) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(eint n, eint nrhs, std::complex<float> *a, eint lda, eint *ipiv, std::complex<float> *b, eint ldb) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgesv(eint n, eint nrhs, std::complex<double> *a, eint lda, eint *ipiv, std::complex<double> *b, eint ldb) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cheev(char job, char uplo, eint n, std::complex<float> *a, eint lda, float *w, std::complex<float> *work, eint lwork, float *rwork)
    -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

auto zheev(char job, char uplo, eint n, std::complex<double> *a, eint lda, double *w, std::complex<double> *work, eint lwork, double *rwork)
    -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

void sscal(eint n, float alpha, float *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sscal(n, alpha, vec, inc);
}

void dscal(eint n, double alpha, double *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dscal(n, alpha, vec, inc);
}

void cscal(eint n, std::complex<float> alpha, std::complex<float> *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cscal(n, alpha, vec, inc);
}

void zscal(eint n, std::complex<double> alpha, std::complex<double> *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zscal(n, alpha, vec, inc);
}

void csscal(eint n, float alpha, std::complex<float> *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::csscal(n, alpha, vec, inc);
}

void zdscal(eint n, double alpha, std::complex<double> *vec, eint inc) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zdscal(n, alpha, vec, inc);
}

auto sdot(eint n, float const *x, eint incx, float const *y, eint incy) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sdot(n, x, incx, y, incy);
}

auto ddot(eint n, double const *x, eint incx, double const *y, eint incy) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::ddot(n, x, incx, y, incy);
}

auto cdot(eint n, std::complex<float> const *x, eint incx, std::complex<float> const *y, eint incy) -> std::complex<float> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cdot(n, x, incx, y, incy);
}

auto zdot(eint n, std::complex<double> const *x, eint incx, std::complex<double> const *y, eint incy) -> std::complex<double> {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zdot(n, x, incx, y, incy);
}

void saxpy(eint n, float alpha_x, float const *x, eint inc_x, float *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(eint n, double alpha_x, double const *x, eint inc_x, double *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x, std::complex<float> *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void zaxpy(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x, std::complex<double> *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void saxpby(eint n, float alpha_x, float const *x, eint inc_x, float b, float *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void daxpby(eint n, double alpha_x, double const *x, eint inc_x, double b, double *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void caxpby(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x, std::complex<float> b, std::complex<float> *y,
            eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void zaxpby(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x, std::complex<double> b,
            std::complex<double> *y, eint inc_y) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void sger(eint m, eint n, float alpha, float const *x, eint inc_x, float const *y, eint inc_y, float *a, eint lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void dger(eint m, eint n, double alpha, double const *x, eint inc_x, double const *y, eint inc_y, double *a, eint lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cger(eint m, eint n, std::complex<float> alpha, std::complex<float> const *x, eint inc_x, std::complex<float> const *y, eint inc_y,
          std::complex<float> *a, eint lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zger(eint m, eint n, std::complex<double> alpha, std::complex<double> const *x, eint inc_x, std::complex<double> const *y, eint inc_y,
          std::complex<double> *a, eint lda) {
    ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

auto sgetrf(eint m, eint n, float *a, eint lda, eint *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgetrf(m, n, a, lda, ipiv);
}

auto dgetrf(eint m, eint n, double *a, eint lda, eint *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgetrf(m, n, a, lda, ipiv);
}

auto cgetrf(eint m, eint n, std::complex<float> *a, eint lda, eint *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgetrf(m, n, a, lda, ipiv);
}

auto zgetrf(eint m, eint n, std::complex<double> *a, eint lda, eint *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgetrf(m, n, a, lda, ipiv);
}

auto sgetri(eint n, float *a, eint lda, eint const *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::sgetri(n, a, lda, ipiv);
}

auto dgetri(eint n, double *a, eint lda, eint const *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dgetri(n, a, lda, ipiv);
}

auto cgetri(eint n, std::complex<float> *a, eint lda, eint const *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::cgetri(n, a, lda, ipiv);
}

auto zgetri(eint n, std::complex<double> *a, eint lda, eint const *ipiv) -> eint {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zgetri(n, a, lda, ipiv);
}

auto slange(char norm_type, eint m, eint n, float const *A, eint lda, float *work) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::slange(norm_type, n, m, A, lda, work);
}

auto dlange(char norm_type, eint m, eint n, double const *A, eint lda, double *work) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dlange(norm_type, n, m, A, lda, work);
}

auto clange(char norm_type, eint m, eint n, std::complex<float> const *A, eint lda, float *work) -> float {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::clange(norm_type, n, m, A, lda, work);
}

auto zlange(char norm_type, eint m, eint n, std::complex<double> const *A, eint lda, double *work) -> double {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zlange(norm_type, n, m, A, lda, work);
}

void slassq(eint n, float const *x, eint incx, float *scale, float *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::slassq(n, x, incx, scale, sumsq);
}

void dlassq(eint n, double const *x, eint incx, double *scale, double *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::dlassq(n, x, incx, scale, sumsq);
}

void classq(eint n, std::complex<float> const *x, eint incx, float *scale, float *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::classq(n, x, incx, scale, sumsq);
}

void zlassq(eint n, std::complex<double> const *x, eint incx, double *scale, double *sumsq) {
    return ::einsums::backend::linear_algebra::EINSUMS_LINEAR_ALGEBRA_NAMESPACE::zlassq(n, x, incx, scale, sumsq);
}

auto sgesdd(char jobz, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt, eint ldvt) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("sgesdd not implemented.");
#endif
}

auto dgesdd(char jobz, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt, eint ldvt) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto cgesdd(char jobz, eint m, eint n, std::complex<float> *a, eint lda, float *s, std::complex<float> *u, eint ldu,
            std::complex<float> *vt, eint ldvt) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto zgesdd(char jobz, eint m, eint n, std::complex<double> *a, eint lda, double *s, std::complex<double> *u, eint ldu,
            std::complex<double> *vt, eint ldvt) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
#else
    throw std::runtime_error("dgesdd not implemented.");
#endif
}

auto sgesvd(char jobu, char jobvt, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt, eint ldvt, float *superb)
    -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
#elif defined(EINSUMS_HAVE_LAPACKE_H)
    return ::einsums::backend::linear_algebra::cblas::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
#else
    throw std::runtime_error("dgesvd not implemented.");
#endif
}

auto dgesvd(char jobu, char jobvt, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt, eint ldvt,
            double *superb) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
#elif defined(EINSUMS_HAVE_LAPACKE_H)
    return ::einsums::backend::linear_algebra::cblas::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
#else
    throw std::runtime_error("dgesvd not implemented.");
#endif
}

auto dgees(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs, eint ldvs) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
#else
    throw std::runtime_error("dgees not implemented.");
#endif
}

auto strsyl(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b, eint ldb, float *c, eint ldc,
            float *scale) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda, double const *b, eint ldb, double *c, eint ldc,
            double *scale) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a, eint lda, std::complex<float> const *b,
            eint ldb, std::complex<float> *c, eint ldc, float *scale) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<double> const *a, eint lda, std::complex<double> const *b,
            eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("dtrsyl not implemented.");
#endif
}

auto sgeqrf(eint m, eint n, float *a, eint lda, float *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::sgeqrf(m, n, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::sgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto dgeqrf(eint m, eint n, double *a, eint lda, double *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dgeqrf(m, n, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::dgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto cgeqrf(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::cgeqrf(m, n, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::cgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto zgeqrf(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::zgeqrf(m, n, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::zgeqrf(m, n, a, lda, tau);
#else
    throw std::runtime_error("dgeqrf not implemented.");
#endif
}

auto sorgqr(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::sorgqr(m, n, k, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::sorgqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto dorgqr(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::dorgqr(m, n, k, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::dorgqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, std::complex<float> const *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::cungqr(m, n, k, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::cungqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

auto zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, std::complex<double> const *tau) -> eint {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::zungqr(m, n, k, a, lda, tau);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::zungqr(m, n, k, a, lda, tau);
#else
    throw std::runtime_error("dorgqr not implemented.");
#endif
}

} // namespace detail
} // namespace einsums::blas