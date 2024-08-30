//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/blas/blas.hpp>
#include <einsums/blas/vendor.hpp>

namespace einsums::blas {

void initialize() {
    vendor::initialize();
}

void finalize() {
    vendor::finalize();
}

namespace detail {
void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, const float *a, int_t lda, const float *b, int_t ldb,
           float beta, float *c, int_t ldc) {
    vendor::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, const double *a, int_t lda, const double *b, int_t ldb,
           double beta, double *c, int_t ldc) {
    vendor::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, const std::complex<float> *a, int_t lda,
           const std::complex<float> *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    vendor::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    vendor::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemv(char transa, int_t m, int_t n, float alpha, const float *a, int_t lda, const float *x, int_t incx, float beta, float *y,
           int_t incy) {
    vendor::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(char transa, int_t m, int_t n, double alpha, const double *a, int_t lda, const double *x, int_t incx, double beta, double *y,
           int_t incy) {
    vendor::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *a, int_t lda, const std::complex<float> *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy) {
    vendor::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy) {
    vendor::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    return vendor::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    return vendor::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

auto sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
           int_t ldvr) -> int_t {
    return vendor::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr,
           int_t ldvr) -> int_t {
    return vendor::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w, std::complex<float> *vl, int_t ldvl,
           std::complex<float> *vr, int_t ldvr) -> int_t {
    return vendor::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w, std::complex<double> *vl,
           int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t {
    return vendor::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    return vendor::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    return vendor::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t {
    return vendor::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t {
    return vendor::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
           float *rwork) -> int_t {
    return vendor::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t {
    return vendor::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

void sscal(int_t n, float alpha, float *vec, int_t inc) {
    vendor::sscal(n, alpha, vec, inc);
}

void dscal(int_t n, double alpha, double *vec, int_t inc) {
    vendor::dscal(n, alpha, vec, inc);
}

void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    vendor::cscal(n, alpha, vec, inc);
}

void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    vendor::zscal(n, alpha, vec, inc);
}

void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc) {
    vendor::csscal(n, alpha, vec, inc);
}

void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc) {
    vendor::zdscal(n, alpha, vec, inc);
}

auto sdot(int_t n, const float *x, int_t incx, const float *y, int_t incy) -> float {
    return vendor::sdot(n, x, incx, y, incy);
}

auto ddot(int_t n, const double *x, int_t incx, const double *y, int_t incy) -> double {
    return vendor::ddot(n, x, incx, y, incy);
}

auto cdot(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float> {
    return vendor::cdot(n, x, incx, y, incy);
}

auto zdot(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double> {
    return vendor::zdot(n, x, incx, y, incy);
}

auto cdotc(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float> {
    return vendor::cdotc(n, x, incx, y, incy);
}

auto zdotc(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double> {
    return vendor::zdotc(n, x, incx, y, incy);
}

void saxpy(int_t n, float alpha_x, const float *x, int_t inc_x, float *y, int_t inc_y) {
    vendor::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void daxpy(int_t n, double alpha_x, const double *x, int_t inc_x, double *y, int_t inc_y) {
    vendor::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void caxpy(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    vendor::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void zaxpy(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    vendor::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

void saxpby(int_t n, float alpha_x, const float *x, int_t inc_x, float b, float *y, int_t inc_y) {
    vendor::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void daxpby(int_t n, double alpha_x, const double *x, int_t inc_x, double b, double *y, int_t inc_y) {
    vendor::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void caxpby(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> b, std::complex<float> *y,
            int_t inc_y) {
    vendor::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void zaxpby(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> b,
            std::complex<double> *y, int_t inc_y) {
    vendor::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

void sger(int_t m, int_t n, float alpha, const float *x, int_t inc_x, const float *y, int_t inc_y, float *a, int_t lda) {
    vendor::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void dger(int_t m, int_t n, double alpha, const double *x, int_t inc_x, const double *y, int_t inc_y, double *a, int_t lda) {
    vendor::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void cger(int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *x, int_t inc_x, const std::complex<float> *y, int_t inc_y,
          std::complex<float> *a, int_t lda) {
    vendor::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

void zger(int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *x, int_t inc_x, const std::complex<double> *y,
          int_t inc_y, std::complex<double> *a, int_t lda) {
    vendor::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

auto sgetrf(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    return vendor::sgetrf(m, n, a, lda, ipiv);
}

auto dgetrf(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    return vendor::dgetrf(m, n, a, lda, ipiv);
}

auto cgetrf(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    return vendor::cgetrf(m, n, a, lda, ipiv);
}

auto zgetrf(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    return vendor::zgetrf(m, n, a, lda, ipiv);
}

auto sgetri(int_t n, float *a, int_t lda, const int_t *ipiv) -> int_t {
    return vendor::sgetri(n, a, lda, ipiv);
}

auto dgetri(int_t n, double *a, int_t lda, const int_t *ipiv) -> int_t {
    return vendor::dgetri(n, a, lda, ipiv);
}

auto cgetri(int_t n, std::complex<float> *a, int_t lda, const int_t *ipiv) -> int_t {
    return vendor::cgetri(n, a, lda, ipiv);
}

auto zgetri(int_t n, std::complex<double> *a, int_t lda, const int_t *ipiv) -> int_t {
    return vendor::zgetri(n, a, lda, ipiv);
}

auto slange(char norm_type, int_t m, int_t n, const float *A, int_t lda, float *work) -> float {
    return vendor::slange(norm_type, m, n, A, lda, work);
}

auto dlange(char norm_type, int_t m, int_t n, const double *A, int_t lda, double *work) -> double {
    return vendor::dlange(norm_type, m, n, A, lda, work);
}

auto clange(char norm_type, int_t m, int_t n, const std::complex<float> *A, int_t lda, float *work) -> float {
    return vendor::clange(norm_type, m, n, A, lda, work);
}

auto zlange(char norm_type, int_t m, int_t n, const std::complex<double> *A, int_t lda, double *work) -> double {
    return vendor::zlange(norm_type, m, n, A, lda, work);
}

void slassq(int_t n, const float *x, int_t incx, float *scale, float *sumsq) {
    return vendor::slassq(n, x, incx, scale, sumsq);
}

void dlassq(int_t n, const double *x, int_t incx, double *scale, double *sumsq) {
    return vendor::dlassq(n, x, incx, scale, sumsq);
}

void classq(int_t n, const std::complex<float> *x, int_t incx, float *scale, float *sumsq) {
    return vendor::classq(n, x, incx, scale, sumsq);
}

void zlassq(int_t n, const std::complex<double> *x, int_t incx, double *scale, double *sumsq) {
    return vendor::zlassq(n, x, incx, scale, sumsq);
}

auto sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    return vendor::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt) -> int_t {
    return vendor::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt) -> int_t {
    return vendor::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt) -> int_t {
    return vendor::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

auto sgesvd(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
            float *superb) -> int_t {
    return vendor::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt,
            double *superb) -> int_t {
    return vendor::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

auto dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t {
    return vendor::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

auto strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const float *a, int_t lda, const float *b, int_t ldb, float *c, int_t ldc,
            float *scale) -> int_t {
    return vendor::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const double *a, int_t lda, const double *b, int_t ldb, double *c,
            int_t ldc, double *scale) -> int_t {
    return vendor::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

auto ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<float> *a, int_t lda, const std::complex<float> *b,
            int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("ctrsyl not implemented.");
#endif
}

auto ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<double> *a, int_t lda, const std::complex<double> *b,
            int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t {
#if defined(EINSUMS_HAVE_MKL_LAPACKE_H)
    return ::einsums::backend::linear_algebra::mkl::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#elif defined(EINSUMS_HAVE_LAPACKE)
    return ::einsums::backend::linear_algebra::cblas::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
#else
    throw std::runtime_error("ztrsyl not implemented.");
#endif
}

auto sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return vendor::sgeqrf(m, n, a, lda, tau);
}

auto dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return vendor::dgeqrf(m, n, a, lda, tau);
}

auto cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return vendor::cgeqrf(m, n, a, lda, tau);
}

auto zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return vendor::zgeqrf(m, n, a, lda, tau);
}

auto sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, const float *tau) -> int_t {
    return vendor::sorgqr(m, n, k, a, lda, tau);
}

auto dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, const double *tau) -> int_t {
    return vendor::dorgqr(m, n, k, a, lda, tau);
}

auto cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, const std::complex<float> *tau) -> int_t {
    return vendor::cungqr(m, n, k, a, lda, tau);
}

auto zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, const std::complex<double> *tau) -> int_t {
    return vendor::zungqr(m, n, k, a, lda, tau);
}

} // namespace detail
} // namespace einsums::blas