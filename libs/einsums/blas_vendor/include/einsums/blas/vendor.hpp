//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/blas/types.hpp>

#include <complex>

namespace einsums::blas::vendor {

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matrices of type double.
 */
void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, const float *a, int_t lda, const float *b, int_t ldb,
           float beta, float *c, int_t ldc);
void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, const double *a, int_t lda, const double *b, int_t ldb,
           double beta, double *c, int_t ldc);
void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, const std::complex<float> *a, int_t lda,
           const std::complex<float> *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, int_t m, int_t n, float alpha, const float *a, int_t lda, const float *x, int_t incx, float beta, float *y,
           int_t incy);
void dgemv(char transa, int_t m, int_t n, double alpha, const double *a, int_t lda, const double *x, int_t incx, double beta, double *y,
           int_t incy);
void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *a, int_t lda, const std::complex<float> *x,
           int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
           const std::complex<double> *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);

void saxpby(const int_t n, const float a, const float *x, const int_t incx, const float b, float *y, const int_t incy);
void daxpby(const int_t n, const double a, const double *x, const int_t incx, const double b, double *y, const int_t incy);
void caxpby(const int_t n, const std::complex<float> a, const std::complex<float> *x, const int_t incx, const std::complex<float> b,
            std::complex<float> *y, const int_t incy);
void zaxpby(const int_t n, const std::complex<double> a, const std::complex<double> *x, const int_t incx, const std::complex<double> b,
            std::complex<double> *y, const int_t incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;

/*!
 * Computes all eigenvalues and left and right eigenvectors of a general matrix.
 */
auto sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
           int_t ldvr) -> int_t;
auto dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr,
           int_t ldvr) -> int_t;
auto cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w, std::complex<float> *vl, int_t ldvl,
           std::complex<float> *vr, int_t ldvr) -> int_t;
auto zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w, std::complex<double> *vl,
           int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
           float *rwork) -> int_t;
auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t;

void sscal(int_t n, float alpha, float *vec, int_t inc);
void dscal(int_t n, double alpha, double *vec, int_t inc);
void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc);
void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc);
void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc);

auto sdot(int_t n, const float *x, int_t incx, const float *y, int_t incy) -> float;
auto ddot(int_t n, const double *x, int_t incx, const double *y, int_t incy) -> double;
auto cdot(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float>;
auto zdot(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double>;
auto cdotc(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y, int_t incy) -> std::complex<float>;
auto zdotc(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y, int_t incy) -> std::complex<double>;

void saxpy(int_t n, float alpha_x, const float *x, int_t inc_x, float *y, int_t inc_y);
void daxpy(int_t n, double alpha_x, const double *x, int_t inc_x, double *y, int_t inc_y);
void caxpy(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> *y, int_t inc_y);
void zaxpy(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> *y, int_t inc_y);

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 *    A := alpha*x*y' + A,
 * where:
 *   alpha is a scalar
 *   x is an m-element vector,
 *   y is an n-element vector,
 *   A is an m-by-n general matrix
 */
void sger(int_t m, int_t n, float alpha, const float *x, int_t inc_x, const float *y, int_t inc_y, float *a, int_t lda);
void dger(int_t m, int_t n, double alpha, const double *x, int_t inc_x, const double *y, int_t inc_y, double *a, int_t lda);
void cger(int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *x, int_t inc_x, const std::complex<float> *y, int_t inc_y,
          std::complex<float> *a, int_t lda);
void zger(int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *x, int_t inc_x, const std::complex<double> *y,
          int_t inc_y, std::complex<double> *a, int_t lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row int_terchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
auto sgetrf(int_t, int_t, float *, int_t, int_t *) -> int_t;
auto dgetrf(int_t, int_t, double *, int_t, int_t *) -> int_t;
auto cgetrf(int_t, int_t, std::complex<float> *, int_t, int_t *) -> int_t;
auto zgetrf(int_t, int_t, std::complex<double> *, int_t, int_t *) -> int_t;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(int_t, float *, int_t, const int_t *) -> int_t;
auto dgetri(int_t, double *, int_t, const int_t *) -> int_t;
auto cgetri(int_t, std::complex<float> *, int_t, const int_t *) -> int_t;
auto zgetri(int_t, std::complex<double> *, int_t, const int_t *) -> int_t;

auto slange(char norm_type, int_t m, int_t n, const float *A, int_t lda, float *work) -> float;
auto dlange(char norm_type, int_t m, int_t n, const double *A, int_t lda, double *work) -> double;
auto clange(char norm_type, int_t m, int_t n, const std::complex<float> *A, int_t lda, float *work) -> float;
auto zlange(char norm_type, int_t m, int_t n, const std::complex<double> *A, int_t lda, double *work) -> double;

void slassq(int_t n, const float *x, int_t incx, float *scale, float *sumsq);
void dlassq(int_t n, const double *x, int_t incx, double *scale, double *sumsq);
void classq(int_t n, const std::complex<float> *x, int_t incx, float *scale, float *sumsq);
void zlassq(int_t n, const std::complex<double> *x, int_t incx, double *scale, double *sumsq);

auto sgesvd(char, char, int_t, int_t, float *, int_t, float *, float *, int_t, float *, int_t, float *) -> int_t;
auto dgesvd(char, char, int_t, int_t, double *, int_t, double *, double *, int_t, double *, int_t, double *) -> int_t;

auto dgesdd(char, int_t, int_t, double *, int_t, double *, double *, int_t, double *, int_t) -> int_t;
auto sgesdd(char, int_t, int_t, float *, int_t, float *, float *, int_t, float *, int_t) -> int_t;
auto zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt) -> int_t;
auto cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt) -> int_t;

auto dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
auto sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;

auto dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const double *a, int_t lda, const double *b, int_t ldb, double *c,
            int_t ldc, double *scale) -> int_t;
auto strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const float *a, int_t lda, const float *b, int_t ldb, float *c, int_t ldc,
            float *scale) -> int_t;

auto sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, const float *tau) -> int_t;
auto dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, const double *tau) -> int_t;
auto cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, const std::complex<float> *tau) -> int_t;
auto zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, const std::complex<double> *tau) -> int_t;

auto dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;

} // namespace einsums::blas::vendor