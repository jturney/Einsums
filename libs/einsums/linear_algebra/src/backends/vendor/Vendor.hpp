//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <complex>

namespace einsums::backend::linear_algebra::vendor {

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb, float beta,
           float *c, int ldc);
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
           const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
           const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy);
void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);
void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy);
void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy);

void saxpby(const int n, const float a, const float *x, const int incx, const float b, float *y, const int incy);
void daxpby(const int n, const double a, const double *x, const int incx, const double b, double *y, const int incy);
void caxpby(const int n, const std::complex<float> a, const std::complex<float> *x, const int incx, const std::complex<float> b,
            std::complex<float> *y, const int incy);
void zaxpby(const int n, const std::complex<double> a, const std::complex<double> *x, const int incx, const std::complex<double> b,
            std::complex<double> *y, const int incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *rwork)
    -> int;
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *rwork)
    -> int;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) -> int;
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;
auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) -> int;
auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) -> int;

void sscal(int n, float alpha, float *vec, int inc);
void dscal(int n, double alpha, double *vec, int inc);
void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc);
void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc);
void csscal(int n, float alpha, std::complex<float> *vec, int inc);
void zdscal(int n, double alpha, std::complex<double> *vec, int inc);

auto sdot(int n, const float *x, int incx, const float *y, int incy) -> float;
auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double;
auto cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) -> std::complex<float>;
auto zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) -> std::complex<double>;

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y);
void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);
void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y);
void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y);

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
void sger(int m, int n, float alpha, const float *x, int inc_x, const float *y, int inc_y, float *a, int lda);
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);
void cger(int m, int n, std::complex<float> alpha, const std::complex<float> *x, int inc_x, const std::complex<float> *y, int inc_y,
          std::complex<float> *a, int lda);
void zger(int m, int n, std::complex<double> alpha, const std::complex<double> *x, int inc_x, const std::complex<double> *y, int inc_y,
          std::complex<double> *a, int lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
auto sgetrf(int, int, float *, int, int *) -> int;
auto dgetrf(int, int, double *, int, int *) -> int;
auto cgetrf(int, int, std::complex<float> *, int, int *) -> int;
auto zgetrf(int, int, std::complex<double> *, int, int *) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(int, float *, int, const int *) -> int;
auto dgetri(int, double *, int, const int *) -> int;
auto cgetri(int, std::complex<float> *, int, const int *) -> int;
auto zgetri(int, std::complex<double> *, int, const int *) -> int;

auto slange(char norm_type, int m, int n, const float *A, int lda, float *work) -> float;
auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;
auto clange(char norm_type, int m, int n, const std::complex<float> *A, int lda, float *work) -> float;
auto zlange(char norm_type, int m, int n, const std::complex<double> *A, int lda, double *work) -> double;

void slassq(int n, const float *x, int incx, float *scale, float *sumsq);
void dlassq(int n, const double *x, int incx, double *scale, double *sumsq);
void classq(int n, const std::complex<float> *x, int incx, float *scale, float *sumsq);
void zlassq(int n, const std::complex<double> *x, int incx, double *scale, double *sumsq);

auto dgesdd(char, int, int, double *, int, double *, double *, int, double *, int, double *, int, int *) -> int;

} // namespace einsums::backend::linear_algebra::vendor
