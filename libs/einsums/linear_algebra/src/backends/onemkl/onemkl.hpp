//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums::backend::linear_algebra::onemkl {

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);

/*!
 * Performs matrix vector multiplication.
 */
void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) -> int;

void dscal(int n, double alpha, double *vec, int inc);

auto ddot(int n, const double *x, int incx, const double *y, int incy) -> double;

void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);

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
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);

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
auto dgetrf(int, int, double *, int, int *) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto dgetri(int, double *, int, const int *, double *, int) -> int;

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;

auto dgesdd(char, int, int, double *, int, double *, double *, int, double *, int, double *, int, int *) -> int;

} // namespace einsums::backend::linear_algebra::onemkl
