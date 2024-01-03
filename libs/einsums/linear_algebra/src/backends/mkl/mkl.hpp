//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/linear_algebra/Types.hpp"
#include "einsums/preprocessor/Namespace.hpp"

#include <complex>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::backend::linear_algebra::mkl)

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
#define mkl_def_gemm(x, type)                                                                                                              \
    void x##gemm(const char transa, const char transb, eint m, eint n, eint k, type alpha, const type *a, eint lda, const type *b,         \
                 eint ldb, type beta, type *c, eint ldc)

mkl_def_gemm(s, float);
mkl_def_gemm(d, double);
mkl_def_gemm(c, std::complex<float>);
mkl_def_gemm(z, std::complex<double>);

#undef gemm

/*!
 * Computes groups of matrix-matrix products with general matrices.
 */
#define mkl_def_gemm_batch_strided(x, type)                                                                                                \
    void x##gemm_batch_strided(const char transa, const char transb, eint m, eint n, eint k, type alpha, const type *a, eint lda,          \
                               eint stridea, const type *b, eint ldb, eint strideb, type beta, type *c, eint ldc, eint stridec,            \
                               eint batch_size)

mkl_def_gemm_batch_strided(s, float);
mkl_def_gemm_batch_strided(d, double);
mkl_def_gemm_batch_strided(c, std::complex<float>);
mkl_def_gemm_batch_strided(z, std::complex<double>);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char const transa, const eint m, const eint n, float const alpha, float const *a, const eint lda, float const *x,
           const eint incx, float const beta, float *y, const eint incy);
void dgemv(char const transa, const eint m, const eint n, double const alpha, double const *a, const eint lda, double const *x,
           const eint incx, double beta, double *y, const eint incy);
void cgemv(char const transa, const eint m, const eint n, const std::complex<float> alpha, std::complex<float> const *a, const eint lda,
           std::complex<float> const *x, const eint incx, const std::complex<float> beta, std::complex<float> *y, const eint incy);
void zgemv(char const transa, const eint m, const eint n, const std::complex<double> alpha, std::complex<double> const *a, const eint lda,
           std::complex<double> const *x, const eint incx, const std::complex<double> beta, std::complex<double> *y, const eint incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char const job, char const uplo, const eint n, float *a, const eint lda, float *w, float *work, const eint lwork) -> eint;
auto dsyev(char const job, char const uplo, const eint n, double *a, const eint lda, double *w, double *work, const eint lwork) -> eint;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char const job, char const uplo, const eint n, std::complex<float> *a, const eint lda, float *w, std::complex<float> *work,
           const eint lwork, float *rwork) -> eint;
auto zheev(char const job, char const uplo, const eint n, std::complex<double> *a, const eint lda, double *w, std::complex<double> *work,
           const eint lwork, double *rwork) -> eint;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(const eint n, const eint nrhs, float *a, const eint lda, eint *ipiv, float *b, const eint ldb) -> eint;
auto dgesv(const eint n, const eint nrhs, double *a, const eint lda, eint *ipiv, double *b, const eint ldb) -> eint;
auto cgesv(const eint n, const eint nrhs, std::complex<float> *a, const eint lda, eint *ipiv, std::complex<float> *b, const eint ldb)
    -> eint;
auto zgesv(const eint n, const eint nrhs, std::complex<double> *a, const eint lda, eint *ipiv, std::complex<double> *b, const eint ldb)
    -> eint;

void sscal(const eint n, float const alpha, float *vec, const eint inc);
void dscal(const eint n, double const alpha, double *vec, const eint inc);
void cscal(const eint n, const std::complex<float> alpha, std::complex<float> *vec, const eint inc);
void zscal(const eint n, const std::complex<double> alpha, std::complex<double> *vec, const eint inc);
void csscal(const eint n, float const alpha, std::complex<float> *vec, const eint inc);
void zdscal(const eint n, double const alpha, std::complex<double> *vec, const eint inc);

auto sdot(const eint n, float const *x, const eint incx, float const *y, const eint incy) -> float;
auto ddot(const eint n, double const *x, const eint incx, double const *y, const eint incy) -> double;
auto cdot(const eint n, std::complex<float> const *x, const eint incx, std::complex<float> const *y, const eint incy)
    -> std::complex<float>;
auto zdot(const eint n, std::complex<double> const *x, const eint incx, std::complex<double> const *y, const eint incy)
    -> std::complex<double>;

void saxpy(const eint n, float const alpha_x, float const *x, const eint inc_x, float *y, const eint inc_y);
void daxpy(const eint n, double const alpha_x, double const *x, const eint inc_x, double *y, const eint inc_y);
void caxpy(const eint n, const std::complex<float> alpha_x, std::complex<float> const *x, const eint inc_x, std::complex<float> *y,
           const eint inc_y);
void zaxpy(const eint n, const std::complex<double> alpha_x, std::complex<double> const *x, const eint inc_x, std::complex<double> *y,
           const eint inc_y);

void saxpby(const eint n, float const a, float const *x, const eint incx, float const b, float *y, const eint incy);
void daxpby(const eint n, double const a, double const *x, const eint incx, double const b, double *y, const eint incy);
void caxpby(const eint n, const std::complex<float> a, std::complex<float> const *x, const eint incx, const std::complex<float> b,
            std::complex<float> *y, const eint incy);
void zaxpby(const eint n, const std::complex<double> a, std::complex<double> const *x, const eint incx, const std::complex<double> b,
            std::complex<double> *y, const eint incy);

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
void sger(const eint m, const eint n, float const alpha, float const *x, const eint inc_x, float const *y, const eint inc_y, float *a,
          const eint lda);
void dger(const eint m, const eint n, double const alpha, double const *x, const eint inc_x, double const *y, const eint inc_y, double *a,
          const eint lda);
void cger(const eint m, const eint n, const std::complex<float> alpha, std::complex<float> const *x, const eint inc_x,
          std::complex<float> const *y, const eint inc_y, std::complex<float> *a, const eint lda);
void zger(const eint m, const eint n, const std::complex<double> alpha, std::complex<double> const *x, const eint inc_x,
          std::complex<double> const *y, const eint inc_y, std::complex<double> *a, const eint lda);

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
auto sgetrf(const eint, const eint, float *, const eint, eint *) -> eint;
auto dgetrf(const eint, const eint, double *, const eint, eint *) -> eint;
auto cgetrf(const eint, const eint, std::complex<float> *, const eint, eint *) -> eint;
auto zgetrf(const eint, const eint, std::complex<double> *, const eint, eint *) -> eint;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(const eint, float *, const eint, eint const *) -> eint;
auto dgetri(const eint, double *, const eint, eint const *) -> eint;
auto cgetri(const eint, std::complex<float> *, const eint, eint const *) -> eint;
auto zgetri(const eint, std::complex<double> *, const eint, eint const *) -> eint;

auto slange(char const norm_type, const eint m, const eint n, float const *A, const eint lda, float *work) -> float;
auto dlange(char const norm_type, const eint m, const eint n, double const *A, const eint lda, double *work) -> double;
auto clange(char const norm_type, const eint m, const eint n, std::complex<float> const *A, const eint lda, float *work) -> float;
auto zlange(char const norm_type, const eint m, const eint n, std::complex<double> const *A, const eint lda, double *work) -> double;

void slassq(const eint n, float const *x, const eint incx, float *scale, float *sumsq);
void dlassq(const eint n, double const *x, const eint incx, double *scale, double *sumsq);
void classq(const eint n, std::complex<float> const *x, const eint incx, float *scale, float *sumsq);
void zlassq(const eint n, std::complex<double> const *x, const eint incx, double *scale, double *sumsq);

auto sgesdd(char, eint, eint, float *, eint, float *, float *, eint, float *, eint) -> eint;
auto dgesdd(char, eint, eint, double *, eint, double *, double *, eint, double *, eint) -> eint;
auto cgesdd(char, eint, eint, std::complex<float> *, eint, float *, std::complex<float> *, eint, std::complex<float> *, eint) -> eint;
auto zgesdd(char, eint, eint, std::complex<double> *, eint, double *, std::complex<double> *, eint, std::complex<double> *, eint) -> eint;

auto sgesvd(char, char, eint, eint, float *, eint, float *, float *, eint, float *, eint, float *) -> eint;
auto dgesvd(char, char, eint, eint, double *, eint, double *, double *, eint, double *, eint, double *) -> eint;

auto sgees(char jobvs, eint n, float *a, eint lda, eint *sdim, float *wr, float *wi, float *vs, eint ldvs) -> eint;
auto dgees(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs, eint ldvs) -> eint;

auto strsyl(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b, eint ldb, float *c, eint ldc,
            float *scale) -> eint;
auto dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda, double const *b, eint ldb, double *c, eint ldc,
            double *scale) -> eint;
auto ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a, eint lda, std::complex<float> const *b,
            eint ldb, std::complex<float> *c, eint ldc, float *scale) -> eint;
auto ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<double> const *a, eint lda, std::complex<double> const *b,
            eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint;

auto sgeqrf(eint m, eint n, float *a, eint lda, float *tau) -> eint;
auto dgeqrf(eint m, eint n, double *a, eint lda, double *tau) -> eint;
auto cgeqrf(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau) -> eint;
auto zgeqrf(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau) -> eint;

auto sorgqr(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint;
auto dorgqr(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint;
auto cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, std::complex<float> const *tau) -> eint;
auto zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, std::complex<double> const *tau) -> eint;

END_EINSUMS_NAMESPACE_HPP(einsums::backend::linear_algebra::mkl)