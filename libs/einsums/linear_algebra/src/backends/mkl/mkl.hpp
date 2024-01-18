//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/preprocessor/namespace.hpp>

#include <einsums/linear_algebra/Types.hpp>

#include <complex>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::backend::linear_algebra::mkl)

void initialize();
void finalize();

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
#define mkl_def_gemm(x, type)                                                                                          \
    void x##gemm(const char transa, const char transb, eint m, eint n, eint k, type alpha, const type *a, eint lda,    \
                 const type *b, eint ldb, type beta, type *c, eint ldc)

mkl_def_gemm(s, float);
mkl_def_gemm(d, double);
mkl_def_gemm(c, std::complex<float>);
mkl_def_gemm(z, std::complex<double>);

#undef gemm

/*!
 * Computes groups of matrix-matrix products with general matrices.
 */
#define mkl_def_gemm_batch_strided(x, type)                                                                            \
    void x##gemm_batch_strided(const char transa, const char transb, eint m, eint n, eint k, type alpha,               \
                               const type *a, eint lda, eint stridea, const type *b, eint ldb, eint strideb,           \
                               type beta, type *c, eint ldc, eint stridec, eint batch_size)

mkl_def_gemm_batch_strided(s, float);
mkl_def_gemm_batch_strided(d, double);
mkl_def_gemm_batch_strided(c, std::complex<float>);
mkl_def_gemm_batch_strided(z, std::complex<double>);

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char const transa, eint const m, eint const n, float const alpha, float const *a, eint const lda,
           float const *x, eint const incx, float const beta, float *y, eint const incy);
void dgemv(char const transa, eint const m, eint const n, double const alpha, double const *a, eint const lda,
           double const *x, eint const incx, double beta, double *y, eint const incy);
void cgemv(char const transa, eint const m, eint const n, std::complex<float> const alpha, std::complex<float> const *a,
           eint const lda, std::complex<float> const *x, eint const incx, std::complex<float> const beta,
           std::complex<float> *y, eint const incy);
void zgemv(char const transa, eint const m, eint const n, std::complex<double> const alpha,
           std::complex<double> const *a, eint const lda, std::complex<double> const *x, eint const incx,
           std::complex<double> const beta, std::complex<double> *y, eint const incy);

/*!
 * Performs symmetric matrix diagonalization.
 */
auto ssyev(char const job, char const uplo, eint const n, float *a, eint const lda, float *w, float *work,
           eint const lwork) -> eint;
auto dsyev(char const job, char const uplo, eint const n, double *a, eint const lda, double *w, double *work,
           eint const lwork) -> eint;

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
auto cheev(char const job, char const uplo, eint const n, std::complex<float> *a, eint const lda, float *w,
           std::complex<float> *work, eint const lwork, float *rwork) -> eint;
auto zheev(char const job, char const uplo, eint const n, std::complex<double> *a, eint const lda, double *w,
           std::complex<double> *work, eint const lwork, double *rwork) -> eint;

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
auto sgesv(eint const n, eint const nrhs, float *a, eint const lda, eint *ipiv, float *b, eint const ldb) -> eint;
auto dgesv(eint const n, eint const nrhs, double *a, eint const lda, eint *ipiv, double *b, eint const ldb) -> eint;
auto cgesv(eint const n, eint const nrhs, std::complex<float> *a, eint const lda, eint *ipiv, std::complex<float> *b,
           eint const ldb) -> eint;
auto zgesv(eint const n, eint const nrhs, std::complex<double> *a, eint const lda, eint *ipiv, std::complex<double> *b,
           eint const ldb) -> eint;

void sscal(eint const n, float const alpha, float *vec, eint const inc);
void dscal(eint const n, double const alpha, double *vec, eint const inc);
void cscal(eint const n, std::complex<float> const alpha, std::complex<float> *vec, eint const inc);
void zscal(eint const n, std::complex<double> const alpha, std::complex<double> *vec, eint const inc);
void csscal(eint const n, float const alpha, std::complex<float> *vec, eint const inc);
void zdscal(eint const n, double const alpha, std::complex<double> *vec, eint const inc);

auto sdot(eint const n, float const *x, eint const incx, float const *y, eint const incy) -> float;
auto ddot(eint const n, double const *x, eint const incx, double const *y, eint const incy) -> double;
auto cdot(eint const n, std::complex<float> const *x, eint const incx, std::complex<float> const *y, eint const incy)
    -> std::complex<float>;
auto zdot(eint const n, std::complex<double> const *x, eint const incx, std::complex<double> const *y, eint const incy)
    -> std::complex<double>;

void saxpy(eint const n, float const alpha_x, float const *x, eint const inc_x, float *y, eint const inc_y);
void daxpy(eint const n, double const alpha_x, double const *x, eint const inc_x, double *y, eint const inc_y);
void caxpy(eint const n, std::complex<float> const alpha_x, std::complex<float> const *x, eint const inc_x,
           std::complex<float> *y, eint const inc_y);
void zaxpy(eint const n, std::complex<double> const alpha_x, std::complex<double> const *x, eint const inc_x,
           std::complex<double> *y, eint const inc_y);

void saxpby(eint const n, float const a, float const *x, eint const incx, float const b, float *y, eint const incy);
void daxpby(eint const n, double const a, double const *x, eint const incx, double const b, double *y, eint const incy);
void caxpby(eint const n, std::complex<float> const a, std::complex<float> const *x, eint const incx,
            std::complex<float> const b, std::complex<float> *y, eint const incy);
void zaxpby(eint const n, std::complex<double> const a, std::complex<double> const *x, eint const incx,
            std::complex<double> const b, std::complex<double> *y, eint const incy);

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
void sger(eint const m, eint const n, float const alpha, float const *x, eint const inc_x, float const *y,
          eint const inc_y, float *a, eint const lda);
void dger(eint const m, eint const n, double const alpha, double const *x, eint const inc_x, double const *y,
          eint const inc_y, double *a, eint const lda);
void cger(eint const m, eint const n, std::complex<float> const alpha, std::complex<float> const *x, eint const inc_x,
          std::complex<float> const *y, eint const inc_y, std::complex<float> *a, eint const lda);
void zger(eint const m, eint const n, std::complex<double> const alpha, std::complex<double> const *x, eint const inc_x,
          std::complex<double> const *y, eint const inc_y, std::complex<double> *a, eint const lda);

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
auto sgetrf(eint const, eint const, float *, eint const, eint *) -> eint;
auto dgetrf(eint const, eint const, double *, eint const, eint *) -> eint;
auto cgetrf(eint const, eint const, std::complex<float> *, eint const, eint *) -> eint;
auto zgetrf(eint const, eint const, std::complex<double> *, eint const, eint *) -> eint;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto sgetri(eint const, float *, eint const, eint const *) -> eint;
auto dgetri(eint const, double *, eint const, eint const *) -> eint;
auto cgetri(eint const, std::complex<float> *, eint const, eint const *) -> eint;
auto zgetri(eint const, std::complex<double> *, eint const, eint const *) -> eint;

auto slange(char const norm_type, eint const m, eint const n, float const *A, eint const lda, float *work) -> float;
auto dlange(char const norm_type, eint const m, eint const n, double const *A, eint const lda, double *work) -> double;
auto clange(char const norm_type, eint const m, eint const n, std::complex<float> const *A, eint const lda, float *work)
    -> float;
auto zlange(char const norm_type, eint const m, eint const n, std::complex<double> const *A, eint const lda,
            double *work) -> double;

void slassq(eint const n, float const *x, eint const incx, float *scale, float *sumsq);
void dlassq(eint const n, double const *x, eint const incx, double *scale, double *sumsq);
void classq(eint const n, std::complex<float> const *x, eint const incx, float *scale, float *sumsq);
void zlassq(eint const n, std::complex<double> const *x, eint const incx, double *scale, double *sumsq);

auto sgesdd(char, eint, eint, float *, eint, float *, float *, eint, float *, eint) -> eint;
auto dgesdd(char, eint, eint, double *, eint, double *, double *, eint, double *, eint) -> eint;
auto cgesdd(char, eint, eint, std::complex<float> *, eint, float *, std::complex<float> *, eint, std::complex<float> *,
            eint) -> eint;
auto zgesdd(char, eint, eint, std::complex<double> *, eint, double *, std::complex<double> *, eint,
            std::complex<double> *, eint) -> eint;

auto sgesvd(char, char, eint, eint, float *, eint, float *, float *, eint, float *, eint, float *) -> eint;
auto dgesvd(char, char, eint, eint, double *, eint, double *, double *, eint, double *, eint, double *) -> eint;

auto sgees(char jobvs, eint n, float *a, eint lda, eint *sdim, float *wr, float *wi, float *vs, eint ldvs) -> eint;
auto dgees(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs, eint ldvs) -> eint;

auto strsyl(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b, eint ldb,
            float *c, eint ldc, float *scale) -> eint;
auto dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda, double const *b, eint ldb,
            double *c, eint ldc, double *scale) -> eint;
auto ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a, eint lda,
            std::complex<float> const *b, eint ldb, std::complex<float> *c, eint ldc, float *scale) -> eint;
auto ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<double> const *a, eint lda,
            std::complex<double> const *b, eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint;

auto sgeqrf(eint m, eint n, float *a, eint lda, float *tau) -> eint;
auto dgeqrf(eint m, eint n, double *a, eint lda, double *tau) -> eint;
auto cgeqrf(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau) -> eint;
auto zgeqrf(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau) -> eint;

auto sorgqr(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint;
auto dorgqr(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint;
auto cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, std::complex<float> const *tau) -> eint;
auto zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, std::complex<double> const *tau) -> eint;

END_EINSUMS_NAMESPACE_HPP(einsums::backend::linear_algebra::mkl)