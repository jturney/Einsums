//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <complex>

extern "C" {
/**
 * Internal kernel for performing the direct product. The direct product function is not in BLAS, so we have to write our own.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The first input vector.
 * @param[in] y The second input vector.
 * @param[out] z The output vector.
 *
 * @versionadded{2.0.0}
 *
 * @versionchangeddesc{2.0.0}
 *  Added AVX2 vectorized kernels and default unvectorized kernels.
 * @endversion
 */
extern EINSUMS_EXPORT void sdirprod_kernel(size_t n, float alpha, float const *x, float const *y, float *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void ddirprod_kernel(size_t n, double alpha, double const *x, double const *y, double *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void cdirprod_kernel(size_t n, std::complex<float> alpha, std::complex<float> const *x, std::complex<float> const *y,
                                           std::complex<float> *z);

/// @copydoc sdirprod_kernel
extern EINSUMS_EXPORT void zdirprod_kernel(size_t n, std::complex<double> alpha, std::complex<double> const *x,
                                           std::complex<double> const *y, std::complex<double> *z);
}

namespace einsums::blas::vendor {

/**
 * @brief Initializes the BLAS library.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize();

/**
 * @brief Tears down the BLAS library.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void finalize();

/**
 * Performs matrix multiplication for general square matrices.
 *
 * @param[in] transa Whether to transpose the left matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] transb Whether to transpose the right matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] m The number of rows in the output matrix.
 * @param[in] n The number of columns in the output matrix.
 * @param[in] k The number of columns in the first input matrix if @p transa is 'n'. May be different.
 * @param[in] alpha The scale factor for the product.
 * @param[in] a The left matrix.
 * @param[in] lda The leading dimension for the left matrix.
 * @param[in] b The right matrix.
 * @param[in] ldb The leading dimension for the right matrix.
 * @param[in] beta The scale factor for the output matrix.
 * @param[inout] c The output matrix.
 * @param[in] ldc The leading dimension for the output matrix.
 *
 * @throws std::invalid_argument If @p transa or @p transb are not one of the valid options.
 * @throws std::domain_error If any of the integer arguments have invalid values.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                          int_t ldb, float beta, float *c, int_t ldc);
/// @copydoc sgemm
EINSUMS_EXPORT void dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                          int_t ldb, double beta, double *c, int_t ldc);
/// @copydoc sgemm
EINSUMS_EXPORT void cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                          int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc sgemm
EINSUMS_EXPORT void zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                          int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c,
                          int_t ldc);

/**
 * Performs a batch of independent general matrix-matrix multiplications.
 *
 * All batches share the same transa, transb, m, n, k, alpha, beta, lda, ldb, ldc.
 * Only the data pointers differ, passed as arrays of length batch_count.
 *
 * @param[in] transa Whether to transpose A. Can be 'N', 'T', or 'C'.
 * @param[in] transb Whether to transpose B. Can be 'N', 'T', or 'C'.
 * @param[in] m Number of rows of C and op(A).
 * @param[in] n Number of columns of C and op(B).
 * @param[in] k Number of columns of op(A) and rows of op(B).
 * @param[in] alpha Scalar multiplier for A*B.
 * @param[in] a_array Array of batch_count pointers to A matrices.
 * @param[in] lda Leading dimension of each A.
 * @param[in] b_array Array of batch_count pointers to B matrices.
 * @param[in] ldb Leading dimension of each B.
 * @param[in] beta Scalar multiplier for C.
 * @param[in,out] c_array Array of batch_count pointers to C matrices.
 * @param[in] ldc Leading dimension of each C.
 * @param[in] batch_count Number of independent GEMMs to perform.
 */
EINSUMS_EXPORT void sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda,
                                float const **b_array, int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count);
/// @copydoc sgemm_batch
EINSUMS_EXPORT void dgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                                double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count);
/// @copydoc sgemm_batch
EINSUMS_EXPORT void cgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                std::complex<float> const **a_array, int_t lda, std::complex<float> const **b_array, int_t ldb,
                                std::complex<float> beta, std::complex<float> **c_array, int_t ldc, int_t batch_count);
/// @copydoc sgemm_batch
EINSUMS_EXPORT void zgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                std::complex<double> const **a_array, int_t lda, std::complex<double> const **b_array, int_t ldb,
                                std::complex<double> beta, std::complex<double> **c_array, int_t ldc, int_t batch_count);

/**
 * Performs matrix vector multiplication.
 *
 * @param[in] transa Whether to transpose the matrix. Case insensitive. Can be 'n', 'c', or 't'.
 * @param[in] m The number of entries in the input vector.
 * @param[in] n The number of entries in the output vector.
 * @param[in] alpha The scale factor for the multiplication.
 * @param[in] a The matrix.
 * @param[in] lda The leading dimension for the matrix.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the input vector. If it is negative, then the multiplication proceeds in reverse.
 * @param[in] beta The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] incy The skip value for the output vector. If it is negative, then the multiplication proceeds in reverse.
 *
 * @throws std::invalid_argument If @p transa or @p transb are not one of the valid options.
 * @throws std::domain_error If any of the integer arguments have invalid values.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                          float *y, int_t incy);
/// @copydoc sgemv
EINSUMS_EXPORT void dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                          double *y, int_t incy);
/// @copydoc sgemv
EINSUMS_EXPORT void cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
/// @copydoc sgemv
EINSUMS_EXPORT void zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);

/**
 * Scales and adds two vectors.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] a The scale factor for the first vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the input vector. If it is negative, then the vector is traversed backwards. If it is zero, then the
 * result will be broadcast to the output vector.
 * @param[in] b The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] incy The skip value for the output vector. If it is negative, then the vector is traversed backwards.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void saxpby(int_t const n, float const a, float const *x, int_t const incx, float const b, float *y, int_t const incy);
/// @copydoc saxpby
EINSUMS_EXPORT void daxpby(int_t const n, double const a, double const *x, int_t const incx, double const b, double *y, int_t const incy);
/// @copydoc saxpby
EINSUMS_EXPORT void caxpby(int_t const n, std::complex<float> const a, std::complex<float> const *x, int_t const incx,
                           std::complex<float> const b, std::complex<float> *y, int_t const incy);
/// @copydoc saxpby
EINSUMS_EXPORT void zaxpby(int_t const n, std::complex<double> const a, std::complex<double> const *x, int_t const incx,
                           std::complex<double> const b, std::complex<double> *y, int_t const incy);

/**
 * Performs symmetric matrix diagonalization.
 *
 * @param[in] job Whether to compute the eigenvectors. Case insensitive. Can be either 'n' or 'v'.
 * @param[in] uplo Whether the data is stored in the upper or lower triangle of the input. Case insensitive. Can be either 'u' or 'l'.
 * @param[in] n The number of rows and columns of the matrix.
 * @param[inout] a The matrix to factor. On exit, its data is overwritten. If the vectors are computed, they are placed in the columns of
 * the matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] w The output for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then the optimal work array size is placed in the first element of this array.
 * @param[in] lwork The size of the work array. If it is -1, then a workspace query is performed, and the optimal workspace size is put in
 * the first element of @p work.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm did not converge. If negative, then one of the parameters was given a
 * bad value. The absolute value indicates which parameter it was.
 *
 * @versionadded{1.0.0}
 */
auto ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
/// @copydoc ssyev
auto dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;

/**
 * Performs symmetric tridiagonal matrix diagonalization.
 *
 * @param[in] n The number of elements on the diagonal.
 * @param[inout] d The diagonal. On exit, it contains the eigenvalues.
 * @param[inout] e The subdiagonal.
 *
 * @return 0 on successs. If positive, then the eigenvalue algorithm failed to converge. If negative, then
 * one of the inputs had a bad value. The absolute value indicates which parameter it was.
 *
 * @versionadded{2.0.0}
 */
auto ssterf(int_t n, float *d, float *e) -> int_t;
/// @copydoc ssterf
auto dsterf(int_t n, double *d, double *e) -> int_t;

/**
 * Compute the eigendecomposition of a general matrix.
 *
 * @param[in] jobvl Indicates whether to compute the left eigenvectors. Case insensitive. Can be 'n' or 'v'.
 * @param[in] jobvr Indicates whetherh to compute the right eigenvectors. Case insensitive. Can be 'n' or 'v'.
 * @param[in] n The number of rows and columns in the matrix.
 * @param[inout] a The matrix to diagonalize. On exit, it will be overwritten with data used in the computation.
 * @param[in] lda The leading dimenson of the matrix.
 * @param[out] w The output for the eigenvalues.
 * @param[out] vl The left eigenvectors. For real inputs, this will have a special storage format.
 * @param[in] ldvl The leading dimension of the left eigenvectors. Must be at least 1, even if not referenced.
 * @param[out] vr The right eigenvectors. For real inputs, this will have a special storage format.
 * @param[in] ldvr The leading dimension of the right eigenvectors. Must be at least 1, even if not referenced.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm failed to converge. If negative, then one of the inputs had an invalid
 * value. The absolute value of the return indicates which parameter was invalid.
 *
 * @versionadded{1.0.0}
 */
auto sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr, int_t ldvr)
    -> int_t;
/// @copydoc sgeev
auto dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr, int_t ldvr)
    -> int_t;
/// @copydoc sgeev
auto cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w, std::complex<float> *vl, int_t ldvl,
           std::complex<float> *vr, int_t ldvr) -> int_t;
/// @copydoc sgeev
auto zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w, std::complex<double> *vl,
           int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t;

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 *
 * @param[in] job Whether to compute the eigenvectors. Case insensitive. Can be 'v' or 'n'.
 * @param[in] uplo Indicates how the data is stored. Case insensitive. Can be 'u' if the data is stored in the upper triangle, or 'l' if the
 * data is stored in the lower triangle.
 * @param[in] n The number of rows and columns in the matrix.
 * @param[inout] a The input matrix. On exit, it will be overwritten. If the eigenvectors are computed, then they will be stored in the
 * columns of this paramter.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] w The output for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then the optimal work array size will be placed in the first element of the work
 * array.
 * @param[in] lwork The size of the work array. If it is -1, then a workspace query is performed and nothing else. The optimal workspace
 * size will then be placed in the first element of the work array.
 * @param[inout] rwork A work array for real data. Must be at least @f$3n - 2@f$ elements.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm failed to converge. If negative, then one of the inputs had an invalid
 * value. The absolute value of the return indicates which parameter was invalid.
 *
 * @versionadded{1.0.0}
 */
auto cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork, float *rwork)
    -> int_t;
auto zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work, int_t lwork,
           double *rwork) -> int_t;

/**
 * Computes the solution to system of linear equations @f$\mathbf{Ax} = \mathbf{B}@f$ for general
 * matrices.
 *
 * @param[in] n The number of rows and columns of the coefficient matrix.
 * @param[in] nrhs The number of columns in the right-hand side matrix.
 * @param[inout] a The coefficient matrix.
 * @param[in] lda The leading dimension of the coefficient matrix.
 * @param[out] ipiv A list for pivots used during LU decomposition.
 * @param[inout] b The right-hand side matrix.
 * @param[in] ldb The leading dimension of the right-hand side matrix.
 *
 * @return 0 on success. If positive, the coefficient matrix was singular. If negative, then there was an invalid parameter.
 * The absolute value indicates which parameter was invalid.
 * @versionadded{1.0.0}
 */
auto sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
/// @copydoc sgesv
auto dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
/// @copydoc sgesv
auto cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
/// @copydoc sgesv
auto zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb) -> int_t;

/**
 * Scales a vector by a scalar. Supports real scalars by real vectors, complex scalars by complex vectors,
 * and real scalars by complex vectors.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The skip value for the vector.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void sscal(int_t n, float alpha, float *vec, int_t inc);
/// @copydoc sscal
EINSUMS_EXPORT void dscal(int_t n, double alpha, double *vec, int_t inc);
/// @copydoc sscal
EINSUMS_EXPORT void cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc);
/// @copydoc sscal
EINSUMS_EXPORT void zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc);
/// @copydoc sscal
EINSUMS_EXPORT void csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc);
/// @copydoc sscal
EINSUMS_EXPORT void zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc);

/**
 * Scales a vector by the reciprocal of a real value.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The denominator for the scale.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void srscl(int_t n, float alpha, float *vec, int_t inc);
EINSUMS_EXPORT void drscl(int_t n, double alpha, double *vec, int_t inc);
EINSUMS_EXPORT void csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc);
EINSUMS_EXPORT void zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc);

/**
 * Compute the dot product between two vectors.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @versionadded{1.0.0}
 */
auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float;
/// @copydoc sdot
auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double;
/**
 * Compute the unconjugated dot product between two vectors. The dot product is normally defined to be @f$ \sum_{i} x_i^*y_i@f$,
 * where the left vector is conjugated. However, this function performs @f$\sum_{i} x_iy_i@f$, akin to how it is done for
 * real vectors. For real vectors, the two definitions are the same. For complex vectors, the two definitions are different,
 * so keep this in mind when selecting between the two functions. This is the equivalent to BLAS's cdot<b>u</b>.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @sa cdotc
 *
 * @versionadded{1.0.0}
 */
auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float>;
/// @copydoc cdot
auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double>;
/**
 * Compute the dot product between two vectors. The dot product is normally defined to be @f$ \sum_{i} x_i^*y_i@f$,
 * where the left vector is conjugated. This is how this function works. There is another way, where the first vector is not conjugated. For
 * that, use cdot. For real vectors, the two definitions are the same. For complex vectors, the two definitions are different, so keep this
 * in mind when selecting between the two functions.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @versionadded{1.0.0}
 */
auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float>;
/// @copydoc cdotc
auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double>;

/**
 * Scale and add a vector to another. Performs @f$\mathbf{y} := \alpha\mathbf{x} + \mathbf{y}@f$.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. If it is negative, the vector is traversed backwards.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. If it is negative, the vector is traversed backwards.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y);
/// @copydoc saxpy
EINSUMS_EXPORT void daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y);
/// @copydoc saxpy
EINSUMS_EXPORT void caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y,
                          int_t inc_y);
/// @copydoc saxpy
EINSUMS_EXPORT void zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y,
                          int_t inc_y);

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{xy}^T + \mathbf{A}
 * @f]
 *
 * @param[in] m The number of elements in the left vector.
 * @param[in] n The number of elements in the right vector.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The left vector.
 * @param[in] inc_x The skip value for the left vector.
 * @param[in] y The right vector.
 * @param[in] inc_y The skip value for the right vector.
 * @param[out] a The matrix to update.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @throws std::domain_error If the dimensions are zero, or the leading dimension of the array is less than the number of columns.
 * @throws std::invalid_argument If either of the increment values is zero.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda);
/// @copydoc sger
EINSUMS_EXPORT void dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda);
/// @copydoc sger
EINSUMS_EXPORT void cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                         std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
/// @copydoc sger
EINSUMS_EXPORT void zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                         std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);
/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{xy}^H + \mathbf{A}
 * @f]
 *
 * @param[in] m The number of elements in the left vector.
 * @param[in] n The number of elements in the right vector.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The left vector.
 * @param[in] inc_x The skip value for the left vector.
 * @param[in] y The right vector.
 * @param[in] inc_y The skip value for the right vector.
 * @param[out] a The matrix to update.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @throws std::domain_error If the dimensions are zero, or the leading dimension of the array is less than the number of columns.
 * @throws std::invalid_argument If either of the increment values is zero.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                          std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
/// @copydoc cgerc
EINSUMS_EXPORT void zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                          std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row int_terchanges.
 *
 * The factorization has the form
 * @f[
 *  \mathbf{A} = \mathbf{PLU}
 * @f]
 *
 * where @f$\mathbf{P}@f$ is a permutation matri, @f$\mathbf{L}@f$ is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and @f$\mathbf{U}@f$ is upper
 * triangular (upper trapezoidal if m < n).
 *
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[inout] A The matrix to factor.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] ipiv The list of pivots used in the computation.
 *
 * @return 0 on success. If positive, the factorization completed successfully, but the matrix is singular so should not be used to solve a
 * system of equations. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{1.0.0}
 */
auto sgetrf(int_t m, int_t n, float *A, int_t lda, int_t *ipiv) -> int_t;
/// @copydoc sgetrf
auto dgetrf(int_t m, int_t n, double *A, int_t lda, int_t *ipiv) -> int_t;
/// @copydoc sgetrf
auto cgetrf(int_t m, int_t n, std::complex<float> *A, int_t lda, int_t *ipiv) -> int_t;
/// @copydoc sgetrf
auto zgetrf(int_t m, int_t n, std::complex<double> *A, int_t lda, int_t *ipiv) -> int_t;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * @param[in] n The number of rows and columns of the matrix.
 * @param[inout] A The matrix to invert after being processed by sgetrf.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] ipiv The pivots produced by sgetrf.
 *
 * @return 0 on success. If positive, then the matrix is singular and the inverse could not be computed.
 * If negative, then one of the parameters was invalid. The absolute value indicates which parameter was invalid.
 *
 * @versionadded{1.0.0}
 */
auto sgetri(int_t n, float *A, int_t lda, int_t const *ipiv) -> int_t;
/// @copydoc sgetri
auto dgetri(int_t n, double *A, int_t lda, int_t const *ipiv) -> int_t;
/// @copydoc sgetri
auto cgetri(int_t n, std::complex<float> *A, int_t lda, int_t const *ipiv) -> int_t;
/// @copydoc sgetri
auto zgetri(int_t n, std::complex<double> *A, int_t lda, int_t const *ipiv) -> int_t;

/**
 * Computes matrix norms.
 *
 * @param[in] norm_type The type of norm to compute. Case insensitive. Can be 'm', 'o', 'i', 'f', 'e', or '1'.
 * Note that that is the character '1', or @c 0x31 . It is not the value 1, which is the start-of-heading character.
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[in] A The matrix in consideration.
 * @param[in] lda The leading dimension of the matrix.
 * @param[inout] work A work array used by certain norms.
 *
 * @return The requested norm of the array.
 *
 * @versionadded{1.0.0}
 */
auto slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float;
/// @copydoc slange
auto dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double;
/// @copydoc slange
auto clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float;
/// @copydoc slange
auto zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double;

/**
 * Compute the sum of squares of a vector without overflow. The sum of squares will be <tt>scale * scale * sumsq</tt>.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] x The vector.
 * @param[in] incx The skip value for the vector.
 * @param[inout] scale The scale factor for the sum that avoids overflow/underflow. If it is non-zero, the @p sumsq will be considered as
 * part of the sum as well.
 * @param[inout] sumsq The scaled sum of squares. If it is non-zero, then it will be considered as part of the sum as well.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq);
/// @copydoc slassq
EINSUMS_EXPORT void dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq);
/// @copydoc slassq
EINSUMS_EXPORT void classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq);
/// @copydoc slassq
EINSUMS_EXPORT void zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq);

/**
 * Compute the Euclidean norm of a vector.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The Euclidean norm of the vector.
 *
 * @versionadded{2.0.0}
 */
float snrm2(int_t n, float const *x, int_t incx);
/// @copydoc snrm2
double dnrm2(int_t n, double const *x, int_t incx);
/// @copydoc snrm2
float scnrm2(int_t n, std::complex<float> const *x, int_t incx);
/// @copydoc snrm2
double dznrm2(int_t n, std::complex<double> const *x, int_t incx);

/**
 * Compute the singular value decomposition using the QR algorithm.
 *
 * @param[in] jobu Whether to compute the U vectors. Case insensitive. Can be either 'a', 's', 'o', or 'n'.
 * @param[in] jobvt Whether to compute the V vectors. Case insensitive. Can be either 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[inout] a The matrix to decompose. Overwritten on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values.
 * @param[out] u The U vectors.
 * @param[in] ldu The leading dimension of the U vectors. Must always be at least 1, even when not referenced.
 * @param[out] vt The transpose of the V vectors.
 * @param[in] ldvt The leading dimension of the transpose of the V vectors. Must always be at least 1, even when not referenced.
 * @param[out] superb Storage space for intermediates used in the calculation.
 *
 * @return 0 on success. If positive, then the algorithm did not converge. If negative, then one of the parameters had an invalid value.
 * The absolute value gives the parameter.
 *
 * @versionadded{1.0.0}
 */
auto sgesvd(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
            float *superb) -> int_t;
/// @copydoc sgesvd
auto dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt,
            double *superb) -> int_t;
/// @copydoc sgesvd
auto cgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt, std::complex<float> *superb) -> int_t;
/// @copydoc sgesvd
auto zgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt, std::complex<double> *superb) -> int_t;

/**
 * Compute the singular value decomposition using the divide-and-conquer algorithm.
 *
 * @param[in] jobz Whether to compute the U and V vectors. Case insensitive. Can be either 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[inout] a The matrix to decompose. Overwritten on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values.
 * @param[out] u The U vectors.
 * @param[in] ldu The leading dimension of the U vectors. Must always be at least 1, even when not referenced.
 * @param[out] vt The transpose of the V vectors.
 * @param[in] ldvt The leading dimension of the transpose of the V vectors. Must always be at least 1, even when not referenced.
 *
 * @return 0 on success. If positive, then the algorithm did not converge. If negative, then one of the parameters had an invalid value.
 * The absolute value gives the parameter.
 *
 * @versionadded{1.0.0}
 */
auto sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t;
/// @copydoc sgesdd
auto dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt) -> int_t;
/// @copydoc sgesdd
auto cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
            std::complex<float> *vt, int_t ldvt) -> int_t;
/// @copydoc sgesdd
auto zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
            std::complex<double> *vt, int_t ldvt) -> int_t;

/**
 * Perform Schur decomposition on a matrix.
 *
 * @param[in] jobvs Whether to compute the Schur vectors. Case insensitive. Can be either 'n' or 'v'.
 * @param[in] n The number of rows and columns in the matrix.
 * @param[inout] a The matrix to decompose.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] sdim The dimensions of the Schur blocks.
 * @param[out] wr The real component of the eigenvalues.
 * @param[out] wi The imaginary component of the eigenvalues.
 * @param[out] vs The Schur vectors.
 * @param[in] ldvs The leading dimension of the Schur vectors. Even if not referenced, it needs to be at least 1.
 *
 * @return 0 on success. If positive and at most @p n , then the algorithm did not converge. If <tt>n + 1</tt> then the eigenvalues could
 * not be sorted due to the problem being ill-conditioned. If <tt>n + 2</tt> then roundoff changed some of the values after reordering. If
 * negative, then one of the parameters had an invalid value. The absolute value gives the parameter.
 *
 * @versionadded{1.0.0}
 */
auto sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;
/// @copydoc sgees
auto dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
/**
 * Perform Schur decomposition on a matrix.
 *
 * @param[in] jobvs Whether to compute the Schur vectors. Case insensitive. Can be either 'n' or 'v'.
 * @param[in] n The number of rows and columns in the matrix.
 * @param[inout] a The matrix to decompose.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] sdim The dimensions of the Schur blocks.
 * @param[out] w The eigenvalues.
 * @param[out] vs The Schur vectors.
 * @param[in] ldvs The leading dimension of the Schur vectors. Even if not referenced, it needs to be at least 1.
 *
 * @return 0 on success. If positive and at most @p n , then the algorithm did not converge. If <tt>n + 1</tt> then the eigenvalues could
 * not be sorted due to the problem being ill-conditioned. If <tt>n + 2</tt> then roundoff changed some of the values after reordering. If
 * negative, then one of the parameters had an invalid value. The absolute value gives the parameter.
 *
 * @versionadded{1.1.0}
 */
auto cgees(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w, std::complex<float> *vs, int_t ldvs)
    -> int_t;
/// @copydoc cgees
auto zgees(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w, std::complex<double> *vs,
           int_t ldvs) -> int_t;

/**
 * Solve a Sylvester equation of the form @f$\mathbf{AX} \pm \mathbf{XB} = \alpha\mathbf{C}@f$.
 *
 * @param[in] trana Whether to transpose the first matrix. Case insensitive. Can be either 't', 'c', or 'n'.
 * @param[in] tranb Whether to transpose the second matrix. Case insensitive. Can be either 't', 'c', or 'n'.
 * @param[in] isgn Whether the operation between the terms is addition or subtraction. Can be either 1 or -1.
 * @param[in] m The number of rows in the output matrix.
 * @param[in] n The number of columns in the output matrix.
 * @param[in] a The first input matrix.
 * @param[in] lda The leading dimension of the first input matrix.
 * @param[in] b The second matrix.
 * @param[in] ldb The leading dimension of the second matrix.
 * @param[inout] c On entry, the right-hand side matrix. On exit, the @f$\mathbf{X}@f$ matrix, scaled to avoid overflow/underflow.
 * @param[in] ldc The leading dimension of the right-hand side matrix.
 * @param[out] scale The scale factor that avoid overflow/underflow in the solution matrix.
 *
 * @return 0 on success, 1 if the inputs have similar eigenvalues that needed to be perturbed. If negative, then there was a parameter with
 * an invalid value. The absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
auto strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb, float *c, int_t ldc,
            float *scale) -> int_t;
/// @copydoc strsyl
auto dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb, double *c,
            int_t ldc, double *scale) -> int_t;
/**
 * Solve a Sylvester equation of the form @f$\mathbf{AX} \pm \mathbf{XB} = \alpha\mathbf{C}@f$.
 *
 * @param[in] trana Whether to transpose the first matrix. Case insensitive. Can be either 't', 'c', or 'n'.
 * @param[in] tranb Whether to transpose the second matrix. Case insensitive. Can be either 't', 'c', or 'n'.
 * @param[in] isgn Whether the operation between the terms is addition or subtraction. Can be either 1 or -1.
 * @param[in] m The number of rows in the output matrix.
 * @param[in] n The number of columns in the output matrix.
 * @param[in] a The first input matrix.
 * @param[in] lda The leading dimension of the first input matrix.
 * @param[in] b The second matrix.
 * @param[in] ldb The leading dimension of the second matrix.
 * @param[inout] c On entry, the right-hand side matrix. On exit, the @f$\mathbf{X}@f$ matrix, scaled to avoid overflow/underflow.
 * @param[in] ldc The leading dimension of the right-hand side matrix.
 * @param[out] scale The scale factor that avoid overflow/underflow in the solution matrix.
 *
 * @return 0 on success, 1 if the inputs have similar eigenvalues that needed to be perturbed. If negative, then there was a parameter with
 * an invalid value. The absolute value of the return gives the parameter.
 *
 * @versionadded{2.0.0}
 */
auto ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> const *b,
            int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t;
/// @copydoc ctrsyl
auto ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> const *b,
            int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t;

/**
 * Perform QR decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The matrix to decompose. On exit, the R matrix is above the diagonal and the reflection vectors are below the diagonal.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] tau The scale factors for the Householder reflectors.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{1.0.0}
 */
auto sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
/// @copydoc sgeqrf
auto dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
/// @copydoc sgeqrf
auto cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
/// @copydoc sgeqrf
auto zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;

/**
 * Extract the Q matrix from a QR decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used to decompose the matrix.
 * @param[inout] a The input matrix after being processed by sgeqrf. On exit, it will contain the Q matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] tau The scale factors for the elementary reflectors used in the decomposition.
 *
 * @return 0 on success. If negative, one of the parameters had an invalid value. The absolute value indicates which parameter was invalid.
 *
 * @versionadded{1.0.0}
 */
auto sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
/// @copydoc sorgqr
auto dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
/// @copydoc sorgqr
auto cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
/// @copydoc sorgqr
auto zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;

/**
 * Multiply a general matrix by the orthogonal matrix Q from geqrf.
 *
 * Computes C := op(Q) * C  (side='L') or  C := C * op(Q)  (side='R')
 * where Q is defined by the Householder reflectors from geqrf.
 * op(Q) = Q (trans='N') or Q^T (trans='T') for real, Q^H (trans='C') for complex.
 *
 * @param[in] side 'L' or 'R': apply Q from the left or right.
 * @param[in] trans 'N', 'T', or 'C': no transpose, transpose, or conjugate transpose.
 * @param[in] m Number of rows of C.
 * @param[in] n Number of columns of C.
 * @param[in] k Number of elementary reflectors (from geqrf).
 * @param[in] a The Householder vectors from geqrf (m×k if side='L', n×k if side='R').
 * @param[in] lda Leading dimension of a.
 * @param[in] tau The scalar factors of the reflectors from geqrf.
 * @param[in,out] c On entry, the m×n matrix C. On exit, overwritten by op(Q)*C or C*op(Q).
 * @param[in] ldc Leading dimension of c.
 * @return 0 on success; negative value indicates invalid parameter.
 */
auto sormqr(char side, char trans, int_t m, int_t n, int_t k, float const *a, int_t lda, float const *tau, float *c, int_t ldc) -> int_t;
/// @copydoc sormqr
auto dormqr(char side, char trans, int_t m, int_t n, int_t k, double const *a, int_t lda, double const *tau, double *c, int_t ldc) -> int_t;
/// @copydoc sormqr
auto cunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<float> const *a, int_t lda, std::complex<float> const *tau,
            std::complex<float> *c, int_t ldc) -> int_t;
/// @copydoc sormqr
auto zunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<double> const *a, int_t lda, std::complex<double> const *tau,
            std::complex<double> *c, int_t ldc) -> int_t;

/**
 * Perform LQ decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The matrix to decompose. On exit, the R matrix is above the diagonal and the reflection vectors are below the diagonal.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] tau The scale factors for the Householder reflectors.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{1.0.0}
 */
auto sgelqf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
/// @copydoc sgelqf
auto dgelqf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
/// @copydoc sgelqf
auto cgelqf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
/// @copydoc sgelqf
auto zgelqf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;

/**
 * Extract the Q matrix from a LQ decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used to decompose the matrix.
 * @param[inout] a The input matrix after being processed by sgelqf. On exit, it will contain the Q matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] tau The scale factors for the elementary reflectors used in the decomposition.
 *
 * @return 0 on success. If negative, one of the parameters had an invalid value. The absolute value indicates which parameter was invalid.
 *
 * @versionadded{1.0.0}
 */
auto sorglq(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
/// @copydoc sorglq
auto dorglq(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
/// @copydoc sorglq
auto cunglq(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
/// @copydoc sorglq
auto zunglq(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;

/**
 * Copy one vector to another.
 *
 * @param[in] n The number of elements to copy.
 * @param[in] x The source vector.
 * @param[in] incx The skip value for the input. If zero, it will fill the output with the same value. If negative, then the input will be
 * traversed backwards.
 * @param[out] y The destination vector.
 * @param[in] incy The skip value for the destination vector. If it is negative, the destination will be traversed backwards.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void scopy(int_t n, float const *x, int_t incx, float *y, int_t incy);
/// @copydoc scopy
EINSUMS_EXPORT void dcopy(int_t n, double const *x, int_t incx, double *y, int_t incy);
/// @copydoc scopy
EINSUMS_EXPORT void ccopy(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> *y, int_t incy);
/// @copydoc scopy
EINSUMS_EXPORT void zcopy(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> *y, int_t incy);

/**
 * Advanced matrix scaling operation that scales a matrix by <tt>cto/cfrom</tt> without overflow/underflow.
 *
 * @param[in] type The kind of matrix being processed. Case insensitive. Can be either 'g', 'l', 'u', 'h', 'b', 'q', or 'z'.
 * @param[in] kl The lower bandwidth of the matrix.
 * @param[in] ku The upper bandwidth of the matrix.
 * @param[in] cfrom The denominator for the scale.
 * @param[in] cto The numerator for the scale.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[out] vec The matrix to scale.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{2.0.0}
 */
int_t slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda);
/// @copydoc slascl
int_t dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda);

/**
 * Perform the direct product between two vectors and add it to another.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The first input vector.
 * @param[in] incx The skip value for the first vector. If negative, then it will be traversed backwards.
 * @param[in] y The second vector.
 * @param[in] incy The skip value for the second vector. If negative, then it will be traversed backwards.
 * @param[inout] z The vector to accumulate to.
 * @param[in] incz The skip value for the accumulation vector. If negative, then it will be traversed backwards.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz);
/// @copydoc sdirprod
EINSUMS_EXPORT void ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz);
/// @copydoc sdirprod
EINSUMS_EXPORT void cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y,
                             int_t incy, std::complex<float> *z, int_t incz);
/// @copydoc sdirprod
EINSUMS_EXPORT void zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y,
                             int_t incy, std::complex<double> *z, int_t incz);

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the absolute values of the components, not the magnitudes.
 *
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
float sasum(int_t n, float const *x, int_t incx);
/// @copydoc sasum
double dasum(int_t n, double const *x, int_t incx);
/// @copydoc sasum
float scasum(int_t n, std::complex<float> const *x, int_t incx);
/// @copydoc sasum
double dzasum(int_t n, std::complex<double> const *x, int_t incx);

/**
 * Computes the sum of the absolute values of the input vector.
 *
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs.
 *
 * @versionadded{2.0.0}
 */
float scsum1(int_t n, std::complex<float> const *x, int_t incx);
/// @copydoc scsum1
double dzsum1(int_t n, std::complex<double> const *x, int_t incx);

/**
 * Conjugate a vector.
 *
 * @param[in] n The number of elements in the vector.
 * @param[inout] x The vector to conjugate.
 * @param[in] incx The skip value for the vector.
 */
EINSUMS_EXPORT void clacgv(int_t n, std::complex<float> *x, int_t incx);
/// @copydoc clacgv
EINSUMS_EXPORT void zlacgv(int_t n, std::complex<double> *x, int_t incx);

// =========================================================================
// New BLAS/LAPACK wrappers
// =========================================================================

// --- BLAS Level 1 ---

/// Swap two vectors.
EINSUMS_EXPORT void sswap(int_t n, float *x, int_t incx, float *y, int_t incy);
/// @copydoc sswap
EINSUMS_EXPORT void dswap(int_t n, double *x, int_t incx, double *y, int_t incy);
/// @copydoc sswap
EINSUMS_EXPORT void cswap(int_t n, std::complex<float> *x, int_t incx, std::complex<float> *y, int_t incy);
/// @copydoc sswap
EINSUMS_EXPORT void zswap(int_t n, std::complex<double> *x, int_t incx, std::complex<double> *y, int_t incy);

/// Index of the element with maximum absolute value (0-based).
auto isamax(int_t n, float const *x, int_t incx) -> int_t;
/// @copydoc isamax
auto idamax(int_t n, double const *x, int_t incx) -> int_t;
/// @copydoc isamax
auto icamax(int_t n, std::complex<float> const *x, int_t incx) -> int_t;
/// @copydoc isamax
auto izamax(int_t n, std::complex<double> const *x, int_t incx) -> int_t;

// --- BLAS Level 2 ---

/// Solve a triangular system with a single vector RHS: op(A)*x = b.
EINSUMS_EXPORT void strsv(char uplo, char trans, char diag, int_t n, float const *a, int_t lda, float *x, int_t incx);
/// @copydoc strsv
EINSUMS_EXPORT void dtrsv(char uplo, char trans, char diag, int_t n, double const *a, int_t lda, double *x, int_t incx);
/// @copydoc strsv
EINSUMS_EXPORT void ctrsv(char uplo, char trans, char diag, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> *x,
                          int_t incx);
/// @copydoc strsv
EINSUMS_EXPORT void ztrsv(char uplo, char trans, char diag, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> *x,
                          int_t incx);

// --- BLAS Level 3 ---

/// Solve a triangular system with a matrix RHS: op(A)*X = alpha*B or X*op(A) = alpha*B.
EINSUMS_EXPORT void strsm(char side, char uplo, char transa, char diag, int_t m, int_t n, float alpha, float const *a, int_t lda, float *b,
                          int_t ldb);
/// @copydoc strsm
EINSUMS_EXPORT void dtrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, double alpha, double const *a, int_t lda,
                          double *b, int_t ldb);
/// @copydoc strsm
EINSUMS_EXPORT void ctrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<float> alpha,
                          std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb);
/// @copydoc strsm
EINSUMS_EXPORT void ztrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<double> alpha,
                          std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb);

/// Symmetric rank-k update: C := alpha*A*A^T + beta*C  or  alpha*A^T*A + beta*C.
EINSUMS_EXPORT void ssyrk(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float beta, float *c, int_t ldc);
/// @copydoc ssyrk
EINSUMS_EXPORT void dsyrk(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double beta, double *c,
                          int_t ldc);
/// @copydoc ssyrk
EINSUMS_EXPORT void csyrk(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc ssyrk
EINSUMS_EXPORT void zsyrk(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> beta, std::complex<double> *c, int_t ldc);

/// Hermitian rank-k update: C := alpha*A*A^H + beta*C  or  alpha*A^H*A + beta*C. alpha and beta are real.
EINSUMS_EXPORT void cherk(char uplo, char trans, int_t n, int_t k, float alpha, std::complex<float> const *a, int_t lda, float beta,
                          std::complex<float> *c, int_t ldc);
/// @copydoc cherk
EINSUMS_EXPORT void zherk(char uplo, char trans, int_t n, int_t k, double alpha, std::complex<double> const *a, int_t lda, double beta,
                          std::complex<double> *c, int_t ldc);

/// Symmetric matrix multiply: C := alpha*A*B + beta*C  or  alpha*B*A + beta*C, where A is symmetric.
EINSUMS_EXPORT void ssymm(char side, char uplo, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                          float beta, float *c, int_t ldc);
/// @copydoc ssymm
EINSUMS_EXPORT void dsymm(char side, char uplo, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                          double beta, double *c, int_t ldc);
/// @copydoc ssymm
EINSUMS_EXPORT void csymm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc ssymm
EINSUMS_EXPORT void zsymm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);

/// Hermitian matrix multiply: C := alpha*A*B + beta*C  or  alpha*B*A + beta*C, where A is Hermitian.
EINSUMS_EXPORT void chemm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc chemm
EINSUMS_EXPORT void zhemm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);

/// Symmetric rank-2k update: C := alpha*A*B^T + alpha*B*A^T + beta*C.
EINSUMS_EXPORT void ssyr2k(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                           float beta, float *c, int_t ldc);
/// @copydoc ssyr2k
EINSUMS_EXPORT void dsyr2k(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                           double beta, double *c, int_t ldc);
/// @copydoc ssyr2k
EINSUMS_EXPORT void csyr2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
/// @copydoc ssyr2k
EINSUMS_EXPORT void zsyr2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);

/// Hermitian rank-2k update: C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C. beta is real.
EINSUMS_EXPORT void cher2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, float beta, std::complex<float> *c, int_t ldc);
/// @copydoc cher2k
EINSUMS_EXPORT void zher2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, double beta, std::complex<double> *c, int_t ldc);

// --- LAPACK ---

/// Cholesky factorization of a symmetric/Hermitian positive-definite matrix.
auto spotrf(char uplo, int_t n, float *a, int_t lda) -> int_t;
/// @copydoc spotrf
auto dpotrf(char uplo, int_t n, double *a, int_t lda) -> int_t;
/// @copydoc spotrf
auto cpotrf(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t;
/// @copydoc spotrf
auto zpotrf(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t;

/// Solve a symmetric/Hermitian positive-definite system using Cholesky factorization.
auto spotrs(char uplo, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t;
/// @copydoc spotrs
auto dpotrs(char uplo, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t;
/// @copydoc spotrs
auto cpotrs(char uplo, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t;
/// @copydoc spotrs
auto zpotrs(char uplo, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t;

/// Invert a symmetric/Hermitian positive-definite matrix from its Cholesky factorization.
auto spotri(char uplo, int_t n, float *a, int_t lda) -> int_t;
/// @copydoc spotri
auto dpotri(char uplo, int_t n, double *a, int_t lda) -> int_t;
/// @copydoc spotri
auto cpotri(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t;
/// @copydoc spotri
auto zpotri(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t;

/// Generalized symmetric eigenvalue problem: A*x = lambda*B*x.
auto ssygv(int_t itype, char jobz, char uplo, int_t n, float *a, int_t lda, float *b, int_t ldb, float *w) -> int_t;
/// @copydoc ssygv
auto dsygv(int_t itype, char jobz, char uplo, int_t n, double *a, int_t lda, double *b, int_t ldb, double *w) -> int_t;
/// Generalized Hermitian eigenvalue problem: A*x = lambda*B*x.
auto chegv(int_t itype, char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb, float *w)
    -> int_t;
/// @copydoc chegv
auto zhegv(int_t itype, char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb, double *w)
    -> int_t;

/// Symmetric eigenvalue decomposition using divide-and-conquer.
auto ssyevd(char jobz, char uplo, int_t n, float *a, int_t lda, float *w) -> int_t;
/// @copydoc ssyevd
auto dsyevd(char jobz, char uplo, int_t n, double *a, int_t lda, double *w) -> int_t;
/// Hermitian eigenvalue decomposition using divide-and-conquer.
auto cheevd(char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w) -> int_t;
/// @copydoc cheevd
auto zheevd(char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w) -> int_t;

/// Solve a general system using an existing LU factorization.
auto sgetrs(char trans, int_t n, int_t nrhs, float const *a, int_t lda, int_t const *ipiv, float *b, int_t ldb) -> int_t;
/// @copydoc sgetrs
auto dgetrs(char trans, int_t n, int_t nrhs, double const *a, int_t lda, int_t const *ipiv, double *b, int_t ldb) -> int_t;
/// @copydoc sgetrs
auto cgetrs(char trans, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, int_t const *ipiv, std::complex<float> *b, int_t ldb)
    -> int_t;
/// @copydoc sgetrs
auto zgetrs(char trans, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, int_t const *ipiv, std::complex<double> *b,
            int_t ldb) -> int_t;

/// Solve a least-squares problem using QR or LQ factorization.
auto sgels(char trans, int_t m, int_t n, int_t nrhs, float *a, int_t lda, float *b, int_t ldb) -> int_t;
/// @copydoc sgels
auto dgels(char trans, int_t m, int_t n, int_t nrhs, double *a, int_t lda, double *b, int_t ldb) -> int_t;
/// @copydoc sgels
auto cgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb) -> int_t;
/// @copydoc sgels
auto zgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb) -> int_t;

/// Solve a triangular system with multiple RHS.
auto strtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t;
/// @copydoc strtrs
auto dtrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t;
/// @copydoc strtrs
auto ctrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b,
            int_t ldb) -> int_t;
/// @copydoc strtrs
auto ztrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b,
            int_t ldb) -> int_t;

/// Invert a triangular matrix.
auto strtri(char uplo, char diag, int_t n, float *a, int_t lda) -> int_t;
/// @copydoc strtri
auto dtrtri(char uplo, char diag, int_t n, double *a, int_t lda) -> int_t;
/// @copydoc strtri
auto ctrtri(char uplo, char diag, int_t n, std::complex<float> *a, int_t lda) -> int_t;
/// @copydoc strtri
auto ztrtri(char uplo, char diag, int_t n, std::complex<double> *a, int_t lda) -> int_t;

} // namespace einsums::blas::vendor