//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/export_definitions.hpp>

#include <einsums/concepts/ComplexTraits.hpp>
#include <einsums/linear_algebra/Types.hpp>

// Namespace for BLAS and LAPACK routines.
namespace einsums::blas {

/**
 * @brief Initializes the underlying BLAS library.
 *
 * Handles any initialization that the underlying BLAS implementation requires.
 * For example, a GPU implementation would likely need to obtain a device handle to
 * run. That would be handled by this function.
 *
 * You typically will not need to call this function manually. \ref einsums::initialize()
 * will handle calling this function for you.
 *
 */
void EINSUMS_EXPORT initialize();

/**
 * @brief Handles any shutdown procedure needed by the BLAS implementation.
 *
 */
void EINSUMS_EXPORT finalize();

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {
// These routines take care of actually making the call to the BLAS equivalent.
void EINSUMS_EXPORT sgemm(char transa, char transb, eint m, eint n, eint k, float alpha, float const *a, eint lda,
                          float const *b, eint ldb, float beta, float *c, eint ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, eint m, eint n, eint k, double alpha, double const *a, eint lda,
                          double const *b, eint ldb, double beta, double *c, eint ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, eint m, eint n, eint k, std::complex<float> alpha,
                          std::complex<float> const *a, eint lda, std::complex<float> const *b, eint ldb,
                          std::complex<float> beta, std::complex<float> *c, eint ldc);
void EINSUMS_EXPORT zgemm(char transa, char transb, eint m, eint n, eint k, std::complex<double> alpha,
                          std::complex<double> const *a, eint lda, std::complex<double> const *b, eint ldb,
                          std::complex<double> beta, std::complex<double> *c, eint ldc);
} // namespace detail
#endif

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two double-precision matrices, C = alpha * A * B + beta * C, where A, B, and C
 * are matrices, and alpha and beta are scalar values.
 *
 * @tparam T The datatype of the GEMM.
 * @param[in] transa Whether to transpose matrix \param a:
 *   - 'N' or 'n' for no transpose,
 *   - 'T' or 't' for transpose,
 *   - 'C' or 'c' for conjugate transpose.
 * @param[in] transb Whether to transpose matrix \param b.
 * @param[in] m The number of rows in matrix A and C.
 * @param[in] n The number of columns in matrix B and C.
 * @param[in] k The number of columns in matrix A and rows in matrix B.
 * @param[in] alpha The scalar alpha.
 * @param[in] a A pointer to the matrix A with dimensions ( \param lda , \param k ) when transa is 'N' or 'n', and (
 * \param lda , \param m ) otherwise.
 * @param[in] lda Leading dimension of A, specifying the distance between two consecutive columns.
 * @param[in] b A pointer to the matrix B with dimensions ( \param ldb , \param n) when transB is 'N' or 'n', and (
 * \param ldb , \param k) otherwise.
 * @param[in] ldb Leading dimension of B, specifying the distance between two consecutive columns.
 * @param[in] beta The scalar beta.
 * @param[in,out] c A pointer to the matrix C with dimensions ( \param ldc , \param n ).
 * @param[in] ldc Leading dimension of C, specifying the distance between two consecutive columns.
 *
 * @note The function performs one of the following matrix operations:
 * - If transA is 'N' or 'n' and transB is 'N' or 'n': C = alpha * A * B + beta * C
 * - If transA is 'N' or 'n' and transB is 'T' or 't': C = alpha * A * B^T + beta * C
 * - If transA is 'T' or 't' and transB is 'N' or 'n': C = alpha * A^T * B + beta * C
 * - If transA is 'T' or 't' and transB is 'T' or 't': C = alpha * A^T * B^T + beta * C
 * - If transA is 'C' or 'c' and transB is 'N' or 'n': C = alpha * A^H * B + beta * C
 * - If transA is 'C' or 'c' and transB is 'T' or 't': C = alpha * A^H * B^T + beta * C
 *
 * @return None.
 */
template <typename T>
void gemm(char transa, char transb, eint m, eint n, eint k, T alpha, T const *a, eint lda, T const *b, eint ldb, T beta,
          T *c, eint ldc);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
// These are the template specialization for the data types we support. If a unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, eint m, eint n, eint k, float alpha, float const *a, eint lda,
                        float const *b, eint ldb, float beta, float *c, eint ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, eint m, eint n, eint k, double alpha, double const *a, eint lda,
                         double const *b, eint ldb, double beta, double *c, eint ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, eint m, eint n, eint k, std::complex<float> alpha,
                                      std::complex<float> const *a, eint lda, std::complex<float> const *b, eint ldb,
                                      std::complex<float> beta, std::complex<float> *c, eint ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, eint m, eint n, eint k, std::complex<double> alpha,
                                       std::complex<double> const *a, eint lda, std::complex<double> const *b, eint ldb,
                                       std::complex<double> beta, std::complex<double> *c, eint ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {
void EINSUMS_EXPORT sgemv(char transa, eint m, eint n, float alpha, float const *a, eint lda, float const *x, eint incx,
                          float beta, float *y, eint incy);
void EINSUMS_EXPORT dgemv(char transa, eint m, eint n, double alpha, double const *a, eint lda, double const *x,
                          eint incx, double beta, double *y, eint incy);
void EINSUMS_EXPORT cgemv(char transa, eint m, eint n, std::complex<float> alpha, std::complex<float> const *a,
                          eint lda, std::complex<float> const *x, eint incx, std::complex<float> beta,
                          std::complex<float> *y, eint incy);
void EINSUMS_EXPORT zgemv(char transa, eint m, eint n, std::complex<double> alpha, std::complex<double> const *a,
                          eint lda, std::complex<double> const *x, eint incx, std::complex<double> beta,
                          std::complex<double> *y, eint incy);
} // namespace detail
#endif

/**
 * @brief Computes a matrix-vector product using a general matrix.
 *
 * The gemv routine performs a matrix-vector operation defined as:
 * @f[
 * y := \alpha * A * x + \beta * y
 * @f]
 * or
 * @f[
 * y := \alpha * A' * x + beta * y
 * @f]
 * or
 * @f[
 * y := \alpha * conjg(A') * x + beta * y
 * @f]
 *
 * @tparam T the underlying data type of the matrix and vector
 * @param transa what to do with \p a - no trans, trans, conjg
 * @param m specifies the number of rows of \p a
 * @param n specifies the number of columns of \p a
 * @param alpha Specifies the scaler alpha
 * @param a Array, size lda * m
 * @param lda Specifies the leading dimension of \p a as declared in the calling function
 * @param x array, vector x
 * @param incx Specifies the increment for the elements of \p x
 * @param beta Specifies the scalar beta. When beta is set to zero, then \p y need not be set on input.
 * @param y array, vector y
 * @param incy Specifies the increment for the elements of \p y .
 */
template <typename T>
void gemv(char transa, eint m, eint n, T alpha, T const *a, eint lda, T const *x, eint incx, T beta, T *y, eint incy);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <>
inline void gemv<float>(char transa, eint m, eint n, float alpha, float const *a, eint lda, float const *x, eint incx,
                        float beta, float *y, eint incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, eint m, eint n, double alpha, double const *a, eint lda, double const *x,
                         eint incx, double beta, double *y, eint incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, eint m, eint n, std::complex<float> alpha,
                                      std::complex<float> const *a, eint lda, std::complex<float> const *x, eint incx,
                                      std::complex<float> beta, std::complex<float> *y, eint incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, eint m, eint n, std::complex<double> alpha,
                                       std::complex<double> const *a, eint lda, std::complex<double> const *x,
                                       eint incx, std::complex<double> beta, std::complex<double> *y, eint incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif

/*!
 * Performs symmetric matrix diagonalization.
 */
namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, eint n, float *a, eint lda, float *w, float *work, eint lwork) -> eint;
auto EINSUMS_EXPORT dsyev(char job, char uplo, eint n, double *a, eint lda, double *w, double *work, eint lwork)
    -> eint;
} // namespace detail

template <typename T>
auto syev(char job, char uplo, eint n, T *a, eint lda, T *w, T *work, eint lwork) -> eint;

template <>
inline auto syev<float>(char job, char uplo, eint n, float *a, eint lda, float *w, float *work, eint lwork) -> eint {
    return detail::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto syev<double>(char job, char uplo, eint n, double *a, eint lda, double *w, double *work, eint lwork)
    -> eint {
    return detail::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, eint n, std::complex<float> *a, eint lda, float *w,
                          std::complex<float> *work, eint lwork, float *rwork) -> eint;
auto EINSUMS_EXPORT zheev(char job, char uplo, eint n, std::complex<double> *a, eint lda, double *w,
                          std::complex<double> *work, eint lwork, double *rworl) -> eint;
} // namespace detail

template <typename T>
auto heev(char job, char uplo, eint n, std::complex<T> *a, eint lda, T *w, std::complex<T> *work, eint lwork, T *rwork)
    -> eint;

template <>
inline auto heev<float>(char job, char uplo, eint n, std::complex<float> *a, eint lda, float *w,
                        std::complex<float> *work, eint lwork, float *rwork) -> eint {
    return detail::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, eint n, std::complex<double> *a, eint lda, double *w,
                         std::complex<double> *work, eint lwork, double *rwork) -> eint {
    return detail::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
namespace detail {
auto EINSUMS_EXPORT sgesv(eint n, eint nrhs, float *a, eint lda, eint *ipiv, float *b, eint ldb) -> eint;
auto EINSUMS_EXPORT dgesv(eint n, eint nrhs, double *a, eint lda, eint *ipiv, double *b, eint ldb) -> eint;
auto EINSUMS_EXPORT cgesv(eint n, eint nrhs, std::complex<float> *a, eint lda, eint *ipiv, std::complex<float> *b,
                          eint ldb) -> eint;
auto EINSUMS_EXPORT zgesv(eint n, eint nrhs, std::complex<double> *a, eint lda, eint *ipiv, std::complex<double> *b,
                          eint ldb) -> eint;
} // namespace detail

template <typename T>
auto gesv(eint n, eint nrhs, T *a, eint lda, eint *ipiv, T *b, eint ldb) -> eint;

template <>
inline auto gesv<float>(eint n, eint nrhs, float *a, eint lda, eint *ipiv, float *b, eint ldb) -> eint {
    return detail::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(eint n, eint nrhs, double *a, eint lda, eint *ipiv, double *b, eint ldb) -> eint {
    return detail::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(eint n, eint nrhs, std::complex<float> *a, eint lda, eint *ipiv,
                                      std::complex<float> *b, eint ldb) -> eint {
    return detail::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(eint n, eint nrhs, std::complex<double> *a, eint lda, eint *ipiv,
                                       std::complex<double> *b, eint ldb) -> eint {
    return detail::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

namespace detail {
void EINSUMS_EXPORT sscal(eint n, float const alpha, float *vec, eint inc);
void EINSUMS_EXPORT dscal(eint n, double const alpha, double *vec, eint inc);
void EINSUMS_EXPORT cscal(eint n, std::complex<float> const alpha, std::complex<float> *vec, eint inc);
void EINSUMS_EXPORT zscal(eint n, std::complex<double> const alpha, std::complex<double> *vec, eint inc);
void EINSUMS_EXPORT csscal(eint n, float const alpha, std::complex<float> *vec, eint inc);
void EINSUMS_EXPORT zdscal(eint n, double const alpha, std::complex<double> *vec, eint inc);
} // namespace detail

template <typename T>
void scal(eint n, T const alpha, T *vec, eint inc);

template <Complex T>
void scal(eint n, RemoveComplexT<T> const alpha, T *vec, eint inc);

template <>
inline void scal<float>(eint n, float const alpha, float *vec, eint inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(eint n, double const alpha, double *vec, eint inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(eint n, std::complex<float> const alpha, std::complex<float> *vec, eint inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(eint n, std::complex<double> const alpha, std::complex<double> *vec, eint inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(eint n, float const alpha, std::complex<float> *vec, eint inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(eint n, double const alpha, std::complex<double> *vec, eint inc) {
    detail::zdscal(n, alpha, vec, inc);
}

namespace detail {
auto EINSUMS_EXPORT sdot(eint n, float const *x, eint incx, float const *y, eint incy) -> float;
auto EINSUMS_EXPORT ddot(eint n, double const *x, eint incx, double const *y, eint incy) -> double;
auto EINSUMS_EXPORT cdot(eint n, std::complex<float> const *x, eint incx, std::complex<float> const *y, eint incy)
    -> std::complex<float>;
auto EINSUMS_EXPORT zdot(eint n, std::complex<double> const *x, eint incx, std::complex<double> const *y, eint incy)
    -> std::complex<double>;
} // namespace detail

template <typename T>
auto dot(eint n, T const *x, eint incx, T const *y, eint incy) -> T;

template <>
inline auto dot<float>(eint n, float const *x, eint incx, float const *y, eint incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(eint n, double const *x, eint incx, double const *y, eint incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(eint n, std::complex<float> const *x, eint incx, std::complex<float> const *y,
                                     eint incy) -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(eint n, std::complex<double> const *x, eint incx, std::complex<double> const *y,
                                      eint incy) -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}

namespace detail {
void EINSUMS_EXPORT saxpy(eint n, float alpha_x, float const *x, eint inc_x, float *y, eint inc_y);
void EINSUMS_EXPORT daxpy(eint n, double alpha_x, double const *x, eint inc_x, double *y, eint inc_y);
void EINSUMS_EXPORT caxpy(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x,
                          std::complex<float> *y, eint inc_y);
void EINSUMS_EXPORT zaxpy(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x,
                          std::complex<double> *y, eint inc_y);
} // namespace detail

template <typename T>
void axpy(eint n, T alpha_x, T const *x, eint inc_x, T *y, eint inc_y);

template <>
inline void axpy<float>(eint n, float alpha_x, float const *x, eint inc_x, float *y, eint inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(eint n, double alpha_x, double const *x, eint inc_x, double *y, eint inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x,
                                      std::complex<float> *y, eint inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x,
                                       std::complex<double> *y, eint inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT saxpby(eint n, float alpha_x, float const *x, eint inc_x, float b, float *y, eint inc_y);
void EINSUMS_EXPORT daxpby(eint n, double alpha_x, double const *x, eint inc_x, double b, double *y, eint inc_y);
void EINSUMS_EXPORT caxpby(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x,
                           std::complex<float> b, std::complex<float> *y, eint inc_y);
void EINSUMS_EXPORT zaxpby(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x,
                           std::complex<double> b, std::complex<double> *y, eint inc_y);
} // namespace detail

template <typename T>
void axpby(eint n, T alpha_x, T const *x, eint inc_x, T b, T *y, eint inc_y);

template <>
inline void axpby<float>(eint n, float alpha_x, float const *x, eint inc_x, float b, float *y, eint inc_y) {
    detail::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(eint n, double alpha_x, double const *x, eint inc_x, double b, double *y, eint inc_y) {
    detail::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(eint n, std::complex<float> alpha_x, std::complex<float> const *x, eint inc_x,
                                       std::complex<float> b, std::complex<float> *y, eint inc_y) {
    detail::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(eint n, std::complex<double> alpha_x, std::complex<double> const *x, eint inc_x,
                                        std::complex<double> b, std::complex<double> *y, eint inc_y) {
    detail::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT sger(eint m, eint n, float alpha, float const *x, eint inc_x, float const *y, eint inc_y, float *a,
                         eint lda);
void EINSUMS_EXPORT dger(eint m, eint n, double alpha, double const *x, eint inc_x, double const *y, eint inc_y,
                         double *a, eint lda);
void EINSUMS_EXPORT cger(eint m, eint n, std::complex<float> alpha, std::complex<float> const *x, eint inc_x,
                         std::complex<float> const *y, eint inc_y, std::complex<float> *a, eint lda);
void EINSUMS_EXPORT zger(eint m, eint n, std::complex<double> alpha, std::complex<double> const *x, eint inc_x,
                         std::complex<double> const *y, eint inc_y, std::complex<double> *a, eint lda);
} // namespace detail

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
template <typename T>
void ger(eint m, eint n, T alpha, T const *x, eint inc_x, T const *y, eint inc_y, T *a, eint lda);

template <>
inline void ger<float>(eint m, eint n, float alpha, float const *x, eint inc_x, float const *y, eint inc_y, float *a,
                       eint lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(eint m, eint n, double alpha, double const *x, eint inc_x, double const *y, eint inc_y,
                        double *a, eint lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(eint m, eint n, std::complex<float> alpha, std::complex<float> const *x,
                                     eint inc_x, std::complex<float> const *y, eint inc_y, std::complex<float> *a,
                                     eint lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(eint m, eint n, std::complex<double> alpha, std::complex<double> const *x,
                                      eint inc_x, std::complex<double> const *y, eint inc_y, std::complex<double> *a,
                                      eint lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

namespace detail {
auto EINSUMS_EXPORT sgetrf(eint, eint, float *, eint, eint *) -> eint;
auto EINSUMS_EXPORT dgetrf(eint, eint, double *, eint, eint *) -> eint;
auto EINSUMS_EXPORT cgetrf(eint, eint, std::complex<float> *, eint, eint *) -> eint;
auto EINSUMS_EXPORT zgetrf(eint, eint, std::complex<double> *, eint, eint *) -> eint;
} // namespace detail

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row einterchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
template <typename T>
auto getrf(eint, eint, T *, eint, eint *) -> eint;

template <>
inline auto getrf<float>(eint m, eint n, float *a, eint lda, eint *ipiv) -> eint {
    return detail::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(eint m, eint n, double *a, eint lda, eint *ipiv) -> eint {
    return detail::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(eint m, eint n, std::complex<float> *a, eint lda, eint *ipiv) -> eint {
    return detail::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(eint m, eint n, std::complex<double> *a, eint lda, eint *ipiv) -> eint {
    return detail::zgetrf(m, n, a, lda, ipiv);
}

namespace detail {
auto EINSUMS_EXPORT sgetri(eint n, float *a, eint lda, eint const *ipiv) -> eint;
auto EINSUMS_EXPORT dgetri(eint n, double *a, eint lda, eint const *ipiv) -> eint;
auto EINSUMS_EXPORT cgetri(eint n, std::complex<float> *a, eint lda, eint const *ipiv) -> eint;
auto EINSUMS_EXPORT zgetri(eint n, std::complex<double> *a, eint lda, eint const *ipiv) -> eint;
} // namespace detail

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
template <typename T>
auto getri(eint n, T *a, eint lda, eint const *ipiv) -> eint;

template <>
inline auto getri<float>(eint n, float *a, eint lda, eint const *ipiv) -> eint {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(eint n, double *a, eint lda, eint const *ipiv) -> eint {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(eint n, std::complex<float> *a, eint lda, eint const *ipiv) -> eint {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(eint n, std::complex<double> *a, eint lda, eint const *ipiv) -> eint {
    return detail::zgetri(n, a, lda, ipiv);
}

/*!
 * Return the value of the 1-norm, Frobenius norm, infinity-norm, or the
 * largest absolute value of any element of a general rectangular matrix
 */
namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, eint m, eint n, float const *A, eint lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, eint m, eint n, double const *A, eint lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, eint m, eint n, std::complex<float> const *A, eint lda, float *work)
    -> float;
auto EINSUMS_EXPORT zlange(char norm_type, eint m, eint n, std::complex<double> const *A, eint lda, double *work)
    -> double;
} // namespace detail

template <typename T>
auto lange(char norm_type, eint m, eint n, T const *A, eint lda, RemoveComplexT<T> *work) -> RemoveComplexT<T>;

template <>
inline auto lange<float>(char norm_type, eint m, eint n, float const *A, eint lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, eint m, eint n, double const *A, eint lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, eint m, eint n, std::complex<float> const *A, eint lda,
                                       float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, eint m, eint n, std::complex<double> const *A, eint lda,
                                        double *work) -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}

namespace detail {
void EINSUMS_EXPORT slassq(eint n, float const *x, eint incx, float *scale, float *sumsq);
void EINSUMS_EXPORT dlassq(eint n, double const *x, eint incx, double *scale, double *sumsq);
void EINSUMS_EXPORT classq(eint n, std::complex<float> const *x, eint incx, float *scale, float *sumsq);
void EINSUMS_EXPORT zlassq(eint n, std::complex<double> const *x, eint incx, double *scale, double *sumsq);
} // namespace detail

template <typename T>
void lassq(eint n, T const *x, eint incx, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq);

template <>
inline void lassq<float>(eint n, float const *x, eint incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(eint n, double const *x, eint incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(eint n, std::complex<float> const *x, eint incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(eint n, std::complex<double> const *x, eint incx, double *scale,
                                        double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}

/*!
 * Computes the singular value decomposition of a general rectangular
 * matrix using a divide and conquer method.
 */
namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt,
                           eint ldvt) -> eint;
auto EINSUMS_EXPORT dgesdd(char jobz, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt,
                           eint ldvt) -> eint;
auto EINSUMS_EXPORT cgesdd(char jobz, eint m, eint n, std::complex<float> *a, eint lda, float *s,
                           std::complex<float> *u, eint ldu, std::complex<float> *vt, eint ldvt) -> eint;
auto EINSUMS_EXPORT zgesdd(char jobz, eint m, eint n, std::complex<double> *a, eint lda, double *s,
                           std::complex<double> *u, eint ldu, std::complex<double> *vt, eint ldvt) -> eint;
} // namespace detail

template <typename T>
auto gesdd(char jobz, eint m, eint n, T *a, eint lda, RemoveComplexT<T> *s, T *u, eint ldu, T *vt, eint ldvt) -> eint;

template <>
inline auto gesdd<float>(char jobz, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu, float *vt,
                         eint ldvt) -> eint {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu, double *vt,
                          eint ldvt) -> eint {
    return detail::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, eint m, eint n, std::complex<float> *a, eint lda, float *s,
                                       std::complex<float> *u, eint ldu, std::complex<float> *vt, eint ldvt) -> eint {
    return detail::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, eint m, eint n, std::complex<double> *a, eint lda, double *s,
                                        std::complex<double> *u, eint ldu, std::complex<double> *vt, eint ldvt)
    -> eint {
    return detail::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

namespace detail {
auto EINSUMS_EXPORT sgesvd(char jobu, char jobvt, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu,
                           float *vt, eint ldvt, float *superb) -> eint;
auto EINSUMS_EXPORT dgesvd(char jobu, char jobvt, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu,
                           double *vt, eint ldvt, double *superb) -> eint;
} // namespace detail

template <typename T>
auto gesvd(char jobu, char jobvt, eint m, eint n, T *a, eint lda, T *s, T *u, eint ldu, T *vt, eint ldvt, T *superb);

template <>
inline auto gesvd<float>(char jobu, char jobvt, eint m, eint n, float *a, eint lda, float *s, float *u, eint ldu,
                         float *vt, eint ldvt, float *superb) {
    return detail::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<double>(char jobu, char jobvt, eint m, eint n, double *a, eint lda, double *s, double *u, eint ldu,
                          double *vt, eint ldvt, double *superb) {
    return detail::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, eint n, float *a, eint lda, eint *sdim, float *wr, float *wi, float *vs,
                          eint ldvs) -> eint;
auto EINSUMS_EXPORT dgees(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs,
                          eint ldvs) -> eint;
} // namespace detail

/*!
 * Computes the Schur Decomposition of a Matrix
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto gees(char jobvs, eint n, T *a, eint lda, eint *sdim, T *wr, T *wi, T *vs, eint ldvs) -> eint;

template <>
inline auto gees<float>(char jobvs, eint n, float *a, eint lda, eint *sdim, float *wr, float *wi, float *vs, eint ldvs)
    -> eint {
    return detail::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, eint n, double *a, eint lda, eint *sdim, double *wr, double *wi, double *vs,
                         eint ldvs) -> eint {
    return detail::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b,
                           eint ldb, float *c, eint ldc, float *scale) -> eint;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda,
                           double const *b, eint ldb, double *c, eint ldc, double *scale) -> eint;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a, eint lda,
                           std::complex<float> const *b, eint ldb, std::complex<float> *c, eint ldc, float *scale)
    -> eint;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, eint isgn, eint m, eint n, std::complex<double> const *a, eint lda,
                           std::complex<double> const *b, eint ldb, std::complex<double> *c, eint ldc, double *scale)
    -> eint;
} // namespace detail

/*!
 * Sylvester Solve
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto trsyl(char trana, char tranb, eint isgn, eint m, eint n, T const *a, eint lda, T const *b, eint ldb, T *c,
           eint ldc, RemoveComplexT<T> *scale) -> eint;

template <>
inline auto trsyl<float>(char trana, char tranb, eint isgn, eint m, eint n, float const *a, eint lda, float const *b,
                         eint ldb, float *c, eint ldc, float *scale) -> eint {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, eint isgn, eint m, eint n, double const *a, eint lda, double const *b,
                          eint ldb, double *c, eint ldc, double *scale) -> eint {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, eint isgn, eint m, eint n, std::complex<float> const *a,
                                       eint lda, std::complex<float> const *b, eint ldb, std::complex<float> *c,
                                       eint ldc, float *scale) -> eint {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, eint isgn, eint m, eint n,
                                        std::complex<double> const *a, eint lda, std::complex<double> const *b,
                                        eint ldb, std::complex<double> *c, eint ldc, double *scale) -> eint {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

/*!
 * Computes a QR factorizaton (Useful for orthonormalizing matrices)
 */
namespace detail {
auto EINSUMS_EXPORT sgeqrf(eint m, eint n, float *a, eint lda, float *tau) -> eint;
auto EINSUMS_EXPORT dgeqrf(eint m, eint n, double *a, eint lda, double *tau) -> eint;
auto EINSUMS_EXPORT cgeqrf(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau) -> eint;
auto EINSUMS_EXPORT zgeqrf(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau) -> eint;
} // namespace detail

template <typename T>
auto geqrf(eint m, eint n, T *a, eint lda, T *tau) -> eint;

template <>
inline auto geqrf<float>(eint m, eint n, float *a, eint lda, float *tau) -> eint {
    return detail::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(eint m, eint n, double *a, eint lda, double *tau) -> eint {
    return detail::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(eint m, eint n, std::complex<float> *a, eint lda, std::complex<float> *tau)
    -> eint {
    return detail::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(eint m, eint n, std::complex<double> *a, eint lda, std::complex<double> *tau)
    -> eint {
    return detail::zgeqrf(m, n, a, lda, tau);
}

/*!
 * Returns the orthogonal/unitary matrix Q from the output of dgeqrf
 */
namespace detail {
auto EINSUMS_EXPORT sorgqr(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint;
auto EINSUMS_EXPORT dorgqr(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint;
auto EINSUMS_EXPORT cungqr(eint m, eint n, eint k, std::complex<float> *a, eint lda, std::complex<float> const *tau)
    -> eint;
auto EINSUMS_EXPORT zungqr(eint m, eint n, eint k, std::complex<double> *a, eint lda, std::complex<double> const *tau)
    -> eint;
} // namespace detail

template <typename T>
auto orgqr(eint m, eint n, eint k, T *a, eint lda, T const *tau) -> eint;

template <>
inline auto orgqr<float>(eint m, eint n, eint k, float *a, eint lda, float const *tau) -> eint {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(eint m, eint n, eint k, double *a, eint lda, double const *tau) -> eint {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(eint m, eint n, eint k, T *a, eint lda, T const *tau) -> eint;

template <>
inline auto ungqr<std::complex<float>>(eint m, eint n, eint k, std::complex<float> *a, eint lda,
                                       std::complex<float> const *tau) -> eint {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(eint m, eint n, eint k, std::complex<double> *a, eint lda,
                                        std::complex<double> const *tau) -> eint {
    return detail::zungqr(m, n, k, a, lda, tau);
}

} // namespace einsums::blas