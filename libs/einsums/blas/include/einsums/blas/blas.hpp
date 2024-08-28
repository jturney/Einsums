//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/blas/types.hpp>
#include <einsums/concepts/complex.hpp>

namespace einsums::blas {

/**
 * @brief Initializes the underlying BLAS and LAPACK library.
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

#if !defined(DOXYGEN)
namespace detail {
// These routines take care of actually making the call to the BLAS equivalent.
void EINSUMS_EXPORT sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, const float *a, int_t lda, const float *b,
                          int_t ldb, float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, const double *a, int_t lda, const double *b,
                          int_t ldb, double beta, double *c, int_t ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, const std::complex<float> *a,
                          int_t lda, const std::complex<float> *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, const std::complex<double> *a,
                          int_t lda, const std::complex<double> *b, int_t ldb, std::complex<double> beta, std::complex<double> *c,
                          int_t ldc);
} // namespace detail
#endif

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two matrices,
 * \f[
 * C = alpha * A * B + beta * C,
 * \f]
 * where A, B, and C are matrices, and
 * alpha and beta are scalar values.
 *
 * @tparam T The datatype of the GEMM.
 * @param transa Whether to transpose matrix a :
 *   - 'N' or 'n' for no transpose,
 *   - 'T' or 't' for transpose,
 *   - 'C' or 'c' for conjugate transpose.
 * @param transb Whether to transpose matrix b .
 * @param m The number of rows in matrix A and C.
 * @param n The number of columns in matrix B and C.
 * @param k The number of columns in matrix A and rows in matrix B.
 * @param alpha The scalar alpha.
 * @param a A pointer to the matrix A with dimensions `(lda, k)` when transa is 'N' or 'n', and `(lda, m)`
 * otherwise.
 * @param lda Leading dimension of A, specifying the distance between two consecutive columns.
 * @param b A pointer to the matrix B with dimensions `(ldb, n)` when transB is 'N' or 'n', and `(ldb, k)`
 * otherwise.
 * @param ldb Leading dimension of B, specifying the distance between two consecutive columns.
 * @param beta The scalar beta.
 * @param c A pointer to the matrix C with dimensions `(ldc, n)`.
 * @param ldc Leading dimension of C, specifying the distance between two consecutive columns.
 *
 * @note The function performs one of the following matrix operations:
 * - If transA is 'N' or 'n' and transB is 'N' or 'n': \f$C = alpha * A * B + beta * C\f$
 * - If transA is 'N' or 'n' and transB is 'T' or 't': \f$C = alpha * A * B^T + beta * C\f$
 * - If transA is 'T' or 't' and transB is 'N' or 'n': \f$C = alpha * A^T * B + beta * C\f$
 * - If transA is 'T' or 't' and transB is 'T' or 't': \f$C = alpha * A^T * B^T + beta * C\f$
 * - If transA is 'C' or 'c' and transB is 'N' or 'n': \f$C = alpha * A^H * B + beta * C\f$
 * - If transA is 'C' or 'c' and transB is 'T' or 't': \f$C = alpha * A^H * B^T + beta * C\f$
 *
 * @return None.
 */
template <typename T>
void gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, const T *a, int_t lda, const T *b, int_t ldb, T beta, T *c,
          int_t ldc);

#if !defined(DOXYGEN)
// These are the template specialization for the data types we support. If an unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, int_t m, int_t n, int_t k, float alpha, const float *a, int_t lda, const float *b,
                        int_t ldb, float beta, float *c, int_t ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int_t m, int_t n, int_t k, double alpha, const double *a, int_t lda, const double *b,
                         int_t ldb, double beta, double *c, int_t ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                      const std::complex<float> *a, int_t lda, const std::complex<float> *b, int_t ldb,
                                      std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                       const std::complex<double> *a, int_t lda, const std::complex<double> *b, int_t ldb,
                                       std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {
void EINSUMS_EXPORT sgemv(char transa, int_t m, int_t n, float alpha, const float *a, int_t lda, const float *x, int_t incx, float beta,
                          float *y, int_t incy);
void EINSUMS_EXPORT dgemv(char transa, int_t m, int_t n, double alpha, const double *a, int_t lda, const double *x, int_t incx, double beta,
                          double *y, int_t incy);
void EINSUMS_EXPORT cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *a, int_t lda,
                          const std::complex<float> *x, int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
void EINSUMS_EXPORT zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
                          const std::complex<double> *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);
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
void gemv(char transa, int_t m, int_t n, T alpha, const T *a, int_t lda, const T *x, int_t incx, T beta, T *y, int_t incy);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <>
inline void gemv<float>(char transa, int_t m, int_t n, float alpha, const float *a, int_t lda, const float *x, int_t incx, float beta,
                        float *y, int_t incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int_t m, int_t n, double alpha, const double *a, int_t lda, const double *x, int_t incx, double beta,
                         double *y, int_t incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *a, int_t lda,
                                      const std::complex<float> *x, int_t incx, std::complex<float> beta, std::complex<float> *y,
                                      int_t incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *a, int_t lda,
                                       const std::complex<double> *x, int_t incx, std::complex<double> beta, std::complex<double> *y,
                                       int_t incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif

/*!
 * Performs symmetric matrix diagonalization.
 */
namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
auto EINSUMS_EXPORT dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;
} // namespace detail

template <typename T>
auto syev(char job, char uplo, int_t n, T *a, int_t lda, T *w, T *work, int_t lwork) -> int_t;

template <>
inline auto syev<float>(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    return detail::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto syev<double>(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    return detail::dsyev(job, uplo, n, a, lda, w, work, lwork);
}

/*!
 * Performs matrix diagonalization on a general matrix.
 */
namespace detail {
auto EINSUMS_EXPORT sgeev(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
                          int_t ldvr) -> int_t;
auto EINSUMS_EXPORT dgeev(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl,
                          double *vr, int_t ldvr) -> int_t;
auto EINSUMS_EXPORT cgeev(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w,
                          std::complex<float> *vl, int_t ldvl, std::complex<float> *vr, int_t ldvr) -> int_t;
auto EINSUMS_EXPORT zgeev(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w,
                          std::complex<double> *vl, int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t;
} // namespace detail

// Complex version
template <typename T>
auto geev(char jobvl, char jobvr, int_t n, T *a, int_t lda, add_complex_t<T> *w, T *vl, int_t ldvl, T *vr, int_t ldvr) -> int_t;

template <>
inline auto geev<float>(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
                        int_t ldvr) -> int_t {
    return detail::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<double>(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr,
                         int_t ldvr) -> int_t {
    return detail::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<float>>(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w,
                                      std::complex<float> *vl, int_t ldvl, std::complex<float> *vr, int_t ldvr) -> int_t {
    return detail::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<double>>(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w,
                                       std::complex<double> *vl, int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t {
    return detail::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
                          float *rwork) -> int_t;
auto EINSUMS_EXPORT zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                          int_t lwork, double *rworl) -> int_t;
} // namespace detail

template <typename T>
auto heev(char job, char uplo, int_t n, std::complex<T> *a, int_t lda, T *w, std::complex<T> *work, int_t lwork, T *rwork) -> int_t;

template <>
inline auto heev<float>(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
                        float *rwork) -> int_t {
    return detail::cheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

template <>
inline auto heev<double>(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                         int_t lwork, double *rwork) -> int_t {
    return detail::zheev(job, uplo, n, a, lda, w, work, lwork, rwork);
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
namespace detail {
auto EINSUMS_EXPORT sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b,
                          int_t ldb) -> int_t;
} // namespace detail

template <typename T>
auto gesv(int_t n, int_t nrhs, T *a, int_t lda, int_t *ipiv, T *b, int_t ldb) -> int_t;

template <>
inline auto gesv<float>(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t {
    return detail::sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<double>(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t {
    return detail::dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<float>>(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b,
                                      int_t ldb) -> int_t {
    return detail::cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto gesv<std::complex<double>>(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b,
                                       int_t ldb) -> int_t {
    return detail::zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

namespace detail {
void EINSUMS_EXPORT sscal(int_t n, float alpha, float *vec, int_t inc);
void EINSUMS_EXPORT dscal(int_t n, double alpha, double *vec, int_t inc);
void EINSUMS_EXPORT cscal(int_t n, std::complex<float> alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zscal(int_t n, std::complex<double> alpha, std::complex<double> *vec, int_t inc);
void EINSUMS_EXPORT csscal(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zdscal(int_t n, double alpha, std::complex<double> *vec, int_t inc);
} // namespace detail

template <typename T>
void scal(int_t n, T alpha, T *vec, int_t inc);

template <Complex T>
void scal(int_t n, remove_complex_t<T> alpha, T *vec, int_t inc);

template <>
inline void scal<float>(int_t n, const float alpha, float *vec, int_t inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(int_t n, const double alpha, double *vec, int_t inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, const std::complex<float> alpha, std::complex<float> *vec, int_t inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, const std::complex<double> alpha, std::complex<double> *vec, int_t inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, const float alpha, std::complex<float> *vec, int_t inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, const double alpha, std::complex<double> *vec, int_t inc) {
    detail::zdscal(n, alpha, vec, inc);
}

namespace detail {
auto EINSUMS_EXPORT sdot(int_t n, const float *x, int_t incx, const float *y, int_t incy) -> float;
auto EINSUMS_EXPORT ddot(int_t n, const double *x, int_t incx, const double *y, int_t incy) -> double;
auto EINSUMS_EXPORT cdot(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y,
                         int_t incy) -> std::complex<float>;
auto EINSUMS_EXPORT zdot(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y,
                         int_t incy) -> std::complex<double>;
auto EINSUMS_EXPORT cdotc(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y,
                          int_t incy) -> std::complex<float>;
auto EINSUMS_EXPORT zdotc(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y,
                          int_t incy) -> std::complex<double>;
} // namespace detail

/**
 * Computes the dot product of two vectors. For complex vector it is the non-conjugated dot product;
 * (c|z)dotu in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param n length of the vectors
 * @param x first vector
 * @param incx how many elements to skip in x
 * @param y second vector
 * @param incy how many elements to skip in yo
 * @return result of the dot product
 */
template <typename T>
auto dot(int_t n, const T *x, int_t incx, const T *y, int_t incy) -> T;

template <>
inline auto dot<float>(int_t n, const float *x, int_t incx, const float *y, int_t incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(int_t n, const double *x, int_t incx, const double *y, int_t incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y,
                                     int_t incy) -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y,
                                      int_t incy) -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}

template <typename T>
auto dotc(int_t n, const T *x, int_t incx, const T *y, int_t incy) -> T;

template <>
inline auto dotc<std::complex<float>>(int_t n, const std::complex<float> *x, int_t incx, const std::complex<float> *y,
                                      int_t incy) -> std::complex<float> {
    return detail::cdotc(n, x, incx, y, incy);
}

template <>
inline auto dotc<std::complex<double>>(int_t n, const std::complex<double> *x, int_t incx, const std::complex<double> *y,
                                       int_t incy) -> std::complex<double> {
    return detail::zdotc(n, x, incx, y, incy);
}

namespace detail {
void EINSUMS_EXPORT saxpy(int_t n, float alpha_x, const float *x, int_t inc_x, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpy(int_t n, double alpha_x, const double *x, int_t inc_x, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpy(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> *y,
                          int_t inc_y);
void EINSUMS_EXPORT zaxpy(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> *y,
                          int_t inc_y);
} // namespace detail

template <typename T>
void axpy(int_t n, T alpha_x, const T *x, int_t inc_x, T *y, int_t inc_y);

template <>
inline void axpy<float>(int_t n, float alpha_x, const float *x, int_t inc_x, float *y, int_t inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(int_t n, double alpha_x, const double *x, int_t inc_x, double *y, int_t inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x,
                                      std::complex<float> *y, int_t inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x,
                                       std::complex<double> *y, int_t inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT saxpby(int_t n, float alpha_x, const float *x, int_t inc_x, float b, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpby(int_t n, double alpha_x, const double *x, int_t inc_x, double b, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpby(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x, std::complex<float> b,
                           std::complex<float> *y, int_t inc_y);
void EINSUMS_EXPORT zaxpby(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x, std::complex<double> b,
                           std::complex<double> *y, int_t inc_y);
} // namespace detail

template <typename T>
void axpby(int_t n, T alpha_x, const T *x, int_t inc_x, T b, T *y, int_t inc_y);

template <>
inline void axpby<float>(int_t n, float alpha_x, const float *x, int_t inc_x, float b, float *y, int_t inc_y) {
    detail::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(int_t n, double alpha_x, const double *x, int_t inc_x, double b, double *y, int_t inc_y) {
    detail::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(int_t n, std::complex<float> alpha_x, const std::complex<float> *x, int_t inc_x,
                                       std::complex<float> b, std::complex<float> *y, int_t inc_y) {
    detail::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(int_t n, std::complex<double> alpha_x, const std::complex<double> *x, int_t inc_x,
                                        std::complex<double> b, std::complex<double> *y, int_t inc_y) {
    detail::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT sger(int_t m, int_t n, float alpha, const float *x, int_t inc_x, const float *y, int_t inc_y, float *a, int_t lda);
void EINSUMS_EXPORT dger(int_t m, int_t n, double alpha, const double *x, int_t inc_x, const double *y, int_t inc_y, double *a, int_t lda);
void EINSUMS_EXPORT cger(int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *x, int_t inc_x,
                         const std::complex<float> *y, int_t inc_y, std::complex<float> *a, int_t lda);
void EINSUMS_EXPORT zger(int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *x, int_t inc_x,
                         const std::complex<double> *y, int_t inc_y, std::complex<double> *a, int_t lda);
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
void ger(int_t m, int_t n, T alpha, const T *x, int_t inc_x, const T *y, int_t inc_y, T *a, int_t lda);

template <>
inline void ger<float>(int_t m, int_t n, float alpha, const float *x, int_t inc_x, const float *y, int_t inc_y, float *a, int_t lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(int_t m, int_t n, double alpha, const double *x, int_t inc_x, const double *y, int_t inc_y, double *a, int_t lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, const std::complex<float> *x, int_t inc_x,
                                     const std::complex<float> *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, const std::complex<double> *x, int_t inc_x,
                                      const std::complex<double> *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

namespace detail {
auto EINSUMS_EXPORT sgetrf(int_t, int_t, float *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT dgetrf(int_t, int_t, double *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT cgetrf(int_t, int_t, std::complex<float> *, int_t, int_t *) -> int_t;
auto EINSUMS_EXPORT zgetrf(int_t, int_t, std::complex<double> *, int_t, int_t *) -> int_t;
} // namespace detail

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
template <typename T>
auto getrf(int_t, int_t, T *, int_t, int_t *) -> int_t;

template <>
inline auto getrf<float>(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    return detail::zgetrf(m, n, a, lda, ipiv);
}

namespace detail {
auto EINSUMS_EXPORT sgetri(int_t n, float *a, int_t lda, const int_t *ipiv) -> int_t;
auto EINSUMS_EXPORT dgetri(int_t n, double *a, int_t lda, const int_t *ipiv) -> int_t;
auto EINSUMS_EXPORT cgetri(int_t n, std::complex<float> *a, int_t lda, const int_t *ipiv) -> int_t;
auto EINSUMS_EXPORT zgetri(int_t n, std::complex<double> *a, int_t lda, const int_t *ipiv) -> int_t;
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
auto getri(int_t n, T *a, int_t lda, const int_t *ipiv) -> int_t;

template <>
inline auto getri<float>(int_t n, float *a, int_t lda, const int_t *ipiv) -> int_t {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(int_t n, double *a, int_t lda, const int_t *ipiv) -> int_t {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(int_t n, std::complex<float> *a, int_t lda, const int_t *ipiv) -> int_t {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(int_t n, std::complex<double> *a, int_t lda, const int_t *ipiv) -> int_t {
    return detail::zgetri(n, a, lda, ipiv);
}

/*!
 * Return the value of the 1-norm, Frobenius norm, infinity-norm, or the
 * largest absolute value of any element of a general rectangular matrix
 */
namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, int_t m, int_t n, const float *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, int_t m, int_t n, const double *A, int_t lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, int_t m, int_t n, const std::complex<float> *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT zlange(char norm_type, int_t m, int_t n, const std::complex<double> *A, int_t lda, double *work) -> double;
} // namespace detail

template <typename T>
auto lange(char norm_type, int_t m, int_t n, const T *A, int_t lda, remove_complex_t<T> *work) -> remove_complex_t<T>;

template <>
inline auto lange<float>(char norm_type, int_t m, int_t n, const float *A, int_t lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int_t m, int_t n, const double *A, int_t lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int_t m, int_t n, const std::complex<float> *A, int_t lda, float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int_t m, int_t n, const std::complex<double> *A, int_t lda,
                                        double *work) -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}

namespace detail {
void EINSUMS_EXPORT slassq(int_t n, const float *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT dlassq(int_t n, const double *x, int_t incx, double *scale, double *sumsq);
void EINSUMS_EXPORT classq(int_t n, const std::complex<float> *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT zlassq(int_t n, const std::complex<double> *x, int_t incx, double *scale, double *sumsq);
} // namespace detail

template <typename T>
void lassq(int_t n, const T *x, int_t incx, remove_complex_t<T> *scale, remove_complex_t<T> *sumsq);

template <>
inline void lassq<float>(int_t n, const float *x, int_t incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int_t n, const double *x, int_t incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int_t n, const std::complex<float> *x, int_t incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int_t n, const std::complex<double> *x, int_t incx, double *scale, double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}

/*!
 * Computes the singular value decomposition of a general rectangular
 * matrix using a divide and conquer method.
 */
namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                           int_t ldvt) -> int_t;
auto EINSUMS_EXPORT cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
                           std::complex<float> *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
                           std::complex<double> *vt, int_t ldvt) -> int_t;
} // namespace detail

template <typename T>
auto gesdd(char jobz, int_t m, int_t n, T *a, int_t lda, remove_complex_t<T> *s, T *u, int_t ldu, T *vt, int_t ldvt) -> int_t;

template <>
inline auto gesdd<float>(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                          int_t ldvt) -> int_t {
    return detail::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u,
                                       int_t ldu, std::complex<float> *vt, int_t ldvt) -> int_t {
    return detail::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u,
                                        int_t ldu, std::complex<double> *vt, int_t ldvt) -> int_t {
    return detail::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

namespace detail {
auto EINSUMS_EXPORT sgesvd(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt,
                           int_t ldvt, float *superb) -> int_t;
auto EINSUMS_EXPORT dgesvd(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                           int_t ldvt, double *superb) -> int_t;
} // namespace detail

template <typename T>
auto gesvd(char jobu, char jobvt, int_t m, int_t n, T *a, int_t lda, T *s, T *u, int_t ldu, T *vt, int_t ldvt, T *superb);

template <>
inline auto gesvd<float>(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
                         float *superb) {
    return detail::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<double>(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                          int_t ldvt, double *superb) {
    return detail::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
} // namespace detail

/*!
 * Computes the Schur Decomposition of a Matrix
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *wr, T *wi, T *vs, int_t ldvs) -> int_t;

template <>
inline auto gees<float>(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t {
    return detail::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t {
    return detail::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const float *a, int_t lda, const float *b, int_t ldb,
                           float *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const double *a, int_t lda, const double *b, int_t ldb,
                           double *c, int_t ldc, double *scale) -> int_t;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<float> *a, int_t lda,
                           const std::complex<float> *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<double> *a, int_t lda,
                           const std::complex<double> *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t;
} // namespace detail

/*!
 * Sylvester Solve
 * (Used in Lyapunov Solves)
 */
template <typename T>
auto trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, const T *a, int_t lda, const T *b, int_t ldb, T *c, int_t ldc,
           remove_complex_t<T> *scale) -> int_t;

template <>
inline auto trsyl<float>(char trana, char tranb, int_t isgn, int_t m, int_t n, const float *a, int_t lda, const float *b, int_t ldb,
                         float *c, int_t ldc, float *scale) -> int_t {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, int_t isgn, int_t m, int_t n, const double *a, int_t lda, const double *b, int_t ldb,
                          double *c, int_t ldc, double *scale) -> int_t {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<float> *a, int_t lda,
                                       const std::complex<float> *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, int_t isgn, int_t m, int_t n, const std::complex<double> *a, int_t lda,
                                        const std::complex<double> *b, int_t ldb, std::complex<double> *c, int_t ldc,
                                        double *scale) -> int_t {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

/*!
 * Computes a QR factorizaton (Useful for orthonormalizing matrices)
 */
namespace detail {
auto EINSUMS_EXPORT sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto EINSUMS_EXPORT dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto EINSUMS_EXPORT cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto EINSUMS_EXPORT zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;
} // namespace detail

template <typename T>
auto geqrf(int_t m, int_t n, T *a, int_t lda, T *tau) -> int_t;

template <>
inline auto geqrf<float>(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return detail::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return detail::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return detail::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return detail::zgeqrf(m, n, a, lda, tau);
}

/*!
 * Returns the orthogonal/unitary matrix Q from the output of dgeqrf
 */
namespace detail {
auto EINSUMS_EXPORT sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, const float *tau) -> int_t;
auto EINSUMS_EXPORT dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, const double *tau) -> int_t;
auto EINSUMS_EXPORT cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, const std::complex<float> *tau) -> int_t;
auto EINSUMS_EXPORT zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, const std::complex<double> *tau) -> int_t;
} // namespace detail

template <typename T>
auto orgqr(int_t m, int_t n, int_t k, T *a, int_t lda, const T *tau) -> int_t;

template <>
inline auto orgqr<float>(int_t m, int_t n, int_t k, float *a, int_t lda, const float *tau) -> int_t {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(int_t m, int_t n, int_t k, double *a, int_t lda, const double *tau) -> int_t {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(int_t m, int_t n, int_t k, T *a, int_t lda, const T *tau) -> int_t;

template <>
inline auto ungqr<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda,
                                       const std::complex<float> *tau) -> int_t {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda,
                                        const std::complex<double> *tau) -> int_t {
    return detail::zungqr(m, n, k, a, lda, tau);
}

} // namespace einsums::blas