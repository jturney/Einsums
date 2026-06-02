//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Concepts/Complex.hpp>

namespace einsums::blas {

/**
 * @class IsBlasable
 *
 * @brief Determines whether a type can be used in a BLAS call.
 *
 * This checks to see if a type is @c float or @c double or a complex type of these..
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
class IsBlasable {
  public:
    constexpr static bool value = false;
};
// Base instantiation.
template <>
struct IsBlasable<float> {
  public:
    constexpr static bool value = true;
};

template <>
struct IsBlasable<double> {
    constexpr static bool value = true;
};

// Complex instantiation.
template <>
struct IsBlasable<std::complex<float>> {
    constexpr static bool value = true;
};

template <>
struct IsBlasable<std::complex<double>> {
    constexpr static bool value = true;
};

/**
 * @property IsBlasable<T>::value
 *
 * The result of the test.
 *
 * @versionadded{1.1.0}
 *
 * @property IsBlasableV
 *
 * @brief Boolean wrapper of IsBlasable<T>.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
constexpr bool IsBlasableV = IsBlasable<T>::value;

/**
 * @concept Blasable
 *
 * @brief Concept version of IsBlasableV<T>.
 *
 * @tparam T The type to test.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
concept Blasable = IsBlasableV<T>;

namespace detail {
// These routines take care of actually making the call to the BLAS equivalent.
void EINSUMS_EXPORT sgemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                          int_t ldb, float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dgemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                          int_t ldb, double beta, double *c, int_t ldc);
void EINSUMS_EXPORT cgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                          int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);

void EINSUMS_EXPORT zgemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                          int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c,
                          int_t ldc);
} // namespace detail

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two matrices,
 * \f[
 * \mathbf{C} := \alpha \mathbf{A}\mathbf{B} + \beta\mathbf{C}
 * \f]
 * where @f$\mathbf{A}@f$, @f$\mathbf{B}@f$, and @f$\mathbf{C}@f$ are matrices, and
 * @f$\alpha@f$ and @f$\beta@f$ are scalar values.
 *
 * @tparam T The datatype of the GEMM.
 * @param[in] transa Whether to transpose matrix a :
 *   - 'N' or 'n' for no transpose,
 *   - 'T' or 't' for transpose,
 *   - 'C' or 'c' for conjugate transpose.
 * @param[in] transb Whether to transpose matrix b .
 * @param[in] m The number of rows in matrix A and C.
 * @param[in] n The number of columns in matrix B and C.
 * @param[in] k The number of columns in matrix A and rows in matrix B.
 * @param[in] alpha The scalar alpha.
 * @param[in] a A pointer to the matrix A with dimensions `(lda, k)` when transa is 'N' or 'n', and `(lda, m)`
 * otherwise.
 * @param[in] lda Leading dimension of A, specifying the distance between two consecutive columns.
 * @param[in] b A pointer to the matrix B with dimensions `(ldb, n)` when transB is 'N' or 'n', and `(ldb, k)`
 * otherwise.
 * @param[in] ldb Leading dimension of B, specifying the distance between two consecutive columns.
 * @param[in] beta The scalar beta.
 * @param[inout] c A pointer to the matrix C with dimensions `(ldc, n)`.
 * @param[in] ldc Leading dimension of C, specifying the distance between two consecutive columns.
 *
 * @note The function performs one of the following matrix operations:
 * - If transA is 'N' or 'n' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{AB} + \beta\mathbf{C}\f$
 * - If transA is 'N' or 'n' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}\mathbf{B}^T + \beta\mathbf{C}\f$
 * - If transA is 'T' or 't' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{A}^T\mathbf{B} + \beta\mathbf{C}\f$
 * - If transA is 'T' or 't' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}^T\mathbf{B}^T + \beta\mathbf{C}\f$
 * - If transA is 'C' or 'c' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{A}^H\mathbf{B} + \beta\mathbf{C}\f$
 * - If transA is 'C' or 'c' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}^H\mathbf{B}^Y + \beta\mathbf{C}\f$
 * - etc.
 *
 * @throws std::invalid_argument If @p transA or @p transB are invalid.
 * @throws std::domain_error If the values of @p m , @p n , or @p k are negative, or the values of @p lda , @p ldb , or @p ldc are
 * invalid.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c,
          int_t ldc);

// These are the template specialization for the data types we support. If an unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                        int_t ldb, float beta, float *c, int_t ldc) {
    detail::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                         int_t ldb, double beta, double *c, int_t ldc) {
    detail::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                      std::complex<float> const *a, int_t lda, std::complex<float> const *b, int_t ldb,
                                      std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    detail::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                       std::complex<double> const *a, int_t lda, std::complex<double> const *b, int_t ldb,
                                       std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    detail::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Perform a batch of independent GEMMs with uniform parameters.
 *
 * All batches share transa, transb, m, n, k, alpha, beta, lda, ldb, ldc.
 * The data pointers are passed as arrays.
 *
 * @tparam T Element type (float, double, complex<float>, complex<double>).
 */
template <typename T>
void gemm_batch(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const **a_array, int_t lda, T const **b_array, int_t ldb,
                T beta, T **c_array, int_t ldc, int_t batch_count);

namespace detail {
void EINSUMS_EXPORT sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda,
                                float const **b_array, int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count);
void EINSUMS_EXPORT dgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                                double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count);
void EINSUMS_EXPORT cgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                std::complex<float> const **a_array, int_t lda, std::complex<float> const **b_array, int_t ldb,
                                std::complex<float> beta, std::complex<float> **c_array, int_t ldc, int_t batch_count);
void EINSUMS_EXPORT zgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                std::complex<double> const **a_array, int_t lda, std::complex<double> const **b_array, int_t ldb,
                                std::complex<double> beta, std::complex<double> **c_array, int_t ldc, int_t batch_count);
} // namespace detail

template <>
inline void gemm_batch<float>(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda,
                              float const **b_array, int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count) {
    detail::sgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
template <>
inline void gemm_batch<double>(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                               double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count) {
    detail::dgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
template <>
inline void gemm_batch<std::complex<float>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                            std::complex<float> const **a_array, int_t lda, std::complex<float> const **b_array, int_t ldb,
                                            std::complex<float> beta, std::complex<float> **c_array, int_t ldc, int_t batch_count) {
    detail::cgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}
template <>
inline void gemm_batch<std::complex<double>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                             std::complex<double> const **a_array, int_t lda, std::complex<double> const **b_array,
                                             int_t ldb, std::complex<double> beta, std::complex<double> **c_array, int_t ldc,
                                             int_t batch_count) {
    detail::zgemm_batch(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

namespace detail {
void EINSUMS_EXPORT sgemv(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                          float *y, int_t incy);
void EINSUMS_EXPORT dgemv(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                          double *y, int_t incy);
void EINSUMS_EXPORT cgemv(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y, int_t incy);
void EINSUMS_EXPORT zgemv(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y, int_t incy);
} // namespace detail

/**
 * @brief Computes a matrix-vector product using a general matrix.
 *
 * The gemv routine performs a matrix-vector operation defined as:
 * @f[
 * \mathbf{y} := \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^T \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^H \mathbf{x} + \beta \mathbf{y}
 * @f]
 *
 * @tparam T the underlying data type of the matrix and vector
 * @param[in] transa what to do with \p a - no trans, trans, conjg
 * @param[in] m specifies the number of rows of \p a
 * @param[in] n specifies the number of columns of \p a
 * @param[in] alpha Specifies the scaler alpha
 * @param[in] a Array, size lda * m
 * @param[in] lda Specifies the leading dimension of \p a as declared in the calling function
 * @param[in] x array, vector x
 * @param[in] incx Specifies the increment for the elements of \p x
 * @param[in] beta Specifies the scalar beta. When beta is set to zero, then \p y need not be set on input.
 * @param[inout] y array, vector y
 * @param[in] incy Specifies the increment for the elements of \p y .
 *
 * @throws std::invalid_argument If @p transA is invalid.
 * @throws std::domain_error If the values of @p m or @p n are negative, the value of @p lda is invalid, or either @p incx or @p incy is
 * zero.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void gemv(char transa, int_t m, int_t n, T alpha, T const *a, int_t lda, T const *x, int_t incx, T beta, T *y, int_t incy);

template <>
inline void gemv<float>(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                        float *y, int_t incy) {
    detail::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                         double *y, int_t incy) {
    detail::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                                      std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y,
                                      int_t incy) {
    detail::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                                       std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y,
                                       int_t incy) {
    detail::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

namespace detail {
auto EINSUMS_EXPORT ssyev(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t;
auto EINSUMS_EXPORT dsyev(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t;
} // namespace detail

/**
 * @brief Performs diagonalization of a symmetrix matrix.
 *
 * The syev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose columns are
 * the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues of @f$
 * \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type the function will handle.
 * @param[in] job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param[in] lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
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

namespace detail {
auto EINSUMS_EXPORT ssterf(int_t n, float *d, float *e) -> int_t;
auto EINSUMS_EXPORT dsterf(int_t n, double *d, double *e) -> int_t;
} // namespace detail

/**
 * @brief Computes the eigenvalues of a symmetric tridiagonal matrix.
 * The sterf routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric tridiagonal matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose
 * columns are the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues
 * of @f$ \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements along the diagonal.
 * @param[inout] d The diagonal elements. On exit, it contains the eigenvalues.
 * @param[inout] e The off-diagonal elements. There is one fewer of these than the diagonal elements.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto sterf(int_t n, T *d, T *e) -> int_t;

template <>
inline auto sterf<float>(int_t n, float *d, float *e) -> int_t {
    return detail::ssterf(n, d, e);
}

template <>
inline auto sterf<double>(int_t n, double *d, double *e) -> int_t {
    return detail::dsterf(n, d, e);
}

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
/**
 * @brief Performs diagonalization of a matrix.
 *
 * The syev routine finds the matrices that satisfy the following equations.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1}
 * \mathbf{A}^T = \mathbf{L} \mathbf{\Lambda} \mathbf{L}^{-1}
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a matrix, @f$\mathbf{P}@f$ is a complex-valued matrix whose columns are
 * the right eigenvectors of @f$\mathbf{A}@f$, @f$\mathbf{L}@f$ is a complex-valued matrix whose columns are the left eigenvectors of
 * @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a complex-valued diagonal matrix, whose elements are the eigenvalues of @f$ A @f$. The
 * eigenvalues are stored in a vector form on exit. The eigenvectors are stored in a special way if the input is a real matrix. If the input
 * is a complex matrix, then the eigenvectors are stored plainly in the columns of the appropriate output matrices.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvl Whether to compute the left eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] jobvr Whether to compute the right eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[out] vl The left eigenvector output. If @p jobvl is 'n', then this is not referenced and may be null.
 * @param[in] ldvl The leading dimension of the left eigenvectors. Even if not referenced, it must be at least 1.
 * @param[out] vr The right eigenvector output. If @p jobvr is 'n', then this is not referenced and may be null.
 * @param[in] ldvr The leading dimension of the right eigenvectors. Even if not referenced, it must be at least 1.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto geev(char jobvl, char jobvr, int_t n, T *a, int_t lda, AddComplexT<T> *w, T *vl, int_t ldvl, T *vr, int_t ldvr) -> int_t;

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

#ifndef DOXGYEN
namespace detail {
auto EINSUMS_EXPORT cheev(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork,
                          float *rwork) -> int_t;
auto EINSUMS_EXPORT zheev(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                          int_t lwork, double *rworl) -> int_t;
} // namespace detail
#endif

/**
 * @brief Performs diagonalization of a Hermitian matrix.
 *
 * The heev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^H
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a Hermitian matrix, @f$\mathbf{P}@f$ is a unitary matrix whose columns are
 * the eigenvectors of @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a diagonal matrix, whose elements are the eigenvalues of
 * @f$\mathbf{A}@f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type this function handles.
 * @param[in] job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param[in] lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 * @param[inout] rwork A work array for real values.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{1.0.0}
 */
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

namespace detail {
auto EINSUMS_EXPORT sgesv(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dgesv(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cgesv(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT zgesv(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b, int_t ldb)
    -> int_t;
} // namespace detail

/**
 * @brief Solve a system of linear equations.
 *
 * Solves equations of the following form.
 * @f[
 * \mathbf{A}\mathbf{x} = \mathbf{B}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of rows and columns of @f$\mathbf{A}@f$ and rows @f$\mathbf{B}@f$.
 * @param[in] nrhs The number of columns of @f$\mathbf{B}@f$
 * @param[inout] a The coefficient matrix. On exit, it contains the LU decomposition of @p a, where the lower-triangle matrix has unit
 * diagonal entries, which are not stored.
 * @param[in] lda The leading dimension of @p a.
 * @param[out] ipiv A list of pivots used in the decomposition.
 * @param[inout] b The results matrix. On exit, it contains the values of @f$\mathbf{x}@f$ that satisfy the system of equations.
 * @param[in] ldb The leading dimension of @p b.
 *
 * @return 0 on success. If positive, then the matrix was singular. If negative, then a bad value was passed to the function.
 * The absolute value indicates which parameter was bad.
 *
 * @versionadded{1.0.0}
 */
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

void EINSUMS_EXPORT srscl(int_t n, float alpha, float *vec, int_t inc);
void EINSUMS_EXPORT drscl(int_t n, double alpha, double *vec, int_t inc);
void EINSUMS_EXPORT csrscl(int_t n, float alpha, std::complex<float> *vec, int_t inc);
void EINSUMS_EXPORT zdrscl(int_t n, double alpha, std::complex<double> *vec, int_t inc);
} // namespace detail

/**
 * @brief Scales a vector by a value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements to scale the vector by.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements of the vector.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void scal(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by a real value.
 *
 * @param[in] n The number of elements to scale the vector by.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements of the vector.
 *
 * @versionadded{1.0.0}
 */
template <Complex T>
void scal(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

template <>
inline void scal<float>(int_t n, float const alpha, float *vec, int_t inc) {
    detail::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(int_t n, double const alpha, double *vec, int_t inc) {
    detail::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    detail::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    detail::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    detail::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    detail::zdscal(n, alpha, vec, inc);
}

/**
 * @brief Scales a vector by the reciprocal of a value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The value to divide all the elements in the vector by.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void rscl(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by the reciprocal of a real value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The value to divide all the elements in the vector by.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <Complex T>
void rscl(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

template <>
inline void rscl<float>(int_t n, float const alpha, float *vec, int_t inc) {
    detail::srscl(n, alpha, vec, inc);
}

template <>
inline void rscl<double>(int_t n, double const alpha, double *vec, int_t inc) {
    detail::drscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    detail::cscal(n, std::complex<float>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    detail::zscal(n, std::complex<double>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    detail::csrscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    detail::zdrscl(n, alpha, vec, inc);
}

namespace detail {
auto EINSUMS_EXPORT sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float;
auto EINSUMS_EXPORT ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double;
auto EINSUMS_EXPORT cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float>;
auto EINSUMS_EXPORT zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double>;
auto EINSUMS_EXPORT cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float>;
auto EINSUMS_EXPORT zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double>;
} // namespace detail

/**
 * Computes the dot product of two vectors. For complex vectors it is the non-conjugated dot product;
 * (c|z)dotu in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param[in] n length of the vectors
 * @param[in] x first vector
 * @param[in] incx how many elements to skip in x
 * @param[in] y second vector
 * @param[in] incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto dot(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

template <>
inline auto dot<float>(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    return detail::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    return detail::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return detail::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return detail::zdot(n, x, incx, y, incy);
}

/**
 * Computes the dot product of two vectors. For complex vector it is the conjugated dot product;
 * (c|z)dotc in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param[in] n length of the vectors
 * @param[in] x first vector
 * @param[in] incx how many elements to skip in x
 * @param[in] y second vector
 * @param[in] incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto dotc(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

template <>
inline auto dotc<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return detail::cdotc(n, x, incx, y, incy);
}

template <>
inline auto dotc<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return detail::zdotc(n, x, incx, y, incy);
}

namespace detail {
void EINSUMS_EXPORT saxpy(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpy(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpy(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> *y,
                          int_t inc_y);
void EINSUMS_EXPORT zaxpy(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> *y,
                          int_t inc_y);
} // namespace detail

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \mathbf{y}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void axpy(int_t n, T alpha_x, T const *x, int_t inc_x, T *y, int_t inc_y);

template <>
inline void axpy<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    detail::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    detail::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> *y, int_t inc_y) {
    detail::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> *y, int_t inc_y) {
    detail::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT saxpby(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y);
void EINSUMS_EXPORT daxpby(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y);
void EINSUMS_EXPORT caxpby(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x, std::complex<float> b,
                           std::complex<float> *y, int_t inc_y);
void EINSUMS_EXPORT zaxpby(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x, std::complex<double> b,
                           std::complex<double> *y, int_t inc_y);
} // namespace detail

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \beta\mathbf{y}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param[in] b The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void axpby(int_t n, T alpha_x, T const *x, int_t inc_x, T b, T *y, int_t inc_y);

template <>
inline void axpby<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y) {
    detail::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y) {
    detail::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                       std::complex<float> b, std::complex<float> *y, int_t inc_y) {
    detail::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                        std::complex<double> b, std::complex<double> *y, int_t inc_y) {
    detail::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

namespace detail {
void EINSUMS_EXPORT sger(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda);
void EINSUMS_EXPORT dger(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda);
void EINSUMS_EXPORT cger(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                         std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
void EINSUMS_EXPORT zger(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                         std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);
void EINSUMS_EXPORT cgerc(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                          std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda);
void EINSUMS_EXPORT zgerc(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                          std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda);
} // namespace detail

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^T + \mathbf{A}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of entries in @p x.
 * @param[in] n The number of entries in @p y.
 * @param[in] alpha The scale factor for the outer product.
 * @param[in] x The left input vector.
 * @param[in] inc_x The skip value for the left input. May be negative to go in reverse.
 * @param[in] y The right input vector.
 * @param[in] inc_y The skip value for the right input. May be negative to go in reverse.
 * @param[inout] a The output matrix.
 * @param[in] lda The leading dimension of @p a.
 *
 * @throws std::domain_error If either of the dimension parameters are negative or the leading dimension of the matrix is less than the
 * number of columns.
 * @throws std::invalid_argument If either of the vector increments are zero.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void ger(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?gerc routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^H + \mathbf{A}
 * @f]
 *
 * @tparam T The type this function handle.
 * @param[in] m The number of entries in @p x.
 * @param[in] n The number of entries in @p y.
 * @param[in] alpha The scale factor for the outer product.
 * @param[in] x The left input vector.
 * @param[in] inc_x The skip value for the left input. May be negative to go in reverse.
 * @param[in] y The right input vector.
 * @param[in] inc_y The skip value for the right input. May be negative to go in reverse.
 * @param[inout] a The output matrix.
 * @param[in] lda The leading dimension of @p a.
 *
 * @throws std::domain_error If either of the dimension parameters are negative or the leading dimension of the matrix is less than the
 * number of columns.
 * @throws std::invalid_argument If either of the vector increments are zero.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void gerc(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

template <>
inline void ger<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                     std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    detail::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                      std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    detail::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    detail::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    detail::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    detail::cgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    detail::zgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
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
 * @f[
 *   \mathbf{A} = \mathbf{PLU}
 * @f]
 * where @f$\mathbf{P}@f$ is a permutation matrix, @f$\mathbf{L}@f$ is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and @f$\mathbf{U}@f$ is upper
 * triangular (upper trapezoidal if m < n).
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input.
 * @param[in] n The number of columns in the input.
 * @param[inout] a The input matrix. On exit, it contains the upper and lower triangular matrices. The elemnts of the lower
 * triangular matrix are not stored since they are all 1.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] ipiv The list of pivots.
 *
 * @return 0 on success. If positive, the matrix is singular and the result should not be used for solving systems of equations.
 * The decomposition was performed, though. If negative, one of the inputs had an invalid value. The absolute value indicates
 * which input it is.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto getrf(int_t m, int_t n, T *a, int_t lda, int_t *ipiv) -> int_t;

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
auto EINSUMS_EXPORT sgetri(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT dgetri(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT cgetri(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t;
auto EINSUMS_EXPORT zgetri(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t;
} // namespace detail

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of rows and columns of the matrix.
 * @param[inout] a The input matrix after being processed by getrf.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] ipiv The pivots from getrf.
 *
 * @return 0 on success. If positive, the matrix is singular and an inverse could not be computed. If negative,
 * one of the inputs is invalid, and the absolute value indicates which input is bad.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto getri(int_t n, T *a, int_t lda, int_t const *ipiv) -> int_t;

template <>
inline auto getri<float>(int_t n, float *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::sgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<double>(int_t n, double *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::dgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<float>>(int_t n, std::complex<float> *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::cgetri(n, a, lda, ipiv);
}

template <>
inline auto getri<std::complex<double>>(int_t n, std::complex<double> *a, int_t lda, int_t const *ipiv) -> int_t {
    return detail::zgetri(n, a, lda, ipiv);
}

namespace detail {
auto EINSUMS_EXPORT slange(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT dlange(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double;
auto EINSUMS_EXPORT clange(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float;
auto EINSUMS_EXPORT zlange(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work) -> double;
} // namespace detail

/**
 * Computes various matrix norms. The available norms are the 1-norm, Frobenius norm, Max-abs norm, and the infinity norm.
 *
 * @tparam T The type this matrix handles.
 * @param[in] norm_type The norm to compute. It is case insensitive. For the 1-norm, it should be '1' or 'o'. For the Frobenius norm it
 * should be 'f' or 'e'. For the max-abs norm it should be 'm'. For the infinity norm, it should be 'i'.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[in] A The matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[inout] work A work array. Only needed for the infinity norm.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto lange(char norm_type, int_t m, int_t n, T const *A, int_t lda, RemoveComplexT<T> *work) -> RemoveComplexT<T>;

template <>
inline auto lange<float>(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float {
    return detail::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double {
    return detail::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float {
    return detail::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work)
    -> double {
    return detail::zlange(norm_type, m, n, A, lda, work);
}

namespace detail {
void EINSUMS_EXPORT   slassq(int_t n, float const *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT   dlassq(int_t n, double const *x, int_t incx, double *scale, double *sumsq);
void EINSUMS_EXPORT   classq(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq);
void EINSUMS_EXPORT   zlassq(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq);
float EINSUMS_EXPORT  snrm2(int_t n, float const *x, int_t incx);
double EINSUMS_EXPORT dnrm2(int_t n, double const *x, int_t incx);
float EINSUMS_EXPORT  scnrm2(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dznrm2(int_t n, std::complex<double> const *x, int_t incx);
} // namespace detail

/**
 * Compute the sum of the squares of the input vector without roundoff error.
 * @f[
 * scale^2 sumsq := \left|\mathbf{x}\right|^2 + scale^2 sumsq
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 * @param[inout] scale The scale value used to avoid overflow/underflow. It is also used as an input to continue a previous calculation.
 * @param[inout] sumsq The result of the operation, scaled to avoid overflow/underflow. It is also used as an input to continue a previous
 * calculation.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
void lassq(int_t n, T const *x, int_t incx, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq);

template <>
inline void lassq<float>(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    detail::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    detail::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    detail::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    detail::zlassq(n, x, incx, scale, sumsq);
}

/**
 * Compute the Euclidean norm of a vector.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 *
 * @return The Euclidean norm of the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> nrm2(int_t n, T const *x, int_t incx);

template <>
inline float nrm2<float>(int_t n, float const *x, int_t incx) {
    return detail::snrm2(n, x, incx);
}

template <>
inline double nrm2<double>(int_t n, double const *x, int_t incx) {
    return detail::dnrm2(n, x, incx);
}

template <>
inline float nrm2<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scnrm2(n, x, incx);
}

template <>
inline double nrm2<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dznrm2(n, x, incx);
}

namespace detail {
auto EINSUMS_EXPORT sgesdd(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT dgesdd(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt)
    -> int_t;
auto EINSUMS_EXPORT cgesdd(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u, int_t ldu,
                           std::complex<float> *vt, int_t ldvt) -> int_t;
auto EINSUMS_EXPORT zgesdd(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u, int_t ldu,
                           std::complex<double> *vt, int_t ldvt) -> int_t;
} // namespace detail

/**
 * Performs singular value decomposition for a matrix using the divide and conquer algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] jobz What computation to do. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns of the input matrix.
 * @param[inout] a The input matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values output.
 * @param[out] u The U matrix from the singular value decomposition.
 * @param[in] ldu The leading dimension of U.
 * @param[out] vt The transpose of the V matrix from the singular value decomposition.
 * @param[in] ldvt The leading dimension of the transpose of the V matrix.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If -4, then the input matrix had a NaN entry. If negative otherwise,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gesdd(char jobz, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt) -> int_t;

template <>
inline auto gesdd<float>(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    return detail::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt)
    -> int_t {
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
auto EINSUMS_EXPORT cgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u,
                           int_t ldu, std::complex<float> *vt, int_t ldvt, std::complex<float> *superb) -> int_t;
auto EINSUMS_EXPORT zgesvd(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u,
                           int_t ldu, std::complex<double> *vt, int_t ldvt, std::complex<double> *superb) -> int_t;
} // namespace detail

/**
 * Performs singular value decomposition for a matrix using the QR algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] jobu Whether to compute the U matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] jobvt Whether to compute the transpose of the V matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns of the input matrix.
 * @param[inout] a The input matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values output.
 * @param[out] u The U matrix from the singular value decomposition.
 * @param[in] ldu The leading dimension of U.
 * @param[out] vt The transpose of the V matrix from the singular value decomposition.
 * @param[in] ldvt The leading dimension of the transpose of the V matrix.
 * @param[inout] superb Temporary storage area for intermediates in the computation.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If negative,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gesvd(char jobu, char jobvt, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt, T *superb);

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

template <>
inline auto gesvd<std::complex<float>>(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s,
                                       std::complex<float> *u, int_t ldu, std::complex<float> *vt, int_t ldvt,
                                       std::complex<float> *superb) {
    return detail::cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<std::complex<double>>(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s,
                                        std::complex<double> *u, int_t ldu, std::complex<double> *vt, int_t ldvt,
                                        std::complex<double> *superb) {
    return detail::zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

namespace detail {
auto EINSUMS_EXPORT sgees(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT dgees(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT cgees(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w,
                          std::complex<float> *vs, int_t ldvs) -> int_t;
auto EINSUMS_EXPORT zgees(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w,
                          std::complex<double> *vs, int_t ldvs) -> int_t;
} // namespace detail

/**
 * Computes the Schur decomposition of a matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvs Whether to compute the unitary matrix for the decomposition.
 * @param[in] n The number of rows and columns of the input matrix.
 * @param[inout] a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param[in] lda The leading dimension of A.
 * @param[in] sdim The number of selected eigenvalues.
 * @param[out] wr The real components of the eigenvalues.
 * @param[out] wi The imaginary components of the eigenvaules.
 * @param[out] vs The Schur vector matrix.
 * @param[in] ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{1.0.0}
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

/**
 * Computes the Schur decomposition of a matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvs Whether to compute the unitary matrix for the decomposition.
 * @param[in] n The number of rows and columns of the input matrix.
 * @param[inout] a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param[in] lda The leading dimension of A.
 * @param[in] sdim The number of selected eigenvalues.
 * @param[out] w The  eigenvalues.
 * @param[out] vs The Schur vector matrix.
 * @param[in] ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *w, T *vs, int_t ldvs) -> int_t;

template <>
inline auto gees<std::complex<float>>(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w,
                                      std::complex<float> *vs, int_t ldvs) -> int_t {
    return detail::cgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

template <>
inline auto gees<std::complex<double>>(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w,
                                       std::complex<double> *vs, int_t ldvs) -> int_t {
    return detail::zgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

namespace detail {
auto EINSUMS_EXPORT strsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb,
                           float *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT dtrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb,
                           double *c, int_t ldc, double *scale) -> int_t;
auto EINSUMS_EXPORT ctrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t;
auto EINSUMS_EXPORT ztrsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale) -> int_t;
} // namespace detail

/**
 * Solves a Sylvester equation. These equations look like the following.
 * @f[
 *  \mathbf{A}\mathbf{X} \pm \mathbf{X}\mathbf{B} = \alpha\mathbf{C}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] trana Whether to transpose the A matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] tranb Whether to transpose the B matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] isgn Whether the sign in the equation is positive or negative.
 * @param[in] m The number of rows in X.
 * @param[in] n The number of columns in X.
 * @param[in] a The A matrix in Schur canonical form.
 * @param[in] lda The leading dimension of A.
 * @param[in] b The B matrix in Schur canonical form.
 * @param[in] ldb The leading dimension of B.
 * @param[inout] c The right hand side matrix. On exit, it contains the value of the X matrix that satisfies the equation.
 * @param[in] ldc The leading dimension of the C matrix.
 * @param[out] scale The scale factor for the right hand side matrix.
 *
 * @return 0 on success. If 1, then some eigenvalues were close and needed to be perturbed. If negative, then one of the inputs
 * had a bad value, and the absolute value of the return gives the parameter.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, T const *a, int_t lda, T const *b, int_t ldb, T *c, int_t ldc,
           RemoveComplexT<T> *scale) -> int_t;

template <>
inline auto trsyl<float>(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb,
                         float *c, int_t ldc, float *scale) -> int_t {
    return detail::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb,
                          double *c, int_t ldc, double *scale) -> int_t {
    return detail::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda,
                                       std::complex<float> const *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
    return detail::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda,
                                        std::complex<double> const *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale)
    -> int_t {
    return detail::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

namespace detail {
auto EINSUMS_EXPORT sgeqrf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto EINSUMS_EXPORT dgeqrf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto EINSUMS_EXPORT cgeqrf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto EINSUMS_EXPORT zgeqrf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;
} // namespace detail

/**
 * Set up for computing the QR decomposition of a matrix.
 *
 * @f[
 * \mathbf{A} = \mathbf{QR}
 * @f]
 *
 * Here, @f$\mathbf{Q}@f$ is an orthogonal matrix and @f$\mathbf{R}@f$ is an upper triangular matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The input matrix. On exit, contains the data needed to compute the Q and R matrices. The
 * entries on and above the diagonal are the entries of the R matrix. The rest is needed to find the Q matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] tau On exit, holds the Householder reflector parameters for computing the Q matrix.
 *
 * @return 0 on success. If negative, one of the inputs had a bad value, and the absolute value of the return
 * tells you which one it was.
 *
 * @versionadded{1.0.0}
 */
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

namespace detail {
auto EINSUMS_EXPORT sorgqr(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
auto EINSUMS_EXPORT dorgqr(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
auto EINSUMS_EXPORT cungqr(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
auto EINSUMS_EXPORT zungqr(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;

auto EINSUMS_EXPORT sormqr(char side, char trans, int_t m, int_t n, int_t k, float const *a, int_t lda, float const *tau, float *c,
                           int_t ldc) -> int_t;
auto EINSUMS_EXPORT dormqr(char side, char trans, int_t m, int_t n, int_t k, double const *a, int_t lda, double const *tau, double *c,
                           int_t ldc) -> int_t;
auto EINSUMS_EXPORT cunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *tau, std::complex<float> *c, int_t ldc) -> int_t;
auto EINSUMS_EXPORT zunmqr(char side, char trans, int_t m, int_t n, int_t k, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *tau, std::complex<double> *c, int_t ldc) -> int_t;
} // namespace detail

/**
 * Extract the Q matrix after a call to geqrf.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used in the calculation.
 * @param[inout] a The input matrix after being processed by geqrf.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[in] tau The scales for the elementary reflectors from geqrf.
 *
 * @return 0 on success. If negative, then one of the inputs had an invalid value, and the absolute value indicates
 * which parameter it is.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto orgqr(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto orgqr<float>(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    return detail::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    return detail::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto ungqr<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau)
    -> int_t {
    return detail::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau)
    -> int_t {
    return detail::zungqr(m, n, k, a, lda, tau);
}

/**
 * @brief Multiply a matrix by the orthogonal/unitary Q from geqrf without forming Q explicitly.
 *
 * Computes C := op(Q) * C (side='L') or C := C * op(Q) (side='R').
 * For real types: op(Q) = Q ('N') or Q^T ('T').
 * For complex types: op(Q) = Q ('N') or Q^H ('C').
 *
 * @param side 'L' or 'R'.
 * @param trans 'N', 'T' (real), or 'C' (complex).
 * @param m Rows of C.
 * @param n Columns of C.
 * @param k Number of reflectors from geqrf.
 * @param a Householder vectors from geqrf.
 * @param lda Leading dimension of a.
 * @param tau Scalar factors from geqrf.
 * @param c The m×n matrix C, overwritten on exit.
 * @param ldc Leading dimension of c.
 */
template <typename T>
auto ormqr(char side, char trans, int_t m, int_t n, int_t k, T const *a, int_t lda, T const *tau, T *c, int_t ldc) -> int_t;

template <>
inline auto ormqr<float>(char side, char trans, int_t m, int_t n, int_t k, float const *a, int_t lda, float const *tau, float *c, int_t ldc)
    -> int_t {
    return detail::sormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

template <>
inline auto ormqr<double>(char side, char trans, int_t m, int_t n, int_t k, double const *a, int_t lda, double const *tau, double *c,
                          int_t ldc) -> int_t {
    return detail::dormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

template <typename T>
auto unmqr(char side, char trans, int_t m, int_t n, int_t k, T const *a, int_t lda, T const *tau, T *c, int_t ldc) -> int_t;

template <>
inline auto unmqr<std::complex<float>>(char side, char trans, int_t m, int_t n, int_t k, std::complex<float> const *a, int_t lda,
                                       std::complex<float> const *tau, std::complex<float> *c, int_t ldc) -> int_t {
    return detail::cunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

template <>
inline auto unmqr<std::complex<double>>(char side, char trans, int_t m, int_t n, int_t k, std::complex<double> const *a, int_t lda,
                                        std::complex<double> const *tau, std::complex<double> *c, int_t ldc) -> int_t {
    return detail::zunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

namespace detail {
auto EINSUMS_EXPORT sgelqf(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t;
auto EINSUMS_EXPORT dgelqf(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t;
auto EINSUMS_EXPORT cgelqf(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t;
auto EINSUMS_EXPORT zgelqf(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t;
} // namespace detail

/**
 * Set up for computing the LQ decomposition of a matrix.
 *
 * @f[
 * \mathbf{A} = \mathbf{LQ}
 * @f]
 *
 * Here, @f$\mathbf{Q}@f$ is an orthogonal matrix and @f$\mathbf{L}@f$ is a lower triangular matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The input matrix. On exit, contains the data needed to compute the Q and L matrices. The
 * entries on and below the diagonal are the entries of the L matrix. The rest is needed to find the Q matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] tau On exit, holds the Householder reflector parameters for computing the Q matrix.
 *
 * @return 0 on success. If negative, one of the inputs had a bad value, and the absolute value of the return
 * tells you which one it was.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto gelqf(int_t m, int_t n, T *a, int_t lda, T *tau) -> int_t;

template <>
inline auto gelqf<float>(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return detail::sgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<double>(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return detail::dgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return detail::cgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return detail::zgelqf(m, n, a, lda, tau);
}

namespace detail {
auto EINSUMS_EXPORT sorglq(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t;
auto EINSUMS_EXPORT dorglq(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t;
auto EINSUMS_EXPORT cunglq(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau) -> int_t;
auto EINSUMS_EXPORT zunglq(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau) -> int_t;
} // namespace detail

/**
 * Extract the Q matrix after a call to gelqf.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used in the calculation.
 * @param[inout] a The input matrix after being processed by gelqf.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[in] tau The scales for the elementary reflectors from gelqf.
 *
 * @return 0 on success. If negative, then one of the inputs had an invalid value, and the absolute value indicates
 * which parameter it is.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto orglq(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto orglq<float>(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    return detail::sorglq(m, n, k, a, lda, tau);
}

template <>
inline auto orglq<double>(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    return detail::dorglq(m, n, k, a, lda, tau);
}

template <typename T>
auto unglq(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto unglq<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau)
    -> int_t {
    return detail::cunglq(m, n, k, a, lda, tau);
}

template <>
inline auto unglq<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau)
    -> int_t {
    return detail::zunglq(m, n, k, a, lda, tau);
}

namespace detail {
void EINSUMS_EXPORT scopy(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y);
void EINSUMS_EXPORT dcopy(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y);
void EINSUMS_EXPORT ccopy(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y);
void EINSUMS_EXPORT zcopy(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y);
} // namespace detail

/**
 * Copy data from one vector to another.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements to copy.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. If negative, the vector is traversed backwards. If zero, the values are broadcast
 * to the output vector.
 * @param[out] y The output vector.
 * @param[in] inc_y The skip value for the output vector. If negative, the vector is traversed backwards.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void copy(int_t n, T const *x, int_t inc_x, T *y, int_t inc_y);

template <>
inline void copy<float>(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    detail::scopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<double>(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    detail::dcopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<float>>(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    detail::ccopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<double>>(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    detail::zcopy(n, x, inc_x, y, inc_y);
}

namespace detail {
int_t EINSUMS_EXPORT slascl(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda);
int_t EINSUMS_EXPORT dlascl(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda);
} // namespace detail

/**
 * Scales a general matrix. The scale factor is <tt> cto / cfrom </tt>, but the scale is performed without overflow/underflow.
 *
 * @tparam T The type this function handles.
 * @param[in] type The type of matrix. Case insensitive. 'g' is for general matrices, 'l' is for lower triangular matrices, 'u' if for upper
 * triangular matrices, 'h' is for hessenberg matrices, 'b' is for symmetric band matrices with lower bandwidth @p kl and upper bandwidth of
 * @p ku and with only the lower half stored, 'q' is the same as 'b' but with the upper half stored instead, and 'z' is the same as 'b' but
 * with a more complicated storage scheme.
 * @param[in] kl The lower bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param[in] ku The upper bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param[in] cfrom The denominator for the scale.
 * @param[in] cto The numerator for the scale.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[inout] A The matrix being scaled.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value of the return indicates which
 * parameter it was.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
int_t lascl(char type, int_t kl, int_t ku, T cfrom, T cto, int_t m, int_t n, T *A, int_t lda);

template <>
inline int_t lascl<float>(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda) {
    return detail::slascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}

template <>
inline int_t lascl<double>(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda) {
    return detail::dlascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}

namespace detail {
void EINSUMS_EXPORT sdirprod(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz);
void EINSUMS_EXPORT ddirprod(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz);
void EINSUMS_EXPORT cdirprod(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx, std::complex<float> const *y,
                             int_t incy, std::complex<float> *z, int_t incz);
void EINSUMS_EXPORT zdirprod(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx, std::complex<double> const *y,
                             int_t incy, std::complex<double> *z, int_t incz);
} // namespace detail

/**
 * Computes the direct product between two vectors.
 *
 * @f[
 * z_i := z_i + \alpha x_i y_i
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The first input vector.
 * @param[in] incx The skip value for the first vector.
 * @param[in] y The second input vector.
 * @param[in] incy The skip value for the second vector.
 * @param[inout] z The accumulation vector.
 * @param[in] incz The skip value for the accumulation vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void dirprod(int_t n, T alpha, T const *x, int_t incx, T const *y, int_t incy, T *z, int_t incz);

template <>
inline void dirprod<float>(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    detail::sdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<double>(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    detail::ddirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<float>>(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx,
                                         std::complex<float> const *y, int_t incy, std::complex<float> *z, int_t incz) {
    detail::cdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<double>>(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx,
                                          std::complex<double> const *y, int_t incy, std::complex<double> *z, int_t incz) {
    detail::zdirprod(n, alpha, x, incx, y, incy, z, incz);
}

namespace detail {
float EINSUMS_EXPORT  sasum(int_t n, float const *x, int_t incx);
double EINSUMS_EXPORT dasum(int_t n, double const *x, int_t incx);
float EINSUMS_EXPORT  scasum(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dzasum(int_t n, std::complex<double> const *x, int_t incx);
float EINSUMS_EXPORT  scsum1(int_t n, std::complex<float> const *x, int_t incx);
double EINSUMS_EXPORT dzsum1(int_t n, std::complex<double> const *x, int_t incx);
} // namespace detail

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the absolute values of the components, not the magnitudes.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> asum(int_t n, T const *x, int_t incx);

template <>
inline float asum(int_t n, float const *x, int_t incx) {
    return detail::sasum(n, x, incx);
}

template <>
inline double asum(int_t n, double const *x, int_t incx) {
    return detail::dasum(n, x, incx);
}

template <>
inline float asum(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scasum(n, x, incx);
}

template <>
inline double asum(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dzasum(n, x, incx);
}

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the magnitudes.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> sum1(int_t n, T const *x, int_t incx);

template <>
inline float sum1(int_t n, float const *x, int_t incx) {
    return detail::sasum(n, x, incx);
}

template <>
inline double sum1(int_t n, double const *x, int_t incx) {
    return detail::dasum(n, x, incx);
}

template <>
inline float sum1(int_t n, std::complex<float> const *x, int_t incx) {
    return detail::scsum1(n, x, incx);
}

template <>
inline double sum1(int_t n, std::complex<double> const *x, int_t incx) {
    return detail::dzsum1(n, x, incx);
}

namespace detail {
void EINSUMS_EXPORT clacgv(int_t n, std::complex<float> *x, int_t incx);
void EINSUMS_EXPORT zlacgv(int_t n, std::complex<double> *x, int_t incx);
} // namespace detail

/**
 * Take the conjugate of a vector. Does nothing if the vector is real.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void lacgv(int_t n, T *x, int_t incx);

template <>
inline void lacgv<float>(int_t n, float *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<double>(int_t n, double *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<std::complex<float>>(int_t n, std::complex<float> *x, int_t incx) {
    detail::clacgv(n, x, incx);
}

template <>
inline void lacgv<std::complex<double>>(int_t n, std::complex<double> *x, int_t incx) {
    detail::zlacgv(n, x, incx);
}

// ---------------------------------------------------------------------------
// trsm
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT strsm(char side, char uplo, char transa, char diag, int_t m, int_t n, float alpha, float const *a, int_t lda, float *b,
                          int_t ldb);
void EINSUMS_EXPORT dtrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, double alpha, double const *a, int_t lda,
                          double *b, int_t ldb);
void EINSUMS_EXPORT ctrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<float> alpha,
                          std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb);
void EINSUMS_EXPORT ztrsm(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<double> alpha,
                          std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb);
} // namespace detail

template <typename T>
void trsm(char side, char uplo, char transa, char diag, int_t m, int_t n, T alpha, T const *a, int_t lda, T *b, int_t ldb);

template <>
inline void trsm<float>(char side, char uplo, char transa, char diag, int_t m, int_t n, float alpha, float const *a, int_t lda, float *b,
                        int_t ldb) {
    detail::strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

template <>
inline void trsm<double>(char side, char uplo, char transa, char diag, int_t m, int_t n, double alpha, double const *a, int_t lda,
                         double *b, int_t ldb) {
    detail::dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

template <>
inline void trsm<std::complex<float>>(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<float> alpha,
                                      std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb) {
    detail::ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

template <>
inline void trsm<std::complex<double>>(char side, char uplo, char transa, char diag, int_t m, int_t n, std::complex<double> alpha,
                                       std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb) {
    detail::ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------
// potrf
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT spotrf(char uplo, int_t n, float *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT dpotrf(char uplo, int_t n, double *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT cpotrf(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT zpotrf(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t;
} // namespace detail

template <typename T>
auto potrf(char uplo, int_t n, T *a, int_t lda) -> int_t;

template <>
inline auto potrf<float>(char uplo, int_t n, float *a, int_t lda) -> int_t {
    return detail::spotrf(uplo, n, a, lda);
}

template <>
inline auto potrf<double>(char uplo, int_t n, double *a, int_t lda) -> int_t {
    return detail::dpotrf(uplo, n, a, lda);
}

template <>
inline auto potrf<std::complex<float>>(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    return detail::cpotrf(uplo, n, a, lda);
}

template <>
inline auto potrf<std::complex<double>>(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    return detail::zpotrf(uplo, n, a, lda);
}

// ---------------------------------------------------------------------------
// potrs
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT spotrs(char uplo, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dpotrs(char uplo, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cpotrs(char uplo, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b, int_t ldb)
    -> int_t;
auto EINSUMS_EXPORT zpotrs(char uplo, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b, int_t ldb)
    -> int_t;
} // namespace detail

template <typename T>
auto potrs(char uplo, int_t n, int_t nrhs, T const *a, int_t lda, T *b, int_t ldb) -> int_t;

template <>
inline auto potrs<float>(char uplo, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    return detail::spotrs(uplo, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto potrs<double>(char uplo, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb) -> int_t {
    return detail::dpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto potrs<std::complex<float>>(char uplo, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, std::complex<float> *b,
                                       int_t ldb) -> int_t {
    return detail::cpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto potrs<std::complex<double>>(char uplo, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, std::complex<double> *b,
                                        int_t ldb) -> int_t {
    return detail::zpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------
// potri
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT spotri(char uplo, int_t n, float *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT dpotri(char uplo, int_t n, double *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT cpotri(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT zpotri(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t;
} // namespace detail

template <typename T>
auto potri(char uplo, int_t n, T *a, int_t lda) -> int_t;

template <>
inline auto potri<float>(char uplo, int_t n, float *a, int_t lda) -> int_t {
    return detail::spotri(uplo, n, a, lda);
}

template <>
inline auto potri<double>(char uplo, int_t n, double *a, int_t lda) -> int_t {
    return detail::dpotri(uplo, n, a, lda);
}

template <>
inline auto potri<std::complex<float>>(char uplo, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    return detail::cpotri(uplo, n, a, lda);
}

template <>
inline auto potri<std::complex<double>>(char uplo, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    return detail::zpotri(uplo, n, a, lda);
}

// ---------------------------------------------------------------------------
// syrk
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT ssyrk(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dsyrk(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double beta, double *c,
                          int_t ldc);
void EINSUMS_EXPORT csyrk(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zsyrk(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> beta, std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void syrk(char uplo, char trans, int_t n, int_t k, T alpha, T const *a, int_t lda, T beta, T *c, int_t ldc);

template <>
inline void syrk<float>(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float beta, float *c, int_t ldc) {
    detail::ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
inline void syrk<double>(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double beta, double *c,
                         int_t ldc) {
    detail::dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
inline void syrk<std::complex<float>>(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                                      int_t lda, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    detail::csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
inline void syrk<std::complex<double>>(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                                       int_t lda, std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    detail::zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// herk
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT cherk(char uplo, char trans, int_t n, int_t k, float alpha, std::complex<float> const *a, int_t lda, float beta,
                          std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zherk(char uplo, char trans, int_t n, int_t k, double alpha, std::complex<double> const *a, int_t lda, double beta,
                          std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void herk(char uplo, char trans, int_t n, int_t k, RemoveComplexT<T> alpha, T const *a, int_t lda, RemoveComplexT<T> beta, T *c, int_t ldc);

template <>
inline void herk<std::complex<float>>(char uplo, char trans, int_t n, int_t k, float alpha, std::complex<float> const *a, int_t lda,
                                      float beta, std::complex<float> *c, int_t ldc) {
    detail::cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
inline void herk<std::complex<double>>(char uplo, char trans, int_t n, int_t k, double alpha, std::complex<double> const *a, int_t lda,
                                       double beta, std::complex<double> *c, int_t ldc) {
    detail::zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// symm
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT ssymm(char side, char uplo, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                          float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dsymm(char side, char uplo, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                          double beta, double *c, int_t ldc);
void EINSUMS_EXPORT csymm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zsymm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void symm(char side, char uplo, int_t m, int_t n, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c, int_t ldc);

template <>
inline void symm<float>(char side, char uplo, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                        float beta, float *c, int_t ldc) {
    detail::ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void symm<double>(char side, char uplo, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                         double beta, double *c, int_t ldc) {
    detail::dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void symm<std::complex<float>>(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a,
                                      int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c,
                                      int_t ldc) {
    detail::csymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void symm<std::complex<double>>(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a,
                                       int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta,
                                       std::complex<double> *c, int_t ldc) {
    detail::zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// hemm
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT chemm(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                          std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zhemm(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                          std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void hemm(char side, char uplo, int_t m, int_t n, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c, int_t ldc);

template <>
inline void hemm<std::complex<float>>(char side, char uplo, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a,
                                      int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c,
                                      int_t ldc) {
    detail::chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void hemm<std::complex<double>>(char side, char uplo, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a,
                                       int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta,
                                       std::complex<double> *c, int_t ldc) {
    detail::zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// sygv
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT ssygv(int_t itype, char jobz, char uplo, int_t n, float *a, int_t lda, float *b, int_t ldb, float *w) -> int_t;
auto EINSUMS_EXPORT dsygv(int_t itype, char jobz, char uplo, int_t n, double *a, int_t lda, double *b, int_t ldb, double *w) -> int_t;
} // namespace detail

template <typename T>
auto sygv(int_t itype, char jobz, char uplo, int_t n, T *a, int_t lda, T *b, int_t ldb, T *w) -> int_t;

template <>
inline auto sygv<float>(int_t itype, char jobz, char uplo, int_t n, float *a, int_t lda, float *b, int_t ldb, float *w) -> int_t {
    return detail::ssygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

template <>
inline auto sygv<double>(int_t itype, char jobz, char uplo, int_t n, double *a, int_t lda, double *b, int_t ldb, double *w) -> int_t {
    return detail::dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

// ---------------------------------------------------------------------------
// hegv
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT chegv(int_t itype, char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb,
                          float *w) -> int_t;
auto EINSUMS_EXPORT zhegv(int_t itype, char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *b,
                          int_t ldb, double *w) -> int_t;
} // namespace detail

template <typename T>
auto hegv(int_t itype, char jobz, char uplo, int_t n, std::complex<T> *a, int_t lda, std::complex<T> *b, int_t ldb, T *w) -> int_t;

template <>
inline auto hegv<float>(int_t itype, char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb,
                        float *w) -> int_t {
    return detail::chegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

template <>
inline auto hegv<double>(int_t itype, char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb,
                         double *w) -> int_t {
    return detail::zhegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

// ---------------------------------------------------------------------------
// syevd
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT ssyevd(char jobz, char uplo, int_t n, float *a, int_t lda, float *w) -> int_t;
auto EINSUMS_EXPORT dsyevd(char jobz, char uplo, int_t n, double *a, int_t lda, double *w) -> int_t;
} // namespace detail

template <typename T>
auto syevd(char jobz, char uplo, int_t n, T *a, int_t lda, T *w) -> int_t;

template <>
inline auto syevd<float>(char jobz, char uplo, int_t n, float *a, int_t lda, float *w) -> int_t {
    return detail::ssyevd(jobz, uplo, n, a, lda, w);
}

template <>
inline auto syevd<double>(char jobz, char uplo, int_t n, double *a, int_t lda, double *w) -> int_t {
    return detail::dsyevd(jobz, uplo, n, a, lda, w);
}

// ---------------------------------------------------------------------------
// heevd
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT cheevd(char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w) -> int_t;
auto EINSUMS_EXPORT zheevd(char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w) -> int_t;
} // namespace detail

template <typename T>
auto heevd(char jobz, char uplo, int_t n, std::complex<T> *a, int_t lda, T *w) -> int_t;

template <>
inline auto heevd<float>(char jobz, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w) -> int_t {
    return detail::cheevd(jobz, uplo, n, a, lda, w);
}

template <>
inline auto heevd<double>(char jobz, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w) -> int_t {
    return detail::zheevd(jobz, uplo, n, a, lda, w);
}

// ---------------------------------------------------------------------------
// getrs
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT sgetrs(char trans, int_t n, int_t nrhs, float const *a, int_t lda, int_t const *ipiv, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dgetrs(char trans, int_t n, int_t nrhs, double const *a, int_t lda, int_t const *ipiv, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cgetrs(char trans, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, int_t const *ipiv,
                           std::complex<float> *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT zgetrs(char trans, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, int_t const *ipiv,
                           std::complex<double> *b, int_t ldb) -> int_t;
} // namespace detail

template <typename T>
auto getrs(char trans, int_t n, int_t nrhs, T const *a, int_t lda, int_t const *ipiv, T *b, int_t ldb) -> int_t;

template <>
inline auto getrs<float>(char trans, int_t n, int_t nrhs, float const *a, int_t lda, int_t const *ipiv, float *b, int_t ldb) -> int_t {
    return detail::sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto getrs<double>(char trans, int_t n, int_t nrhs, double const *a, int_t lda, int_t const *ipiv, double *b, int_t ldb) -> int_t {
    return detail::dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto getrs<std::complex<float>>(char trans, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda, int_t const *ipiv,
                                       std::complex<float> *b, int_t ldb) -> int_t {
    return detail::cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline auto getrs<std::complex<double>>(char trans, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda, int_t const *ipiv,
                                        std::complex<double> *b, int_t ldb) -> int_t {
    return detail::zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

// ---------------------------------------------------------------------------
// gels
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT sgels(char trans, int_t m, int_t n, int_t nrhs, float *a, int_t lda, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dgels(char trans, int_t m, int_t n, int_t nrhs, double *a, int_t lda, double *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT cgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<float> *a, int_t lda, std::complex<float> *b, int_t ldb)
    -> int_t;
auto EINSUMS_EXPORT zgels(char trans, int_t m, int_t n, int_t nrhs, std::complex<double> *a, int_t lda, std::complex<double> *b, int_t ldb)
    -> int_t;
} // namespace detail

template <typename T>
auto gels(char trans, int_t m, int_t n, int_t nrhs, T *a, int_t lda, T *b, int_t ldb) -> int_t;

template <>
inline auto gels<float>(char trans, int_t m, int_t n, int_t nrhs, float *a, int_t lda, float *b, int_t ldb) -> int_t {
    return detail::sgels(trans, m, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto gels<double>(char trans, int_t m, int_t n, int_t nrhs, double *a, int_t lda, double *b, int_t ldb) -> int_t {
    return detail::dgels(trans, m, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto gels<std::complex<float>>(char trans, int_t m, int_t n, int_t nrhs, std::complex<float> *a, int_t lda, std::complex<float> *b,
                                      int_t ldb) -> int_t {
    return detail::cgels(trans, m, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto gels<std::complex<double>>(char trans, int_t m, int_t n, int_t nrhs, std::complex<double> *a, int_t lda,
                                       std::complex<double> *b, int_t ldb) -> int_t {
    return detail::zgels(trans, m, n, nrhs, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------
// swap
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT sswap(int_t n, float *x, int_t incx, float *y, int_t incy);
void EINSUMS_EXPORT dswap(int_t n, double *x, int_t incx, double *y, int_t incy);
void EINSUMS_EXPORT cswap(int_t n, std::complex<float> *x, int_t incx, std::complex<float> *y, int_t incy);
void EINSUMS_EXPORT zswap(int_t n, std::complex<double> *x, int_t incx, std::complex<double> *y, int_t incy);
} // namespace detail

template <typename T>
void swap(int_t n, T *x, int_t incx, T *y, int_t incy);

template <>
inline void swap<float>(int_t n, float *x, int_t incx, float *y, int_t incy) {
    detail::sswap(n, x, incx, y, incy);
}

template <>
inline void swap<double>(int_t n, double *x, int_t incx, double *y, int_t incy) {
    detail::dswap(n, x, incx, y, incy);
}

template <>
inline void swap<std::complex<float>>(int_t n, std::complex<float> *x, int_t incx, std::complex<float> *y, int_t incy) {
    detail::cswap(n, x, incx, y, incy);
}

template <>
inline void swap<std::complex<double>>(int_t n, std::complex<double> *x, int_t incx, std::complex<double> *y, int_t incy) {
    detail::zswap(n, x, incx, y, incy);
}

// ---------------------------------------------------------------------------
// iamax
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT isamax(int_t n, float const *x, int_t incx) -> int_t;
auto EINSUMS_EXPORT idamax(int_t n, double const *x, int_t incx) -> int_t;
auto EINSUMS_EXPORT icamax(int_t n, std::complex<float> const *x, int_t incx) -> int_t;
auto EINSUMS_EXPORT izamax(int_t n, std::complex<double> const *x, int_t incx) -> int_t;
} // namespace detail

template <typename T>
auto iamax(int_t n, T const *x, int_t incx) -> int_t;

template <>
inline auto iamax<float>(int_t n, float const *x, int_t incx) -> int_t {
    return detail::isamax(n, x, incx);
}

template <>
inline auto iamax<double>(int_t n, double const *x, int_t incx) -> int_t {
    return detail::idamax(n, x, incx);
}

template <>
inline auto iamax<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx) -> int_t {
    return detail::icamax(n, x, incx);
}

template <>
inline auto iamax<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx) -> int_t {
    return detail::izamax(n, x, incx);
}

// ---------------------------------------------------------------------------
// trsv
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT strsv(char uplo, char trans, char diag, int_t n, float const *a, int_t lda, float *x, int_t incx);
void EINSUMS_EXPORT dtrsv(char uplo, char trans, char diag, int_t n, double const *a, int_t lda, double *x, int_t incx);
void EINSUMS_EXPORT ctrsv(char uplo, char trans, char diag, int_t n, std::complex<float> const *a, int_t lda, std::complex<float> *x,
                          int_t incx);
void EINSUMS_EXPORT ztrsv(char uplo, char trans, char diag, int_t n, std::complex<double> const *a, int_t lda, std::complex<double> *x,
                          int_t incx);
} // namespace detail

template <typename T>
void trsv(char uplo, char trans, char diag, int_t n, T const *a, int_t lda, T *x, int_t incx);

template <>
inline void trsv<float>(char uplo, char trans, char diag, int_t n, float const *a, int_t lda, float *x, int_t incx) {
    detail::strsv(uplo, trans, diag, n, a, lda, x, incx);
}

template <>
inline void trsv<double>(char uplo, char trans, char diag, int_t n, double const *a, int_t lda, double *x, int_t incx) {
    detail::dtrsv(uplo, trans, diag, n, a, lda, x, incx);
}

template <>
inline void trsv<std::complex<float>>(char uplo, char trans, char diag, int_t n, std::complex<float> const *a, int_t lda,
                                      std::complex<float> *x, int_t incx) {
    detail::ctrsv(uplo, trans, diag, n, a, lda, x, incx);
}

template <>
inline void trsv<std::complex<double>>(char uplo, char trans, char diag, int_t n, std::complex<double> const *a, int_t lda,
                                       std::complex<double> *x, int_t incx) {
    detail::ztrsv(uplo, trans, diag, n, a, lda, x, incx);
}

// ---------------------------------------------------------------------------
// syr2k
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT ssyr2k(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                           float beta, float *c, int_t ldc);
void EINSUMS_EXPORT dsyr2k(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                           double beta, double *c, int_t ldc);
void EINSUMS_EXPORT csyr2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zsyr2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void syr2k(char uplo, char trans, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c, int_t ldc);

template <>
inline void syr2k<float>(char uplo, char trans, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b, int_t ldb,
                         float beta, float *c, int_t ldc) {
    detail::ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void syr2k<double>(char uplo, char trans, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b, int_t ldb,
                          double beta, double *c, int_t ldc) {
    detail::dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void syr2k<std::complex<float>>(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                                       int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c,
                                       int_t ldc) {
    detail::csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void syr2k<std::complex<double>>(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                                        int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta,
                                        std::complex<double> *c, int_t ldc) {
    detail::zsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// her2k
// ---------------------------------------------------------------------------
namespace detail {
void EINSUMS_EXPORT cher2k(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                           std::complex<float> const *b, int_t ldb, float beta, std::complex<float> *c, int_t ldc);
void EINSUMS_EXPORT zher2k(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                           std::complex<double> const *b, int_t ldb, double beta, std::complex<double> *c, int_t ldc);
} // namespace detail

template <typename T>
void her2k(char uplo, char trans, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, RemoveComplexT<T> beta, T *c,
           int_t ldc);

template <>
inline void her2k<std::complex<float>>(char uplo, char trans, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                                       int_t lda, std::complex<float> const *b, int_t ldb, float beta, std::complex<float> *c, int_t ldc) {
    detail::cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void her2k<std::complex<double>>(char uplo, char trans, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                                        int_t lda, std::complex<double> const *b, int_t ldb, double beta, std::complex<double> *c,
                                        int_t ldc) {
    detail::zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// ---------------------------------------------------------------------------
// trtrs
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT strtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT dtrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb)
    -> int_t;
auto EINSUMS_EXPORT ctrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda,
                           std::complex<float> *b, int_t ldb) -> int_t;
auto EINSUMS_EXPORT ztrtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda,
                           std::complex<double> *b, int_t ldb) -> int_t;
} // namespace detail

template <typename T>
auto trtrs(char uplo, char trans, char diag, int_t n, int_t nrhs, T const *a, int_t lda, T *b, int_t ldb) -> int_t;

template <>
inline auto trtrs<float>(char uplo, char trans, char diag, int_t n, int_t nrhs, float const *a, int_t lda, float *b, int_t ldb) -> int_t {
    return detail::strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto trtrs<double>(char uplo, char trans, char diag, int_t n, int_t nrhs, double const *a, int_t lda, double *b, int_t ldb)
    -> int_t {
    return detail::dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto trtrs<std::complex<float>>(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<float> const *a, int_t lda,
                                       std::complex<float> *b, int_t ldb) -> int_t {
    return detail::ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

template <>
inline auto trtrs<std::complex<double>>(char uplo, char trans, char diag, int_t n, int_t nrhs, std::complex<double> const *a, int_t lda,
                                        std::complex<double> *b, int_t ldb) -> int_t {
    return detail::ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

// ---------------------------------------------------------------------------
// trtri
// ---------------------------------------------------------------------------
namespace detail {
auto EINSUMS_EXPORT strtri(char uplo, char diag, int_t n, float *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT dtrtri(char uplo, char diag, int_t n, double *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT ctrtri(char uplo, char diag, int_t n, std::complex<float> *a, int_t lda) -> int_t;
auto EINSUMS_EXPORT ztrtri(char uplo, char diag, int_t n, std::complex<double> *a, int_t lda) -> int_t;
} // namespace detail

template <typename T>
auto trtri(char uplo, char diag, int_t n, T *a, int_t lda) -> int_t;

template <>
inline auto trtri<float>(char uplo, char diag, int_t n, float *a, int_t lda) -> int_t {
    return detail::strtri(uplo, diag, n, a, lda);
}

template <>
inline auto trtri<double>(char uplo, char diag, int_t n, double *a, int_t lda) -> int_t {
    return detail::dtrtri(uplo, diag, n, a, lda);
}

template <>
inline auto trtri<std::complex<float>>(char uplo, char diag, int_t n, std::complex<float> *a, int_t lda) -> int_t {
    return detail::ctrtri(uplo, diag, n, a, lda);
}

template <>
inline auto trtri<std::complex<double>>(char uplo, char diag, int_t n, std::complex<double> *a, int_t lda) -> int_t {
    return detail::ztrtri(uplo, diag, n, a, lda);
}

} // namespace einsums::blas