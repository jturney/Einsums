//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/blas/blas.hpp>
#include <einsums/concepts/complex.hpp>
#include <einsums/concepts/smart_pointer.hpp>
#include <einsums/concepts/tensor.hpp>
#include <einsums/linear_algebra/base.hpp>
#include <einsums/linear_algebra/detail/index_utilities.hpp>
#include <einsums/linear_algebra/unoptimized.hpp>
#include <einsums/tensor/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

/**
 * @brief Computes the square sum of a tensor.
 *
 * returns the values scale_out and sumsq_out such that
 * \f[
 *   (scale_{out}^{2})*sumsq_{out} = a( 1 )^{2} +...+ a( n )^{2} + (scale_{in}^{2})*sumsq_{in},
 * \f]
 *
 * Under the hood the LAPACK routine `lassq` is used.
 *
 * @code
 * NEED TO ADD AN EXAMPLE
 * @endcode
 *
 * @tparam AType The type of the tensor
 * @tparam ADataType The underlying data type of the tensor
 * @tparam ARank The rank of the tensor
 * @param a The tensor to compute the sum of squares for
 * @param scale scale_in and scale_out for the equation provided
 * @param sumsq sumsq_in and sumsq_out for the equation provided
 */
template <TensorConcept AType>
void sum_square(const AType &a, remove_complex_t<typename AType::value_type> *scale, remove_complex_t<typename AType::value_type> *sumsq) {
    LabeledSection0();
    detail::sum_square(a, scale, sumsq);
}

/**
 * @brief General matrix multiplication.
 *
 * Takes two rank-2 tensors ( \p A and \p B ) performs the multiplication and stores the result in to another
 * rank-2 tensor that is passed in ( \p C ).
 *
 * In this equation, \p TransA is op(A) and \p TransB is op(B).
 * @f[
 * C = \alpha \;op(A) \;op(B) + \beta C
 * @f]
 *
 * @code
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto B = einsums::create_random_tensor("B", 3, 3);
 * auto C = einsums::create_tensor("C", 3, 3);
 *
 * einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
 * @endcode
 *
 * @tparam TransA Tranpose A? true or false
 * @tparam TransB Tranpose B? true or false
 * @param alpha Scaling factor for the product of A and B
 * @param A First input tensor
 * @param B Second input tensor
 * @param beta Scaling factor for the output tensor C
 * @param C Output tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires std::convertible_to<U, typename AType::value_type>;
        requires SameUnderlying<AType, BType, CType>;
    }
void gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C) {
    LabeledSection0();
    detail::gemm<TransA, TransB>(alpha, A, B, beta, C);
}

/**
 * @brief General matrix multiplication. Returns new tensor.
 *
 * Takes two rank-2 tensors performs the multiplication and returns the result
 *
 * @code
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto B = einsums::create_random_tensor("B", 3, 3);
 *
 * auto C = einsums::linear_algebra::gemm<false, false>(1.0, A, B);
 * @endcode
 *
 * @tparam TransA Tranpose A?
 * @tparam TransB Tranpose B?
 * @param alpha Scaling factor for the product of A and B
 * @param A First input tensor
 * @param B Second input tensor
 * @returns resulting tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, typename U>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires std::is_same_v<remove_view_t<AType>, remove_view_t<BType>>;
        requires std::convertible_to<U, typename AType::value_type>;
        requires SameUnderlying<AType, BType>;
    }
auto gemm(const U alpha, const AType &A, const BType &B) -> remove_view_t<AType> {
    LabeledSection0();

    remove_view_t<AType> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};
    gemm<TransA, TransB>(static_cast<typename AType::value_type>(alpha), A, B, static_cast<typename AType::value_type>(0.0), &C);

    return C;
}

/**
 * @brief Computes a common double multiplication between two matrices.
 *
 * Computes @f$ C = OP(B)^T OP(A) OP(B) @f$.
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void symm_gemm(const AType &A, const BType &B, CType *C) {
    LabeledSection0();

    detail::symm_gemm<TransA, TransB>(A, B, C);
}

/**
 * @brief General matrix-vector multiplication.
 *
 * This function performs one of the matrix-vector operations
 * \f[
 *    y := alpha*A*z + beta*y\mathrm{,\ or\ }y := alpha*A^{T}*z + beta*y,
 * \f]
 * where alpha and beta are scalars, z and y are vectors and A is an
 * \f$m\f$ by \f$n\f$ matrix.
 *
 * @code
 * NEED TO ADD AN EXAMPLE
 * @endcode
 *
 * @tparam TransA Transpose matrix A? true or false
 * @tparam AType The type of the matrix A
 * @tparam XType The type of the vector z
 * @tparam YType The type of the vector y
 * @tparam ARank The rank of the matrix A
 * @tparam XYRank  The rank of the vectors z and y
 * @tparam T The underlying data type
 * @param alpha Scaling factor for the product of A and z
 * @param A Matrix A
 * @param z Vector z
 * @param beta Scaling factor for the output vector y
 * @param y Output vector y
 */
template <bool TransA, MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires InSamePlace<AType, XType, YType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::value_type>;
    }
void gemv(const U alpha, const AType &A, const XType &z, const U beta, YType *y) {
    LabeledSection1(fmt::format("<TransA={}>", TransA));

    detail::gemv<TransA>(alpha, A, z, beta, y);
}

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced.
 *
 * @code
 * // Create tensors A and b.
 * auto A = einsums::create_tensor("A", 3, 3);
 * auto b = einsums::create_tensor("b", 3);
 *
 * // Fill A with the symmetric data.
 * A.vector_data() = einsums::VectorData{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
 *
 * // On exit, A is destroyed and replaced with the eigenvectors.
 * // b is replaced with the eigenvalues in ascending order.
 * einsums::linear_algebra::syev(&A, &b);
 * @endcode
 *
 * @tparam AType The type of the tensor A
 * @tparam ARank The rank of the tensor A (required to be 2)
 * @tparam WType The type of the tensor W
 * @tparam WRank The rank of the tensor W (required to be 1)
 * @tparam T The underlying data type (required to be real)
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param A
 *   On entry, the symmetric matrix A in the leading N-by-N upper triangular part of A.
 *   On exit, if eigenvectors are requested, the orthonormal eigenvectors of A.
 *   Any data previously stored in A is destroyed.
 * @param W On exit, the eigenvalues in ascending order.
 */
template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires SameUnderlying<AType, WType>;
        requires !Complex<AType>;
    }
void syev(AType *A, WType *W) {

    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
    detail::syev<ComputeEigenvectors>(A, W);
}

/**
 * @brief Compute the general eigendecomposition of a matrix.
 *
 * Can only be used to compute both left and right eigen vectors or neither.
 */
template <bool ComputeLeftRightEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires std::is_same_v<typename WType::value_type, add_complex_t<typename AType::value_type>>;
    }
void geev(AType *A, WType *W, AType *lvecs, AType *rvecs) {
    LabeledSection1(fmt::format("<ComputeLeftRightEigenvectors={}>", ComputeLeftRightEigenvectors));

    detail::geev<ComputeLeftRightEigenvectors>(A, W, lvecs, rvecs);
}

template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires Complex<AType>;
        requires NotComplex<WType>;
        requires std::is_same_v<typename WType::value_type, remove_complex_t<typename AType::value_type>>;
    }
void heev(AType *A, WType *W) {
    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
    detail::heev<ComputeEigenvectors>(A, W);
}

template <MatrixConcept AType, MatrixConcept BType>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires SameUnderlying<AType, BType>;
    }
auto gesv(AType *A, BType *B) -> int {

    LabeledSection0();
    return detail::gesv(A, B);
}

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced.
 *
 * @code
 * // Create tensors A and b.
 * auto A = einsums::create_tensor("A", 3, 3);
 *
 * // Fill A with the symmetric data.
 * A.vector_data() = einsums::VectorData{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
 *
 * // On exit, A is not destroyed. The eigenvectors and eigenvalues are returned in a std::tuple.
 * auto [evecs, evals ] = einsums::linear_algebra::syev(A);
 * @endcode
 *
 * @tparam AType The type of the tensor A
 * @tparam T The underlying data type (required to be real)
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param A The symmetric matrix A in the leading N-by-N upper triangular part of A.
 * @return std::tuple<Tensor<T, 2>, Tensor<T, 1>> The eigenvectors and eigenvalues.
 */
template <bool ComputeEigenvectors = true, MatrixConcept AType>
    requires(NotComplex<AType>)
auto syev(const AType &A) -> std::tuple<remove_view_t<AType>, basic_tensor_like<AType, typename AType::value_type, 1>> {
    LabeledSection0();

    assert(A.dim(0) == A.dim(1));

    remove_view_t<AType> a = A;

    basic_tensor_like<AType, typename AType::value_type, 1> w{"eigenvalues", A.dim(0)};

    syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

/**
 * Scales a tensor by a scalar.
 *
 * @code
 * auto A = einsums::create_ones_tensor("A", 3, 3);
 *
 * // A is filled with 1.0
 * einsums::linear_algebra::scale(2.0, &A);
 * // A is now filled with 2.0
 * @endcode
 *
 * @tparam AType The type of the tensor
 * @tparam ARank The rank of the tensor
 * @tparam T The underlying data type
 * @param scale The scalar to scale the tensor by
 * @param A The tensor to scale
 */
template <TensorConcept AType>
void scale(typename AType::value_type scale, AType *A) {
    LabeledSection0();

    detail::scale(scale, A);
}

template <MatrixConcept AType>
void scale_row(size_t row, typename AType::value_type scale, AType *A) {
    LabeledSection0();

    detail::scale_row(row, scale, A);
}

template <MatrixConcept AType>
void scale_column(size_t col, typename AType::value_type scale, AType *A) {
    LabeledSection0();

    detail::scale_column(col, scale, A);
}

/**
 * @brief Computes the matrix power of a to alpha.  Return a new tensor, does not destroy a.
 *
 * @tparam AType
 * @param a Matrix to take power of
 * @param alpha The power to take
 * @param cutoff Values below cutoff are considered zero.
 *
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<double, 2>, AType>, AType>
 */
template <MatrixConcept AType>
auto pow(const AType &a, typename AType::value_type alpha,
         typename AType::value_type cutoff = std::numeric_limits<typename AType::value_type>::epsilon()) -> remove_view_t<AType> {
    LabeledSection0();

    return detail::pow(a, alpha, cutoff);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <VectorConcept AType, VectorConcept BType>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires SameUnderlying<AType, BType>;
    }
auto dot(const AType &A, const BType &B) -> typename AType::value_type {
    LabeledSection0();

    return detail::dot(A, B);
}
#endif

/**
 * @brief Performs the dot product between two tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}B_{ijk\cdots}@f$
 *
 * @param A One of the tensors
 * @param B The other tensor.
 */
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires InSamePlace<AType, BType>;
        requires AType::rank != 1;
    }
auto dot(const AType &A, const BType &B) -> typename AType::value_type {

    LabeledSection0();

    return detail::dot(A, B);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <VectorConcept AType, VectorConcept BType>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires SameUnderlying<AType, BType>;
    }
auto true_dot(const AType &A, const BType &B) -> typename AType::value_type {
    LabeledSection0();

    return detail::true_dot(A, B);
}
#endif

/**
 * @brief Performs the true dot product between two tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}^* B_{ijk\cdots}@f$, where the asterisk indicates the complex conjugate.
 * If the tensors are real-valued, then this is equivalent to dot.
 *
 * @param A One of the tensors. The complex conjugate is taken of this.
 * @param B The other tensor.
 */
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType>;
        requires InSamePlace<AType, BType>;
        requires AType::rank != 1;
    }
auto true_dot(const AType &A, const BType &B) -> typename AType::value_type {

    LabeledSection0();

    return detail::true_dot(A, B);
}

/**
 * @brief Performs the dot product between three tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}B_{ijk\cdots}C_{ijk\cdots}@f$
 *
 * @param A One of the tensors.
 * @param B The second tensor.
 * @param C The third tensor.
 */
template <TensorConcept AType, TensorConcept BType, TensorConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
    }
auto dot(const AType &A, const BType &B, const CType &C) -> typename AType::value_type {

    LabeledSection0();
    return detail::dot(A, B, C);
}

template <TensorConcept XType, TensorConcept YType>
    requires requires {
        requires InSamePlace<XType, YType>;
        requires SameUnderlyingAndRank<XType, YType>;
    }
void axpy(typename XType::value_type alpha, const XType &X, YType *Y) {
    LabeledSection0();

    detail::axpy(alpha, X, Y);
}

template <TensorConcept XType, TensorConcept YType>
    requires requires {
        requires InSamePlace<XType, YType>;
        requires SameUnderlyingAndRank<XType, YType>;
    }
void axpby(typename XType::value_type alpha, const XType &X, typename XType::value_type beta, YType *Y) {
    LabeledSection0();

    detail::axpby(alpha, X, beta, Y);
}

template <MatrixConcept AType, VectorConcept XYType>
    requires requires {
        requires SameUnderlying<AType, XYType>;
        requires InSamePlace<AType, XYType>;
    }
void ger(typename AType::value_type alpha, const XYType &X, const XYType &Y, AType *A) {
    LabeledSection0();

    detail::ger(alpha, X, Y, A);
}

/**
 * @brief Computes the LU factorization of a general m-by-n matrix.
 *
 * The routine computes the LU factorization of a general m-by-n matrix A as
 * \f[
 * A = P*L*U
 * \f]
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements and U is upper triangular. The routine uses
 * partial pivoting, with row interchanges.
 *
 * @tparam TensorType
 * @param A
 * @param pivot
 * @return
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
auto getrf(TensorType *A, std::vector<blas::int_t> *pivot) -> int {
    LabeledSection0();

    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        // println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }
    int result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->stride(0), pivot->data());

    if (result < 0) {
        println_warn("getrf: argument {} has an invalid value", -result);
        abort();
    }

    return result;
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @tparam T The underlying data type
 * @tparam TensorRank The rank of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
auto getri(TensorType *A, const std::vector<blas::int_t> &pivot) -> int {
    LabeledSection0();

    int result = blas::getri(A->dim(0), A->data(), A->stride(0), pivot.data());

    if (result < 0) {
        println_warn("getri: argument {} has an invalid value", -result);
    }
    return result;
}

/**
 * @brief Inverts a matrix.
 *
 * Utilizes the LAPACK routines getrf and getri to invert a matrix.
 *
 * @tparam TensorType The type of the tensor
 * @tparam T The underlying data type
 * @tparam TensorRank The rank of the tensor
 * @param A Matrix to invert. On exit, the inverse of A.
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
void invert(TensorType *A) {
    if constexpr (is_incore_block_tensor_v<TensorType>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            einsums::linear_algebra::invert(&(A->block(i)));
        }
    } else {
        LabeledSection0();

        std::vector<blas::int_t> pivot(A->dim(0));
        int                      result = getrf(A, &pivot);
        if (result > 0) {
            println_abort("invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed", result,
                          result);
        }

        result = getri(A, pivot);
        if (result > 0) {
            println_abort("invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed", result,
                          result);
        }
    }
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <SmartPointer SmartPtr>
void invert(SmartPtr *A) {
    LabeledSection0();

    return invert(A->get());
}
#endif

/**
 * @brief Indicates the type of norm to compute.
 */
enum class Norm : char {
    MaxAbs    = 'M', /**< \f$val = max(abs(Aij))\f$, largest absolute value of the matrix A. */
    One       = '1', /**< \f$val = norm1(A)\f$, 1-norm of the matrix A (maximum column sum) */
    Infinity  = 'I', /**< \f$val = normI(A)\f$, infinity norm of the matrix A (maximum row sum) */
    Frobenius = 'F', /**< \f$val = normF(A)\f$, Frobenius norm of the matrix A (square root of sum of squares). */
    //    Two       = 'F'
};

/**
 * @brief Computes the norm of a matrix.
 *
 * Returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real matrix A.
 *
 * @note
 * This function assumes that the matrix is stored in column-major order.
 *
 * @code
 * using namespace einsums;
 *
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto norm = einsums::linear_algebra::norm(einsums::linear_algebra::Norm::One, A);
 * @endcode
 *
 * @tparam AType The type of the matrix
 * @tparam ADataType The underlying data type of the matrix
 * @tparam ARank The rank of the matrix
 * @param norm_type where Norm::One denotes the one norm of a matrix (maximum column sum),
 *   Norm::Infinity denotes the infinity norm of a matrix  (maximum row sum) and
 *   Norm::Frobenius denotes the Frobenius norm of a matrix (square root of sum of
 *   squares). Note that \f$ max(abs(A(i,j))) \f$ is not a consistent matrix norm.
 * @param a The matrix to compute the norm of
 * @return The norm of the matrix
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto norm(Norm norm_type, const AType &a) -> remove_complex_t<typename AType::value_type> {
    LabeledSection0();

    std::vector<remove_complex_t<typename AType::value_type>> work(4 * a.dim(0), 0.0);
    return blas::lange(static_cast<char>(norm_type), a.dim(0), a.dim(1), a.data(), a.stride(0), work.data());
}

template <TensorConcept AType>
remove_complex_t<AType> vec_norm(const AType &a) {
    return std::sqrt(std::abs(true_dot(a, a)));
}

// Uses the original svd function found in lapack, gesvd, request all left and right vectors.
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto svd(const AType &_A) -> std::tuple<tensor<typename AType::value_type, 2>, tensor<remove_complex_t<typename AType::value_type>, 1>,
                                        tensor<typename AType::value_type, 2>> {
    using T = typename AType::value_type;
    LabeledSection0();

    //    DisableOMPThreads const nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    tensor<T, 2> A = _A;

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.stride(0);

    // Test if it is absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>("superb", std::min(m, n));
    superb.zero();

    //    int info{0};
    int info = blas::gesvd('A', 'A', m, n, A.data(), lda, S.data(), U.data(), m, Vt.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto svd_nullspace(const AType &_A) -> tensor<typename AType::value_type, 2> {
    using T = typename AType::value_type;
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    tensor<T, 2> A = _A;

    blas::int_t m   = A.dim(0);
    blas::int_t n   = A.dim(1);
    blas::int_t lda = A.stride(0);

    auto U = create_tensor<T>("U", m, m);
    zero(U);
    auto S = create_tensor<T>("S", n);
    zero(S);
    auto V = create_tensor<T>("V", n, n);
    zero(V);
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);

    int info = blas::gesvd('N', 'A', m, n, A.data(), lda, S.data(), U.data(), m, V.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    // Determine the rank of the nullspace matrix
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (S(i) > 1e-12) {
            rank++;
        }
    }

    // println("rank {}", rank);
    auto Vview     = V(range{rank, V.dim(0)}, all);
    auto nullspace = Tensor(V);

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace.dim(0); i++) {
        T sum{0};
        for (int j = 0; j < nullspace.dim(1); j++) {
            sum += std::pow(nullspace(i, j), 2.0);
        }
        sum = std::sqrt(sum);
        scale_row(i, sum, &nullspace);
    }

    return nullspace;
}

enum class Vectors : char { All = 'A', Some = 'S', Overwrite = 'O', None = 'N' };

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto svd_dd(const AType &_A, Vectors job = Vectors::All)
    -> std::tuple<tensor<typename AType::value_type, 2>, tensor<remove_complex_t<typename AType::value_type>, 1>,
                  tensor<typename AType::value_type, 2>> {
    using T = typename AType::value_type;
    LabeledSection0();

    //    DisableOMPThreads const nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    tensor<T, 2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    zero(U);
    auto S = create_tensor<remove_complex_t<T>>("S", std::min(m, n));
    zero(S);
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    zero(Vt);

    int info = blas::gesdd(static_cast<char>(job), static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(n), S.data(),
                           U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n));

    if (info != 0) {
        if (info < 0) {
            println_abort("svd_a: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd_a: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto truncated_svd(const AType &_A, size_t k)
    -> std::tuple<tensor<typename AType::value_type, 2>, tensor<remove_complex_t<AType>, 1>, tensor<typename AType::value_type, 2>> {
    using T = typename AType::value_type;
    LabeledSection0();

    size_t m = _A.dim(0);
    size_t n = _A.dim(1);

    // Omega Test Matrix
    auto omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    tensor<T, 2> Y("Y", m, k + 5);
    gemm<false, false>(T{1.0}, _A, omega, T{0.0}, &Y);

    tensor<T, 1> tau("tau", std::min(m, k + 5));
    // Compute QR factorization of Y
    int info1 = blas::geqrf(m, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    if constexpr (!is_complex_v<T>) {
        int info2 = blas::orgqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    } else {
        int info2 = blas::ungqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    }

    // Cast the matrix A into a smaller rank (B)
    tensor<T, 2> B("B", k + 5, n);
    gemm<true, false>(T{1.0}, Y, _A, T{0.0}, &B);

    // Perform svd on B
    auto [Utilde, S, Vt] = svd_dd(B);

    // Cast U back into full basis
    tensor<T, 2> U("U", m, k + 5);
    gemm<false, false>(T{1.0}, Y, Utilde, T{0.0}, &U);

    return std::make_tuple(U, S, Vt);
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto truncated_syev(const AType &A, size_t k) -> std::tuple<tensor<typename AType::value_type, 2>, tensor<typename AType::value_type, 1>> {
    using T = typename AType::value_type;
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("Non-square matrix used as input of truncated_syev!");
    }

    size_t n = A.dim(0);

    // Omega Test Matrix
    tensor<T, 2> omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    tensor<T, 2> Y("Y", n, k + 5);
    gemm<false, false>(T{1.0}, A, omega, T{0.0}, &Y);

    tensor<T, 1> tau("tau", std::min(n, k + 5));
    // Compute QR factorization of Y
    blas::int_t const info1 = blas::geqrf(n, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    blas::int_t const info2 = blas::orgqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));

    tensor<T, 2> &Q1 = Y;

    // Cast the matrix A into a smaller rank (B)
    // B = Q^T * A * Q
    tensor<T, 2> Btemp("Btemp", k + 5, n);
    gemm<true, false>(1.0, Q1, A, 0.0, &Btemp);
    tensor<T, 2> B("B", k + 5, k + 5);
    gemm<false, false>(1.0, Btemp, Q1, 0.0, &B);

    // Create buffer for eigenvalues
    tensor<T, 1> w("eigenvalues", k + 5);

    // Diagonalize B
    syev(&B, &w);

    // Cast U back into full basis (B is column-major so we need to transpose it)
    tensor<T, 2> U("U", n, k + 5);
    gemm<false, true>(1.0, Q1, B, 0.0, &U);

    return std::make_tuple(U, w);
}

template <MatrixConcept AType, typename T>
    requires requires {
        requires CoreTensorConcept<AType>;
        requires std::is_same_v<typename AType::value_type, T>;
    }
inline auto pseudoinverse(const AType &A, T tol) -> tensor<T, 2> {
    LabeledSection0();

    auto [U, S, Vh] = svd_a(A);

    size_t new_dim;
    for (size_t v = 0; v < S.dim(0); v++) {
        T val = S(v);
        if (val > tol)
            scale_column(v, T{1.0} / val, &U);
        else {
            new_dim = v;
            break;
        }
    }

    tensor_view<T, 2> U_view = U(all, range{0, new_dim});
    tensor_view<T, 2> V_view = Vh(range{0, new_dim}, all);

    tensor<T, 2> pinv("pinv", A.dim(0), A.dim(1));
    gemm<false, false>(1.0, U_view, V_view, 0.0, &pinv);

    return pinv;
}

template <MatrixConcept AType, MatrixConcept QType>
    requires requires {
        requires CoreTensorConcept<AType>;
        requires CoreTensorConcept<QType>;
        requires SameUnderlying<AType, QType>;
    }
inline auto solve_continuous_lyapunov(const AType &A, const QType &Q) -> tensor<typename AType::value_type, 2> {
    using T = typename AType::value_type;
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}), do not match", A.dim(0), A.dim(1));
    }
    if (Q.dim(0) != Q.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of Q ({} x {}), do not match", Q.dim(0), Q.dim(1));
    }
    if (A.dim(0) != Q.dim(0)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}) and Q ({} x {}), do not match", A.dim(0), A.dim(1), Q.dim(0),
                      Q.dim(1));
    }

    size_t n = A.dim(0);

    /// TODO: Break this off into a separate schur function
    // Compute Schur Decomposition of A
    tensor<T, 2>             R = A; // R is a copy of A
    tensor<T, 2>             wr("Schur Real Buffer", n, n);
    tensor<T, 2>             wi("Schur Imaginary Buffer", n, n);
    tensor<T, 2>             U("Lyapunov U", n, n);
    std::vector<blas::int_t> sdim(1);
    blas::gees('V', n, R.data(), n, sdim.data(), wr.data(), wi.data(), U.data(), n);

    // Compute F = U^T * Q * U
    tensor<T, 2> Fbuff = gemm<true, false>(1.0, U, Q);
    tensor<T, 2> F     = gemm<false, false>(1.0, Fbuff, U);

    // Call the Sylvester Solve
    std::vector<T> scale(1);
    blas::trsyl('N', 'N', 1, n, n, const_cast<const T *>(R.data()), n, const_cast<const T *>(R.data()), n, F.data(), n, scale.data());

    tensor<T, 2> Xbuff = gemm<false, false>(scale[0], U, F);
    tensor<T, 2> X     = gemm<false, true>(1.0, Xbuff, U);

    return X;
}

// TODO: Bring this back
// ALIAS_TEMPLATE_FUNCTION(solve_lyapunov, solve_continuous_lyapunov)

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto qr(const AType &_A) -> std::tuple<tensor<typename AType::value_type, 2>, tensor<typename AType::value_type, 1>> {
    using T = typename AType::value_type;
    LabeledSection0();

    // Copy A because it will be overwritten by the QR call.
    tensor<T, 2>      A = _A;
    const blas::int_t m = A.dim(0);
    const blas::int_t n = A.dim(1);

    tensor<T, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    blas::int_t info = blas::geqrf(m, n, A.data(), n, tau.data());

    if (info != 0) {
        println_abort("{} parameter to geqrf has an illegal value.", -info);
    }

    // Extract Matrix Q out of QR factorization
    // blas::int_t info2 = blas::orgqr(m, n, tau.dim(0), A.data(), n, const_cast<const double *>(tau.data()));
    return {A, tau};
}

template <MatrixConcept AType, VectorConcept TauType>
    requires requires {
        requires CoreTensorConcept<AType>;
        requires CoreTensorConcept<TauType>;
        requires SameUnderlying<AType, TauType>;
    }
auto q(const AType &qr, const TauType &tau) -> tensor<typename AType::value_type, 2> {
    using T             = typename AType::value_type;
    const blas::int_t m = qr.dim(1);
    const blas::int_t p = qr.dim(0);

    tensor<T, 2> Q = qr;

    blas::int_t info = blas::orgqr(m, m, p, Q.data(), m, tau.data());
    if (info != 0) {
        println_abort("{} parameter to orgqr has an illegal value. {} {} {}", -info, m, m, p);
    }

    return Q;
}

template <TensorConcept AType, TensorConcept BType, TensorConcept CType, typename T>
    requires(SameRank<AType, BType, CType>)
void direct_product(T alpha, const AType &A, const BType &B, T beta, CType *C) {
    LabeledSection0();

    detail::direct_product(alpha, A, B, beta, C);
}

/**
 * Computes the determinant of a matrix.
 */
template <MatrixConcept AType>
typename AType::value_type det(const AType &A) {
    using T = typename AType::value_type;
    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Can only take the determinant of a square matrix.");
    }

    remove_view_t<AType> temp = A;

    std::vector<blas::int_t> pivots;
    int                      singular = getrf(&temp, &pivots);
    if (singular > 0) {
        return T{0.0}; // Matrix is singular, so it has a determinant of zero.
    }

    T ret{1.0};

    int parity = 0;

    // Calculate the effect of the pivots.
#pragma omp parallel for simd reduction(+ : parity)
    for (int i = 0; i < A.dim(0); i++) {
        int         temp_parity = 0;
        blas::int_t curr        = pivots.at(i);

        bool skip = false;

        while (curr != i + 1) {
            if (curr < i + 1) {
                skip = true;
                break;
            }
            temp_parity++;
            curr = pivots.at(curr - 1);
        }

        if (!skip) {
            parity += temp_parity;
        }
    }

// Calculate the contribution of the diagonal elements.
#pragma omp parallel for simd reduction(* : ret)
    for (int i = 0; i < A.dim(0); i++) {
        ret *= A(i, i);
    }

    if (parity % 2 == 1) {
        ret *= T{-1.0};
    }

    return ret;
}

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
