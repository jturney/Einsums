//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SmartPointer.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/LinearAlgebra/BlockTensor.hpp>
#include <Einsums/LinearAlgebra/DiskAlgebra.hpp>
#include <Einsums/LinearAlgebra/TiledTensor.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace einsums::linear_algebra {
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
 * @tparam AType The type of the tensor.
 * @param[in] a The tensor to compute the sum of squares for.
 * @param[inout] scale scale_in and scale_out for the equation provided.
 * @param[inout] sumsq sumsq_in and sumsq_out for the equation provided.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle tensors. Can also handle non-unit strides and both row- and colum-major layouts.
 *      Includes support for block and tiled tensors.
 * @endversion
 */
template <TensorConcept AType>
void sum_square(AType const &a, RemoveComplexT<typename AType::ValueType> *scale,
                RemoveComplexT<typename AType::ValueType> *sumsq) noexcept(BasicTensorConcept<AType>) {
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
 * @tparam AType The tensor type of A.
 * @tparam BType The tensor type of B.
 * @tparam CType The tensor type of C.
 * @tparam U The type for the scale factors.
 * @param[in] alpha Scaling factor for the product of A and B
 * @param[in] A First input tensor
 * @param[in] B Second input tensor
 * @param[in] beta Scaling factor for the output tensor C
 * @param[inout] C Output tensor
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U>
    requires requires {
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    LabeledSection0();

    detail::gemm<TransA, TransB>(alpha, A, B, beta, C);
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
 * @tparam AType The tensor type of A.
 * @tparam BType The tensor type of B.
 * @tparam CType The tensor type of C.
 * @tparam U The type for the scale factors.
 * @param[in] transA Whether to transpose A. Case insensitive. Can be 'n', 't', or 'c'.
 * @param[in] transB Whether to transpose B. Case insensitive. Can be 'n', 't', or 'c'.
 * @param[in] alpha Scaling factor for the product of A and B
 * @param[in] A First input tensor
 * @param[in] B Second input tensor
 * @param[in] beta Scaling factor for the output tensor C
 * @param[inout] C Output tensor
 * @tparam T the underlying data type
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 * @throws std::invalid_argument If the transpose characters are invalid.
 *
 * @versionadded{2.0.0}
 */
template <MatrixConcept AType, MatrixConcept BType, MatrixConcept CType, typename U>
    requires requires {
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    detail::gemm(transA, transB, alpha, A, B, beta, C);
}

// Runtime-rank overloads. These accept tensors whose rank is known only at
// runtime (RuntimeTensor / RuntimeTensorView) by routing directly through
// the TensorImpl-level kernel, which performs a runtime rank-2 check and
// throws ``rank_error`` on mismatch. Distinguished from the static-rank
// overloads above by requiring at least one operand to be dynamic-rank;
// concept overload resolution disambiguates the calls.
template <bool TransA, bool TransB, BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType, typename U>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<BType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<CType>::Rank == einsums::dynamic_rank;
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    LabeledSection0();
    detail::gemm<TransA, TransB>(alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType, typename U>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<BType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<CType>::Rank == einsums::dynamic_rank;
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    detail::gemm(transA, transB, alpha, A.impl(), B.impl(), beta, &C->impl());
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
 * @tparam AType The tensor type of A.
 * @tparam BType The tensor type of B.
 * @tparam U The type for the scale factors.
 * @param[in] alpha Scaling factor for the product of A and B
 * @param[in] A First input tensor
 * @param[in] B Second input tensor
 * @returns resulting tensor
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, typename U>
    requires requires {
        requires std::is_same_v<RemoveViewT<AType>, RemoveViewT<BType>>;
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType>;
    }
[[nodiscard]] auto gemm(U const alpha, AType const &A, BType const &B) -> RemoveViewT<AType> {
    LabeledSection0();

    RemoveViewT<AType> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};
    gemm<TransA, TransB>(static_cast<typename AType::ValueType>(alpha), A, B, static_cast<typename AType::ValueType>(0.0), &C);

    return C;
}

/**
 * @brief Computes a common double multiplication between two matrices.
 *
 * Computes @f$ C = OP(B)^T OP(A) OP(B) @f$.
 *
 * @tparam TransA Whether to transpose the A matrix.
 * @tparam TransB Whether to tranpsose the B matrix.
 * @tparam AType The tensor type of A.
 * @tparam BType The tensor type of B.
 * @tparam CType The tensor type of C.
 * @param[in] A The inner tensor.
 * @param[in] B The outer tensor.
 * @param[out] C The output tensor.
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    LabeledSection0();

    detail::symm_gemm<TransA, TransB>(A, B, C);
}

// Runtime-rank symm_gemm overload: checks rank-2 at runtime and computes
// ``C = B^T * A * B`` (with optional transposes) via two gemm calls.
template <bool TransA, bool TransB, BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<BType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<CType>::Rank == einsums::dynamic_rank;
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    LabeledSection0();

    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "symm_gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }

    size_t const temp_rows = TransA ? A.dim(1) : A.dim(0);
    size_t const temp_cols = TransB ? B.dim(0) : B.dim(1);

    using T = typename AType::ValueType;
    *C      = typename CType::ValueType(0.0);

    Tensor<T, 2> temp{"temp", temp_rows, temp_cols};
    gemm<TransA, TransB>(T{1.0}, A, B, T{0.0}, &temp);
    gemm<!TransB, false>(T{1.0}, B, temp, T{0.0}, C);
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
 * @tparam TransA Transpose matrix A? true or false
 * @tparam AType The type of the matrix A
 * @tparam XType The type of the vector z
 * @tparam YType The type of the vector y
 * @param[in] alpha Scaling factor for the product of A and z
 * @param[in] A Matrix A
 * @param[in] z Vector z
 * @param[in] beta Scaling factor for the output vector y
 * @param[out] y Output vector y
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool TransA, MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    LabeledSection("gemv<TransA={}>", TransA);

    detail::gemv<TransA>(alpha, A, z, beta, y);
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
 * @tparam AType The type of the matrix A
 * @tparam XType The type of the vector z
 * @tparam YType The type of the vector y
 * @param[in] transA Whether to transpose A. Case insensitive. Can be 'n', 't', or 'c'.
 * @param[in] alpha Scaling factor for the product of A and z
 * @param[in] A Matrix A
 * @param[in] z Vector z
 * @param[in] beta Scaling factor for the output vector y
 * @param[out] y Output vector y
 *
 * @throws rank_error If all of the tensors are not rank-2. Only happens when the inputs do not have compile-time rank.
 * @throws tensor_compat_error If the tensors have incompatible dimensions.
 * @throws std::invalid_argument If the transpose character is invalid.
 *
 * @versionadded{2.0.0}
 */
template <MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    LabeledSection("gemv<transA={}>", transA);

    detail::gemv(transA, alpha, A, z, beta, y);
}

// Runtime-rank gemv overloads: accept dynamic-rank operands. The detail
// kernel runtime-checks A.rank() == 2 and z.rank() == y.rank() == 1.
template <bool TransA, BasicTensorConcept AType, BasicTensorConcept XType, BasicTensorConcept YType, typename U>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<XType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<YType>::Rank == einsums::dynamic_rank;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    LabeledSection("gemv<TransA={}>", TransA);
    detail::gemv<TransA>(alpha, A.impl(), z.impl(), beta, &y->impl());
}

template <BasicTensorConcept AType, BasicTensorConcept XType, BasicTensorConcept YType, typename U>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<XType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<YType>::Rank == einsums::dynamic_rank;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    detail::gemv(transA, alpha, A.impl(), z.impl(), beta, &y->impl());
}

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced if the eigenvectors are not computed.
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
 * @tparam WType The type of the tensor W
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param[inout] A
 *   On entry, the symmetric matrix A in the leading N-by-N upper triangular part of A.
 *   On exit, if eigenvectors are requested, the orthonormal eigenvectors of A.
 *   Any data previously stored in A is destroyed.
 * @param[out] W On exit, the eigenvalues in ascending order.
 *
 * @throws rank_error If the inputs have the wrong ranks. The A tensor needs to be rank-2 and the W tensor needs to be rank-1.
 * @throws dimension_error If the matrix input is not square.
 * @throws tensor_compat_error If the length of the eigenvalue vector does not have the same size as the number of rows in the matrix.
 * @throws std::invalid_argument If values passed to internal functions were invalid. This is often due to passing uninitialized or
 * zero-size tensors.
 * @throws std::runtime_error If the eigenvalue algorithm fails to converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires SameUnderlying<AType, WType>;
        requires !Complex<AType>;
    }
void syev(AType *A, WType *W) {
    LabeledSection("syev<ComputeEigenvectors={}>", ComputeEigenvectors);
    detail::syev<ComputeEigenvectors>(A, W);
}

// Runtime-rank syev overload: runtime rank-2 + rank-1 check inside the
// TensorImpl-level kernel.
template <bool ComputeEigenvectors = true, BasicTensorConcept AType, BasicTensorConcept WType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<WType>::Rank == einsums::dynamic_rank;
        requires InSamePlace<AType, WType>;
        requires SameUnderlying<AType, WType>;
        requires !Complex<AType>;
    }
void syev(AType *A, WType *W) {
    LabeledSection("syev<ComputeEigenvectors={}>", ComputeEigenvectors);
    detail::syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

/**
 * @brief Compute the general eigendecomposition of a matrix.
 *
 * @tparam AType The tensor type of A.
 * @tparam WType The tensor type of W.
 * @param[inout] A The tensor to decompose. On exit, it will be overwritten with values used for the computation.
 * @param[out] W The eigenvalues.
 * @param[out] lvecs The left eigenvectors. If null, then these will not be computed.
 * @param[out] rvecs The right eigenvectors. If null, then these will not be computed.
 *
 * @throws rank_error If the inputs have the wrong ranks. The A tensor needs to be rank-2 and the W tensor needs to be rank-1.
 * @throws dimension_error If the matrix input is not square.
 * @throws tensor_compat_error If the length of the eigenvalue vector does not have the same size as the number of rows in the matrix,
 * or the eigenvector outputs, if not null, do not have the same dimensions as the input.
 * @throws std::invalid_argument If values passed to internal functions were invalid. This is often due to passing uninitialized or
 * zero-size tensors.
 * @throws std::runtime_error If the eigenvalue algorithm fails to converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Eigenvector output is no longer handled by a template paramter. Now, it is decided by whether the outputs are null pointers.
 *      When one of the eigenvector outputs is null, that output will not be calculated. It can now handle non-unit inner strides,
 *      though this is done by copying data. It can also handle row- and column-major order without a copy.
 * @endversion
 */
template <MatrixConcept AType, VectorConcept WType, typename LVecPtr, typename RVecPtr>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires std::is_same_v<typename WType::ValueType, AddComplexT<typename AType::ValueType>>;
        requires std::is_null_pointer_v<LVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<LVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<LVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
        requires std::is_null_pointer_v<RVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<RVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<RVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
    }
void geev(AType *A, WType *W, LVecPtr lvecs, RVecPtr rvecs) {
    char jobvl = (lvecs == nullptr) ? 'n' : 'v';
    char jobvr = (rvecs == nullptr) ? 'n' : 'v';
    LabeledSection("geev<jobvl = {}, jobvr = {}>", jobvl, jobvr);

    detail::geev(A, W, lvecs, rvecs);
}

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a complex Hermitian matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced if the eigenvectors are not computed.
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
 * @tparam WType The type of the tensor W
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param[inout] A
 *   On entry, the symmetric matrix A in the leading N-by-N upper triangular part of A.
 *   On exit, if eigenvectors are requested, the orthonormal eigenvectors of A.
 *   Any data previously stored in A is destroyed.
 * @param[out] W On exit, the eigenvalues in ascending order.
 *
 * @throws rank_error If the inputs have the wrong ranks. The A tensor needs to be rank-2 and the W tensor needs to be rank-1.
 * @throws dimension_error If the matrix input is not square.
 * @throws tensor_compat_error If the length of the eigenvalue vector does not have the same size as the number of rows in the matrix.
 * @throws std::invalid_argument If values passed to internal functions were invalid. This is often due to passing uninitialized or
 * zero-size tensors.
 * @throws std::runtime_error If the eigenvalue algorithm fails to converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires Complex<AType>;
        requires NotComplex<WType>;
        requires std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>>;
    }
void heev(AType *A, WType *W) {
    LabeledSection("heev<ComputeEigenvectors={}>", ComputeEigenvectors);
    detail::heev<ComputeEigenvectors>(A, W);
}

// Runtime-rank heev overload. The TensorImpl-level ``heev`` deduces a single
// ``AType`` for both operands; for the complex→real eigenvalue case we
// instead route through the ``syev`` impl which has a separate signature
// for the real-eigenvalue output (heev forwards to syev internally).
template <bool ComputeEigenvectors = true, BasicTensorConcept AType, BasicTensorConcept WType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<WType>::Rank == einsums::dynamic_rank;
        requires InSamePlace<AType, WType>;
        requires Complex<AType>;
        requires NotComplex<WType>;
        requires std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>>;
    }
void heev(AType *A, WType *W) {
    LabeledSection("heev<ComputeEigenvectors={}>", ComputeEigenvectors);
    detail::syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

/**
 * Solve a system of linear equations.
 *
 * @f[
 *  \mathbf{Ax} = \mathbf{B}
 * @f]
 *
 * @tparam AType The type of the A tensor.
 * @tparam BType The type of the B tensor. Can be a matrix or vector.
 * @param[inout] A The coefficient matrix. On exit, it is overwritten by the LU decomposition of the input. The diagonal elements of the
 * lower-triangular matrix are all 1, and are not stored.
 * @param[inout] B The right-hand side matrix. On exit, it will contain the values of the variables that satisfy the system of equations.
 *
 * @return 0 on success. If positive, then the coefficient matrix was singular. The decomposition was performed, but the system was unable
 * to be solved. If negative, then one of the parameters in the underlying LAPACK call was invalid. The absolute value gives which parameter
 * was invalid.
 *
 * @throws rank_error If the coefficient matrix is not rank-2 or the result matrix is not rank-1 or rank-2.
 * @throws dimension_error If the coefficient matrix is not square, or the number of rows of the result matrix is not the same as the number
 * of rows of the coefficient matrix.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      The B matrix can now be a rank-1 tensor. A bug was also fixed where the B matrix was implicitly transposed.
 *      Can also handle non-unit strides and row- and column-major layouts.
 * @endversion
 */
template <MatrixConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlying<AType, BType>;
        requires MatrixConcept<BType> || VectorConcept<BType>;
    }
[[nodiscard]] auto gesv(AType *A, BType *B) -> int {

    LabeledSection0();
    return detail::gesv(A, B);
}

// Runtime-rank gesv overload: TensorImpl-level kernel runtime-checks rank-2
// + rank-1-or-2.
template <BasicTensorConcept AType, BasicTensorConcept BType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<BType>::Rank == einsums::dynamic_rank;
        requires SameUnderlying<AType, BType>;
    }
[[nodiscard]] auto gesv(AType *A, BType *B) -> int {
    LabeledSection0();
    return detail::gesv(&A->impl(), &B->impl());
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
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param[in] A The symmetric matrix A in the leading N-by-N upper triangular part of A.
 * @return std::tuple<Tensor<T, 2>, Tensor<T, 1>> The eigenvectors and eigenvalues.
 *
 * @throws rank_error If the inputs have the wrong ranks. The A tensor needs to be rank-2 and the W tensor needs to be rank-1.
 * @throws dimension_error If the matrix input is not square.
 * @throws tensor_compat_error If the length of the eigenvalue vector does not have the same size as the number of rows in the matrix.
 * @throws std::invalid_argument If values passed to internal functions were invalid. This is often due to passing uninitialized or
 * zero-size tensors.
 * @throws std::runtime_error If the eigenvalue algorithm fails to converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <bool ComputeEigenvectors = true, MatrixConcept AType>
    requires(NotComplex<AType>)
[[nodiscard]] auto syev(AType const &A) -> std::tuple<RemoveViewT<AType>, BasicTensorLike<AType, typename AType::ValueType, 1>> {
    LabeledSection0();

    EINSUMS_ASSERT(A.dim(0) == A.dim(1));

    RemoveViewT<AType> a = A;

    BasicTensorLike<AType, typename AType::ValueType, 1> w{"eigenvalues", A.dim(0)};

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
 * @tparam AType The type of the tensor.
 * @param[in] scale The scalar to scale the tensor by.
 * @param[inout] A The tensor to scale.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept AType>
void scale(typename AType::ValueType scale, AType *A) {
    LabeledSection0();

    detail::scale(scale, A);
}

/**
 * Scales a row in a matrix by a value.
 *
 * @code
 * auto A = einsums::create_ones_tensor("A", 3, 3);
 *
 * // A is filled with 1.0
 * einsums::linear_algebra::scale_row(1, 2.0, &A);
 * // The second row of A is now filled with 2.0. The rest is filled with 1.0.
 * @endcode
 *
 * @tparam AType The type of the tensor.
 * @param[in] row The index of the row to scale.
 * @param[in] scale The scalar to scale the tensor by.
 * @param[inout] A The tensor to scale.
 *
 * @throws std::out_of_range If the row is outside of what the input matrix stores.
 * @throws rank_error If the input matrix is not rank-2.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <MatrixConcept AType>
void scale_row(size_t row, typename AType::ValueType scale, AType *A) {
    LabeledSection0();

    detail::scale_row(row, scale, A);
}

/**
 * Scales a column in a matrix by a value.
 *
 * @code
 * auto A = einsums::create_ones_tensor("A", 3, 3);
 *
 * // A is filled with 1.0
 * einsums::linear_algebra::scale_column(1, 2.0, &A);
 * // The second column of A is now filled with 2.0. The rest is filled with 1.0.
 * @endcode
 *
 * @tparam AType The type of the tensor.
 * @param[in] col The index of the column to scale.
 * @param[in] scale The scalar to scale the tensor by.
 * @param[inout] A The tensor to scale.
 *
 * @throws std::out_of_range If the row is outside of what the input matrix stores.
 * @throws rank_error If the input matrix is not rank-2.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <MatrixConcept AType>
void scale_column(size_t col, typename AType::ValueType scale, AType *A) {
    LabeledSection0();

    detail::scale_column(col, scale, A);
}

/**
 * @brief Computes the matrix power of a to alpha.  Return a new tensor, does not destroy a.
 *
 * @tparam AType
 * @param[in] a Matrix to take power of
 * @param[in] alpha The power to take
 * @param[in] cutoff Values below cutoff are considered zero.
 *
 * @return The matrix power.
 *
 * @warning If any of the eigenvalues when exponentiated gives a non-finite or complex value, that eigenvalue will be set to zero. This may
 * cause numerical imprecisions.
 *
 * @throws rank_error If the inputs have the wrong ranks. The A tensor needs to be rank-2.
 * @throws dimension_error If the matrix input is not square.
 * @throws std::invalid_argument If values passed to internal functions were invalid. This is often due to passing uninitialized or
 * zero-size tensors.
 * @throws std::runtime_error If the eigenvalue algorithm fails to converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <MatrixConcept AType>
[[nodiscard]] auto pow(AType const &a, typename AType::ValueType alpha,
                       typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon()) -> RemoveViewT<AType> {
    LabeledSection0();

    return detail::pow(a, alpha, cutoff);
}

/**
 * @brief Performs the dot product between two tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}B_{ijk\cdots}@f$. This may differ from the geometric dot product for complex tensors.
 * This does not conjugate either tensor, while the geometric dot product conjugates the left tensor.
 *
 * @tparam AType,BType The tensor types.
 * @param[in] A,B The tensors to dot together.
 *
 * @return The dot product of the tensors.
 *
 * @throws rank_error If the input tensors do not have the same rank.
 * @throws dimension_error If the input tensors do not have the same shape.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameRank<AType, BType>;
        requires InSamePlace<AType, BType>;
    }
[[nodiscard]] auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {

    LabeledSection0();

    return detail::dot(A, B);
}

/**
 * @brief Performs the true dot product between two tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}^* B_{ijk\cdots}@f$, where the asterisk indicates the complex conjugate.
 * If the tensors are real-valued, then this is equivalent to dot.
 *
 * @tparam AType,BType The tensor types.
 * @param[in] A One of the tensors. The complex conjugate is taken of this.
 * @param[in] B The other tensor.
 *
 * @return The dot product between two tensors.
 *
 * @throws rank_error If the input tensors do not have the same rank.
 * @throws dimension_error If the input tensors do not have the same shape.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameRank<AType, BType>;
        requires InSamePlace<AType, BType>;
    }
[[nodiscard]] auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {

    LabeledSection0();

    return detail::true_dot(A, B);
}

/**
 * @brief Performs the dot product between three tensors.
 *
 * This performs @f$\sum_{ijk\cdots} A_{ijk\cdots}B_{ijk\cdots}C_{ijk\cdots}@f$
 *
 * @tparam AType,BType,CType The tensor types.
 * @param[in] A,B,C The tensors to dot together.
 *
 * @return The triple dot product.
 *
 * @throws rank_error If the input tensors do not have the same rank.
 * @throws dimension_error If the input tensors do not have the same shape.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept AType, TensorConcept BType, TensorConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameRank<AType, BType, CType>;
    }
[[nodiscard]] auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {

    LabeledSection0();
    return detail::dot(A, B, C);
}

/**
 * Scale and add two tensors together.
 *
 * @f[
 * \mathbf{y} := \alpha \mathbf{x} + \mathbf{y}
 * @f]
 *
 * @tparam XType,YType The tensor types.
 * @param[in] alpha The scale factor for the input.
 * @param[in] X The input tensor.
 * @param[inout] Y The output tensor.
 *
 * @throws rank_error If the input tensors do not have the same rank.
 * @throws dimension_error If the input tensors do not have the same shape.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept XType, TensorConcept YType>
    requires requires {
        requires InSamePlace<XType, YType>;
        requires SameUnderlyingAndRank<XType, YType>;
    }
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {
    LabeledSection0();

    detail::axpy(alpha, X, Y);
}

/**
 * Scale and add two tensors together.
 *
 * @f[
 * \mathbf{y} := \alpha \mathbf{x} + \beta \mathbf{y}
 * @f]
 *
 * @tparam XType,YType The tensor types.
 * @param[in] alpha The scale factor for the input.
 * @param[in] X The input tensor.
 * @param[in] beta The scale factor for the output.
 * @param[inout] Y The output tensor.
 *
 * @throws rank_error If the input tensors do not have the same rank.
 * @throws dimension_error If the input tensors do not have the same shape.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept XType, TensorConcept YType>
    requires requires {
        requires InSamePlace<XType, YType>;
        requires SameUnderlyingAndRank<XType, YType>;
    }
void axpby(typename XType::ValueType alpha, XType const &X, typename XType::ValueType beta, YType *Y) {
    LabeledSection0();

    detail::axpby(alpha, X, beta, Y);
}

/**
 * Perform a rank-1 update. Neither vector is conjugated.
 *
 * @f[
 * \mathbf{A} := \alpha\mathbf{xy}^T + \mathbf{A}
 * @f]
 *
 * @tparam AType The type for the output matrix.
 * @tparam XYType The type for the input vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] X The left vector.
 * @param[in] Y The right vector.
 * @param[inout] A The output matrix.
 *
 * @throws rank_error If the X and Y tensors are not rank-1 or the A tensor is not rank-2.
 * @throws tensor_compat_error If the number of elements in the X vector is not the same as the number of rows in A, or the number of
 * elements in the Y vector is not the same as the number of columns of A.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <MatrixConcept AType, VectorConcept XType, VectorConcept YType>
    requires requires { requires SameUnderlying<AType, XType, YType>; }
void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    LabeledSection0();

    detail::ger(alpha, X, Y, A);
}

// Runtime-rank ger overload.
template <BasicTensorConcept AType, BasicTensorConcept XType, BasicTensorConcept YType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank || std::remove_cvref_t<XType>::Rank == einsums::dynamic_rank ||
                         std::remove_cvref_t<YType>::Rank == einsums::dynamic_rank;
        requires SameUnderlying<AType, XType, YType>;
    }
void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    LabeledSection0();
    detail::ger(alpha, X.impl(), Y.impl(), &A->impl());
}

/**
 * Perform a rank-1 update. The right vector is conjugated.
 *
 * @f[
 * \mathbf{A} := \alpha\mathbf{xy}^H + \mathbf{A}
 * @f]
 *
 * @tparam AType The type for the output matrix.
 * @tparam XYType The type for the input vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] X The left vector.
 * @param[in] Y The right vector.
 * @param[inout] A The output matrix.
 *
 * @throws rank_error If the X and Y tensors are not rank-1 or the A tensor is not rank-2.
 * @throws tensor_compat_error If the number of elements in the X vector is not the same as the number of rows in A, or the number of
 * elements in the Y vector is not the same as the number of columns of A.
 *
 * @versionadded{2.0.0}
 */
template <MatrixConcept AType, VectorConcept XType, VectorConcept YType>
    requires requires { requires SameUnderlying<AType, XType, YType>; }
void gerc(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    LabeledSection0();

    detail::gerc(alpha, X, Y, A);
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
 * @tparam TensorType The type for the tensor to decompose.
 * @tparam Pivots The type for the pivots.
 * @param[inout] A The tensor to decompose. On exit, it contains the U matrix above the diagonal and the L matrix below. The diagonal
 * entries of the L matrix are not stored and are all 1.
 * @param[out] pivot The pivots used during the decomposition.
 * @return 0 on success. If positive, the matrix is singular. This is a success, and the matrix is decomposed, but the result should not be
 * used to solve equations. If negative, one of the inputs to the underlying LAPACK call is invalid. The absolute value indicates which
 * parameter.
 *
 * @warning Do not ignore the return value. It tells you if your matrix is singular.
 *
 * @throws rank_error If the tensor input is not rank-2.
 * @throws std::invalid_argument If one of the values passed to the internal call was invalid.
 * @throws std::length_error If the pivot buffer type can not be resized and does not have enough space for the pivots.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      The pivots can now be any contiguous container of blas::int_t , such as vectors and arrays.
 *      It can also be spans.
 *      Can also handle non-unit strides and row- and column-major layouts.
 *      The BlockTensor implementation no longe creates temporary buffers for the pivots at each step. Instead, it uses spans to process
 *      sections of the pivots.
 * @endversion
 */
template <MatrixConcept TensorType, typename Pivots,
          bool          resizable = requires(Pivots a, typename Pivots::size_type size) { a.resize(size); }>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(CoreTensorConcept<TensorType>);
    }
[[nodiscard]] auto getrf(TensorType *A, Pivots *pivot) -> int {
    auto pivot_size = std::min(A->dim(0), A->dim(1));
    if constexpr (resizable) {

        if (pivot->size() < pivot_size) {
            pivot->resize(pivot_size);
        }
    } else {
        if (pivot->size() < pivot_size) {
            EINSUMS_THROW_EXCEPTION(std::length_error, "Pivot buffer too small and can not be resized!");
        }
    }
    return detail::getrf(A, pivot);
}

// Runtime-rank getrf overload: TensorImpl-level kernel runtime-checks rank-2.
template <BasicTensorConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        requires std::remove_cvref_t<TensorType>::Rank == einsums::dynamic_rank;
        typename Pivots::value_type;
        typename Pivots::size_type;
        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
    }
[[nodiscard]] auto getrf(TensorType *A, Pivots *pivot) -> int {
    auto pivot_size = std::min(A->dim(0), A->dim(1));
    if (pivot->size() < pivot_size) {
        pivot->resize(pivot_size);
    }
    return detail::getrf(&A->impl(), pivot);
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor.
 * @tparam Pivots The type for the pivots.
 * @param[inout] A The matrix to invert after being processed by getrf.
 * @param[in] pivot The pivot vector from getrf.
 *
 * @throws rank_error If the input tensor is not a matrix.
 * @throws dimension_error If the input matrix is not square.
 * @throws std::invalid_argument If an invalid argument got passed to the internal call. This likely means that the tensor is not
 * initialized.
 * @throws std::runtime_error If the matrix passed in is singular. This means that the return value from getrf was ignored.
 * @throws std::length_error If the pivot buffer is too small.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      The pivots can now be any contiguous container of blas::int_t , such as vectors and arrays. It can also be spans.
 *      Can also handle non-unit strides and row- and column-major layouts. Raises exceptions as opposed to returning a status code.
 *      The BlockTensor implementation no longe creates temporary buffers for the pivots at each step. Instead, it uses spans to process
 *      sections of the pivots.
 * @endversion
 */
template <MatrixConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(CoreTensorConcept<TensorType>);
    }
void getri(TensorType *A, Pivots const &pivot) {
    if (pivot.size() < std::min(A->dim(0), A->dim(1))) {
        EINSUMS_THROW_EXCEPTION(
            std::length_error,
            "The pivot buffer is not the right size! Make sure it has enough data and you call getrf before calling getri.");
    }

    detail::getri(A, pivot);
}

/**
 * @brief Inverts a matrix.
 *
 * Utilizes the LAPACK routines getrf and getri to invert a matrix.
 *
 * @tparam TensorType The type of the tensor.
 * @param[inout] A Matrix to invert. On exit, the inverse of A, assuming it is non-singular. If it is singular, it may be overwritten by the
 * LU decomposition.
 *
 * @throws rank_error If the input is not a matrix.
 * @throws dimension_error If the matrix is not square.
 * @throws std::runtime_error If the matrix is singular.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This function now throws errors instead of simply aborting on error. Can also handle non-unit strides and row- and column-major
 *      layouts.
 * @endversion
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
void invert(TensorType *A) {
    detail::invert(A);
}

// Runtime-rank invert overload: TensorImpl-level kernel runtime-checks
// rank-2 + square.
template <BasicTensorConcept TensorType>
    requires requires {
        requires std::remove_cvref_t<TensorType>::Rank == einsums::dynamic_rank;
        requires CoreTensorConcept<TensorType>;
    }
void invert(TensorType *A) {
    detail::invert(&A->impl());
}

template <SmartPointer SmartPtr>
void invert(SmartPtr *A) {
    LabeledSection0();

    invert(A->get());
}

/**
 * @brief Indicates the type of norm to compute.
 */
enum class APIARY_EXPOSE APIARY_MODULE("linalg") Norm : char{
    MAXABS    = 'M', /**< \f$val = max(abs(Aij))\f$, largest absolute value of the matrix A. */
    ONE       = '1', /**< \f$val = norm1(A)\f$, 1-norm of the matrix A (maximum column sum) */
    INFTY     = 'I', /**< \f$val = normI(A)\f$, infinity norm of the matrix A (maximum row sum) */
    FROBENIUS = 'F', /**< \f$val = normF(A)\f$, Frobenius norm of the matrix A (square root of sum of squares). */
    TWO       = '2'  /**< @f$val = \sqrt{max(\lambda(A^H A))}@f$, norm induced by the 2-norm of a vector. Also called the spectral norm. */
};

/**
 * @brief Computes the norm of a matrix.
 *
 * Returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real matrix A.
 *
 *
 * @code
 * using namespace einsums;
 *
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto norm = einsums::linear_algebra::norm(einsums::linear_algebra::Norm::One, A);
 * @endcode
 *
 * @tparam AType The type of the matrix
 * @param[in] norm_type where Norm::ONE denotes the one norm of a matrix (maximum column sum),
 *   Norm::INFTY denotes the infinity norm of a matrix  (maximum row sum),
 *   Norm::FROBENIUS denotes the Frobenius norm of a matrix (square root of sum of
 *   squares), and Norm::TWO denotes the induced 2-norm (square root of the maximum eigenvalue of A^H A).
 * @note Note that \f$ max(abs(A(i,j))) \f$ is not a consistent matrix norm.
 * @param[in] a The matrix to compute the norm of.
 * @return The requested norm of the matrix.
 *
 * @throws rank_error If the input is neither a matrix or vector.
 * @throws enum_error If an invalid enum value is passed.
 *
 * @versionaddeddesc{1.0.0}
 *      There is a bug in this version where, due to a difference in memory layout,
 *      the 1-norm and infinity-norm are switched.
 * @endversion
 * @versionchangeddesc{2.0.0}
 *      Renamed the members to be all caps to follow the recommended style.
 *      Fixed the bug in previous versions, and added the 2-norm. The function is also able to handle vectors,
 *      in which case the vector norms will be used. The norms for the vectors are the same ones that induce the
 *      matrix norms. This means that the MAXABS norm and the infinity-norm are the same, and the
 *      2-norm and the Frobenius norm are the same for vectors, since the MAXABS and Frobenius norms are not induced.
 *      Can handle non-unit strides and row- and column-major layouts.
 * @endversion
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto norm(Norm norm_type, AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    LabeledSection0();

    return detail::norm(static_cast<char>(norm_type), a);
}

// Runtime-rank norm overload: delegates to the TensorImpl form which
// already handles rank-1 (vector) and rank-2 (matrix) inputs.
template <BasicTensorConcept AType>
    requires requires {
        requires std::remove_cvref_t<AType>::Rank == einsums::dynamic_rank;
        requires CoreTensorConcept<AType>;
    }
[[nodiscard]] auto norm(Norm norm_type, AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    LabeledSection0();

    return detail::norm(static_cast<char>(norm_type), a.impl());
}

/**
 * Compute the Euclidean norm of a vector. For higher-rank tensors,
 * this is the same as taking the square root of the sum of the squares of all of the elements.
 *
 * @tparam AType The type of the input tensor.
 * @param[in] a The tensor to handle.
 *
 * @return The vector norm of the tensor.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Can now handle non-unit strides and both row- and colum-major layouts.
 * @endversion
 */
template <TensorConcept AType>
[[nodiscard]] auto vec_norm(AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    LabeledSection0();
    return detail::vec_norm(a);
}

/**
 * Compute modes for the singular value decomposition.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      The enum member names are now all caps to match the recommended style. The overwrite option is also removed since the
 *      input should be const and should not be overwritten.
 * @endversion
 */
enum class APIARY_EXPOSE APIARY_MODULE("linalg") Vectors : char{
    ALL  = 'A' /**< Compute all vectors. */,
    SOME = 'S' /**< Compute some of the vectors. The number is the smaller of the number of rows and columns. */,
    // Unused since the A tensor should be const.
    // OVERWRITE = 'O', /**< Put the output of U into the A tensor. */
    NONE = 'N' /**< Don't compute any vectors. */
};

/**
 * Computes the singular value decomposition.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\SigmaV}^T
 * @f]
 *
 * Uses the original svd function found in lapack, gesvd, request all left and right vectors.
 *
 * @tparam AType The type of the input tensor.
 * @param[in] A the matrix to decompose.
 * @param[in] jobu Whether to compute the U matrix.
 * @param[in] jobvt Whether to compute the transpose of the V matrix.
 *
 * @return A tuple containing the U matrix, singular value vector, and the transpose of the V matrix.
 *
 * @throws rank_error If the tensor being decomposed is not rank-2.
 * @throws std::invalid_argument If one of the parameters passed to the internal functions is invalid.
 * @throws std::runtime_error If the decomposition algorithm did not converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Now accepts a job specification and outputs optional tensors depending on the job.
 * @endversion
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto svd(AType const &A, Vectors jobu = Vectors::ALL, Vectors jobvt = Vectors::ALL)
    -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                  std::optional<Tensor<typename AType::ValueType, 2>>> {
    LabeledSection0();

    return detail::svd(A.impl(), static_cast<char>(jobu), static_cast<char>(jobvt));
}

/**
 * Computes the nullspace based on the singular value decomposition.
 *
 * @tparam AType The type of the matrix to decompose.
 * @param[in] _A The matrix to decompose.
 *
 * @throws std::invalid_argument If one of the parameters passed to the internal functions is invalid.
 * @throws std::runtime_error If the decomposition algorithm did not converge.
 *
 * @return The nullspace of the input matrix.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto svd_nullspace(AType const &_A) -> Tensor<typename AType::ValueType, 2> {
    using T = typename AType::ValueType;
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A{false, "A temp", _A.dim(0), _A.dim(1)};
    A = _A;

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.impl().get_lda();

    // Test if it is absolutely necessary to zero out these tensors first.
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>(false, "Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>(false, "superb", std::min(m, n));
    superb.zero();

    //    int info{0};
    int info = blas::gesvd('N', 'A', m, n, A.data(), lda, S.data(), (T *)nullptr, 1, Vt.data(), Vt.impl().get_lda(), superb.data());

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "svd: Argument {} has an invalid value.\n#3 (m) = {}, #4 (n) = {}, #6 (lda) = {}, #9 (ldu) = {}, #11 (ldvt) = {}", -info, m,
                n, lda, 1, Vt.impl().get_lda());
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "svd: Algorithm did not converge!");
        }
    }

    // Determine the rank of the nullspace matrix
    int rank = 0;
    for (int i = 0; i < std::min(n, m); i++) {
        if (S(i) > 1e-12) {
            rank++;
        }
    }

    // println("rank {}", rank);
    auto Vview     = Vt(Range{rank, Vt.dim(0)}, All);
    auto nullspace = Tensor<T, 2>(Vview.impl().transpose_view());

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace.dim(1); i++) {
        RemoveComplexT<T> norm = vec_norm(nullspace(All, i));
        if (norm > std::numeric_limits<RemoveComplexT<T>>::epsilon()) {
            scale_column(i, T{1.0} / norm, &nullspace);
        }
    }

    if constexpr (IsComplexV<T>) {
        einsums::detail::impl_conj(nullspace.impl());
    }

    return nullspace;
}

/**
 * Computes the singular value decomposition using the divide-and-conquer algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\SigmaV}^T
 * @f]
 *
 * @tparam AType The type of the input tensor.
 * @param[in] A the matrix to decompose.
 * @param[in] job Which vectors should be computed.
 *
 * @return A tuple containing the U matrix, singular value vector, and the transpose of the V matrix.
 *
 * @throws rank_error If the tensor being decomposed is not rank-2.
 * @throws std::invalid_argument If one of the parameters passed to the internal functions is invalid.
 * @throws std::runtime_error If the decomposition algorithm did not converge.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Now returns a tuple with optional tensors so that if Vectors::NONE is passed, the extra tensors are not allocated.
 * @endversion
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto svd_dd(AType const &A, Vectors job = Vectors::ALL)
    -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                  std::optional<Tensor<typename AType::ValueType, 2>>> {
    return detail::svd_dd(A, static_cast<char>(job));
}

/**
 * Perform the truncated singular value decomposition of a matrix.
 *
 * Randomized SVD with a fixed over-sampling factor of 5: the algorithm
 * draws an ``n x (k+5)`` sketch and runs a small dense SVD against it,
 * which requires ``A.dim(0) >= k + 5``. Calling with ``m < k + 5`` will
 * trip an out-of-range access inside the projection step. (See Halko,
 * Martinsson & Tropp, 2011, for the over-sampling rationale.)
 *
 * @tparam AType The type of the matrix.
 * @param[in] _A The matrix to decompose. Must have ``_A.dim(0) >= k + 5``.
 * @param[in] k The number of singular values to use.
 *
 * @return A tuple containing the vectors and singular values.
 *
 * @throws std::invalid_argument If a parameter passed to one of the internal calls is invalid.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto truncated_svd(AType const &_A, size_t k)
    -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                  Tensor<typename AType::ValueType, 2>> {
    using T = typename AType::ValueType;
    LabeledSection0();

    size_t m = _A.dim(0);
    size_t n = _A.dim(1);

    // Omega Test Matrix
    auto omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", m, k + 5);
    gemm<false, false>(T{1.0}, _A, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(m, k + 5));
    // Compute QR factorization of Y
    int info1 = blas::geqrf(m, k + 5, Y.data(), Y.impl().get_lda(), tau.data());

    if (info1 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to geqrf was invalid! #1 (m): {}, #2 (n): {}, #4 (lda): {}.",
                                print::ordinal(-info1), m, k + 5, Y.impl().get_lda());
    } else if (info1 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in geqrf.");
    }

    // Apply Q^T to A without explicitly forming Q, using ormqr/unmqr.
    // This avoids the O(m*(k+5)^2) cost of orgqr/ungqr.
    // B = Q^T * A: copy A into a workspace, apply Q^T from the left,
    // then extract the first (k+5) rows.
    Tensor<T, 2> A_work("A_work", m, n);
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            A_work(row, col) = _A(row, col);
        }
    }

    int info2;
    if constexpr (!IsComplexV<T>) {
        info2 = blas::ormqr('L', 'T', m, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), A_work.data(), A_work.impl().get_lda());
    } else {
        info2 = blas::unmqr('L', 'C', m, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), A_work.data(), A_work.impl().get_lda());
    }

    if (info2 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to ormqr/unmqr was invalid.", print::ordinal(-info2));
    } else if (info2 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in ormqr/unmqr.");
    }

    // Extract first (k+5) rows of Q^T * A into B.
    Tensor<T, 2> B("B", k + 5, n);
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < k + 5; ++row) {
            B(row, col) = A_work(row, col);
        }
    }

    // Perform svd on B
    auto [Utilde, S, Vt] = svd_dd(B);

    // Apply Q to Utilde to get U = Q * Utilde, again without forming Q.
    // Place Utilde ((k+5)×(k+5)) into an m×(k+5) workspace with zero padding.
    Tensor<T, 2> U("U", m, k + 5);
    U.zero();
    auto &Ut = Utilde.value();
    for (size_t col = 0; col < k + 5; ++col) {
        for (size_t row = 0; row < k + 5; ++row) {
            U(row, col) = Ut(row, col);
        }
    }

    int info3;
    if constexpr (!IsComplexV<T>) {
        info3 = blas::ormqr('L', 'N', m, k + 5, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), U.data(), U.impl().get_lda());
    } else {
        info3 = blas::unmqr('L', 'N', m, k + 5, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), U.data(), U.impl().get_lda());
    }

    if (info3 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to ormqr/unmqr was invalid.", print::ordinal(-info3));
    } else if (info3 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in ormqr/unmqr.");
    }

    return std::make_tuple(U, S, Vt.value());
}

/**
 * Perform the truncated eigendecomposition of a matrix.
 *
 * Randomized variant: like ``truncated_svd``, the algorithm uses an
 * over-sampling factor of 5 internally and requires ``A.dim(0) >= k + 5``.
 * Smaller inputs will trip an out-of-range access inside the projection
 * step. Results are *approximate* top-``k`` eigenpairs; expect some drift
 * from a full ``syev`` for tightly clustered eigenvalues.
 *
 * @tparam AType The type of the matrix.
 * @param[in] A The matrix to decompose. Must have ``A.dim(0) >= k + 5``.
 * @param[in] k The number of eigenvalues to use.
 *
 * @return A tuple containing the eigenvectors and eigenvalues.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto truncated_syev(AType const &A, size_t k)
    -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<typename AType::ValueType, 1>> {
    using T = typename AType::ValueType;
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Non-square matrix used as input of truncated_syev!");
    }

    size_t n = A.dim(0);

    // Omega Test Matrix
    Tensor<T, 2> omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", n, k + 5);
    gemm<false, false>(T{1.0}, A, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(n, k + 5));
    // Compute QR factorization of Y
    blas::int_t const info1 = blas::geqrf(n, k + 5, Y.data(), Y.impl().get_lda(), tau.data());

    if (info1 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to geqrf was invalid! #1 (m): {}, #2 (n): {}, #4 (lda): {}.",
                                print::ordinal(-info1), n, k + 5, Y.impl().get_lda());
    } else if (info1 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in geqrf.");
    }

    // Apply Q^T to A without forming Q, using ormqr/unmqr.
    // B = Q^T * A * Q: first compute Q^T * A, then multiply by Q from the right.

    // Step 1: Btemp = Q^T * A  (apply Q^T from the left to a copy of A)
    Tensor<T, 2> A_work("A_work", n, n);
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < n; ++row) {
            A_work(row, col) = A(row, col);
        }
    }

    blas::int_t info2;
    if constexpr (!IsComplexV<T>) {
        info2 = blas::ormqr('L', 'T', n, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), A_work.data(), A_work.impl().get_lda());
    } else {
        info2 = blas::unmqr('L', 'C', n, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), A_work.data(), A_work.impl().get_lda());
    }

    if (info2 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to ormqr/unmqr was invalid.", print::ordinal(-info2));
    } else if (info2 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in ormqr/unmqr.");
    }

    // Step 2: B = (Q^T * A) * Q, applying Q from the right to the first (k+5) rows of A_work
    // A_work is now Q^T * A (n×n). We need B = (first k+5 rows) * Q = Btemp * Q.
    // Use ormqr with side='R' on the first (k+5) rows.
    Tensor<T, 2> Btemp("Btemp", k + 5, n);
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < k + 5; ++row) {
            Btemp(row, col) = A_work(row, col);
        }
    }

    blas::int_t info2b;
    if constexpr (!IsComplexV<T>) {
        info2b =
            blas::ormqr('R', 'N', k + 5, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), Btemp.data(), Btemp.impl().get_lda());
    } else {
        info2b =
            blas::unmqr('R', 'N', k + 5, n, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), Btemp.data(), Btemp.impl().get_lda());
    }

    if (info2b < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to ormqr/unmqr was invalid.", print::ordinal(-info2b));
    } else if (info2b > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in ormqr/unmqr.");
    }

    // Extract (k+5)×(k+5) block: B = Btemp[:, :k+5]
    Tensor<T, 2> B("B", k + 5, k + 5);
    for (size_t col = 0; col < k + 5; ++col) {
        for (size_t row = 0; row < k + 5; ++row) {
            B(row, col) = Btemp(row, col);
        }
    }

    // Create buffer for eigenvalues
    Tensor<T, 1> w("eigenvalues", k + 5);

    // Diagonalize B
    syev(&B, &w);

    // Apply Q to B (eigenvectors) to get U = Q * B, without forming Q.
    // B is (k+5)×(k+5), result U is n×(k+5).
    Tensor<T, 2> U("U", n, k + 5);
    U.zero();
    for (size_t col = 0; col < k + 5; ++col) {
        for (size_t row = 0; row < k + 5; ++row) {
            U(row, col) = B(row, col);
        }
    }

    blas::int_t info3;
    if constexpr (!IsComplexV<T>) {
        info3 = blas::ormqr('L', 'N', n, k + 5, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), U.data(), U.impl().get_lda());
    } else {
        info3 = blas::unmqr('L', 'N', n, k + 5, tau.dim(0), Y.data(), Y.impl().get_lda(), tau.data(), U.data(), U.impl().get_lda());
    }

    if (info3 < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} parameter to ormqr/unmqr was invalid.", print::ordinal(-info3));
    } else if (info3 > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "An unknown error has occurred in ormqr/unmqr.");
    }

    return std::make_tuple(U, w);
}

// template <DiskTensorConcept AType>
//     requires requires {
//         requires MatrixConcept<AType>;
//         requires !einsums::IsComplexV<typename AType::ValueType>;
//     }
// auto truncated_syev(AType const &A, size_t in_k)
//     -> std::tuple<DiskTensor<typename AType::ValueType, 2>, Tensor<typename AType::ValueType, 1>> {
//     LabeledSection("truncated_syev");
//     using T = typename AType::ValueType;
//     // Davidson-Liu algorithm.

//     size_t k = in_k;

//     // First, create the output tensors.
//     DiskTensor<T, 2> evecs("/temp/syev", A.dim(0), std::min(A.dim(1), k));

//     evecs.write(create_random_tensor<T>("Eigenvectors", A.dim(0), std::min(A.dim(1), k)));

//     Tensor<T, 1> evals("Eigenvalues", std::min(A.dim(1), k));

//     // Orthonormalize the eigenvectors.
//     // Start by normalizing.
//     for (int i = 0; i < evecs.dim(1); i++) {
//         auto view = evecs(All, i);
//         auto norm = vec_norm(view.get());
//         view.get() /= norm;
//     }

//     // Then orthonormalizing.
//     for (int i = 1; i < evecs.dim(1); i++) {
//         auto dest_view = evecs(All, i);
//         for (int j = 0; j < i; j++) {
//             auto src_view = evecs(All, j);
//             auto overlap  = dot(src_view.get(), dest_view.get());

//             axpy(-overlap, src_view.get(), &dest_view.get());
//         }

//         auto norm = vec_norm(dest_view.get());
//         dest_view.get() /= norm;
//     }

//     BufferTensor<T, 2> subspace("subspace", evecs.dim(1), evecs.dim(1));

//     std::string name = fmt::format("/output/syev{}", A.name());
//     if (A.name().size() == 0) {
//         name = "";
//     }

//     DiskTensor<T, 2> temp(name, A.dim(0), k);

//     DiskTensor<T, 2>   temp2("/temp/syev", A.dim(0), k);
//     BufferTensor<T, 1> subspace_vals("subspace eigenvalues", evecs.dim(1));
//     BufferTensor<T, 2> correction("correction vector", A.dim(0), in_k), z("z intermediate", A.dim(0), in_k);

//     // Set up the convergence condition.
//     std::vector<bool> converged(k);

//     for (int i = 0; i < k; i++) {
//         converged[i] = false;
//     }

//     // Get the diagonal entries for computing the over-relaxation parameter.
//     BufferTensor<T, 1> diagonal("diagonal", A.dim(0));

//     for (int i = 0; i < A.dim(0) / 64; i++) {
//         auto  block_view   = A(Range{64 * i, 64 * (i + 1)}, Range{64 * i, 64 * (i + 1)});
//         auto &block_tensor = block_view.get();

//         diagonal(Range{64 * i, 64 * (i + 1)}) = block_tensor.tie_indices(0, 1);
//     }

//     if (A.dim(0) % 64 != 0) {
//         auto  block_view   = A(Range{64 * (A.dim(0) / 64), A.dim(0)}, Range{64 * (A.dim(0) / 64), A.dim(0)});
//         auto &block_tensor = block_view.get();

//         diagonal(Range{64 * (A.dim(0) / 64), A.dim(0)}) = block_tensor.tie_indices(0, 1);
//     }

//     do {
//         // Calculate the subspace matrix.
//         gemm('n', 'n', 1.0, A, evecs, 0.0, &temp2);
//         gemm('t', 'n', 1.0, evecs, temp2, 0.0, &subspace);

//         // Decompose the subspace.

//         syev(&subspace, &subspace_vals);

//         // Compute the approximate eigenvectors.
//         gemm('n', 'n', 1.0, evecs, subspace(All, Range{0, in_k}), 0.0, &temp);

//         // Compute the residuals.
//         size_t new_k = evecs.dim(1);
//         for (int i = 0; i < in_k; i++) {
//             // Compute the residual.
//             {
//                 auto temp_vec = temp(All, Range{0, in_k});
//                 correction    = temp_vec.get();

//                 gemv('n', T{1.0}, A, temp_vec, T{0.0}, &correction);
//                 axpy(-subspace_vals(i), temp_vec.get(), &correction);
//             }

//             // Check for convergence.
//             if constexpr (std::is_same_v<T, float>) {
//                 if (vec_norm(correction) < 1e-3) {
//                     converged[i] = true;
//                     continue;
//                 } else {
//                     converged[i] = false;
//                 }
//             } else {
//                 if (vec_norm(correction) < 1e-6) {
//                     converged[i] = true;
//                     continue;
//                 } else {
//                     converged[i] = false;
//                 }
//             }

//             // Solve for the correction.
//             for (size_t j = 0; j < A.dim(0); j++) {
//                 correction(j) /= subspace_vals(i) - diagonal(j);
//             }

//             // correction /= vec_norm(correction);

//             // Now that we have the correction, solve for the corrected correction.
//             z = correction; // Save the original correction.
//             for (int j = 0; j < new_k; j++) {
//                 auto  view = evecs(All, j);
//                 auto &tens = view.get();
//                 axpy(-dot(tens, z), tens, &correction);
//             }

//             // Check the norms.
//             auto norm = vec_norm(correction);
//             if (norm / vec_norm(z) > 1e-3) {
//                 correction /= norm;
//                 evecs.resize(A.dim(0), new_k + 1);
//                 auto view  = evecs(All, new_k);
//                 view.get() = correction;
//                 new_k++;
//                 break;
//             }
//         }

//         if (new_k != k) {
//             subspace.resize(new_k, new_k);
//             subspace_vals.resize(new_k);
//             temp2.resize(A.dim(0), new_k);
//         } else {
//             temp2 = evecs;
//             gemm('n', 'n', T{1.0}, temp2, subspace, T{0.0}, &evecs);
//         }

//         k = new_k;
//     } while (!std::all_of(converged.begin(), converged.end(), [](bool n) { return n; }));

//     // Calculate the actual eigenvectors.

//     // Calculate the subspace matrix.
//     gemm('n', 'n', 1.0, A, evecs, 0.0, &temp2);
//     gemm('t', 'n', 1.0, evecs, temp2, 0.0, &subspace);

//     // Decompose the subspace.

//     syev(&subspace, &subspace_vals);

//     evals = subspace_vals(Range{0, in_k});

//     // Compute the approximate eigenvectors.
//     gemm('n', 'n', 1.0, evecs, subspace(All, Range{0, in_k}), 0.0, &temp);

//     // Sort.
//     for (int i = 0; i < in_k; i++) {
//         T   min     = evals(i);
//         int min_pos = i;

//         for (int j = i + 1; j < in_k; j++) {
//             if (min > evals(j)) {
//                 min     = evals(j);
//                 min_pos = j;
//             }
//         }

//         if (min_pos != i) {
//             auto &vec1 = temp(All, min_pos).get();
//             auto &vec2 = temp(All, i).get();

//             std::swap(vec1, vec2);
//             std::swap(evals(i), evals(min_pos));
//         }
//     }

//     evecs.unlink();
//     temp2.unlink();

//     return std::make_tuple(std::move(temp), std::move(evals));
// }

/**
 * Compute the pseudoinverse of a matrix.
 *
 * @tparam AType The type of the matrix.
 * @tparam T The type for the tolerance.
 * @param[in] A The matrix to invert.
 * @param[in] tol The zero cutoff.
 *
 * @return The pseudoinverse of a matrix.
 *
 * @throws rank_error If the tensor being decomposed is not rank-2.
 * @throws std::invalid_argument If one of the parameters passed to the internal functions is invalid.
 * @throws std::runtime_error If the SVD algorithm did not converge.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType, typename T>
    requires requires {
        requires CoreTensorConcept<AType>;
        requires std::is_same_v<typename AType::ValueType, T>;
    }
inline auto pseudoinverse(AType const &A, T tol) -> Tensor<T, 2> {
    LabeledSection0();

    auto [U, S, Vh] = svd_dd(A);

    size_t new_dim{0};
    for (size_t v = 0; v < S.dim(0); v++) {
        T val = S(v);
        if (val > tol)
            scale_column(v, T{1.0} / val, &U.value());
        else {
            new_dim = v;
            break;
        }
    }

    TensorView<T, 2> U_view = U.value()(All, Range{0, new_dim});
    TensorView<T, 2> V_view = Vh.value()(Range{0, new_dim}, All);

    Tensor<T, 2> pinv("pinv", A.dim(0), A.dim(1));
    gemm<false, false>(1.0, U_view, V_view, 0.0, &pinv);

    return pinv;
}

/**
 * Solves a continuous Lyapunov equation.
 *
 * @f[
 *  \mathbf{AX} + \mathbf{XA}^H + \mathbf{Q} = 0
 * @f]
 *
 * @tparam AType The type of the A matrix.
 * @tparam QType The type of the Q matrix.
 * @param[in] A The A matrix.
 * @param[in] Q The Q matrix.
 *
 * @return The solution to the equation.
 *
 * @throws dimension_error If the input matrices are not square.
 * @throws tensor_compat_error If the input matrices do not have the same size.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType, MatrixConcept QType>
    requires requires {
        requires CoreTensorConcept<AType>;
        requires CoreTensorConcept<QType>;
        requires SameUnderlying<AType, QType>;
    }
inline auto solve_continuous_lyapunov(AType const &A, QType const &Q) -> Tensor<typename AType::ValueType, 2> {
    using T = typename AType::ValueType;
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "solve_continuous_lyapunov: Dimensions of A ({} x {}), do not match", A.dim(0), A.dim(1));
    }
    if (Q.dim(0) != Q.dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "solve_continuous_lyapunov: Dimensions of Q ({} x {}), do not match", Q.dim(0), Q.dim(1));
    }
    if (A.dim(0) != Q.dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "solve_continuous_lyapunov: Dimensions of A ({} x {}) and Q ({} x {}), do not match",
                                A.dim(0), A.dim(1), Q.dim(0), Q.dim(1));
    }

    size_t n = A.dim(0);

    /// @todo Break this off into a separate schur function
    // Compute Schur Decomposition of A
    Tensor<T, 2> R(false, "A copy", n, n);
    R = A; // R is a copy of A
    Tensor<T, 2>              wr(false, "Schur Real Buffer", n, n);
    Tensor<T, 2>              wi(false, "Schur Imaginary Buffer", n, n);
    Tensor<T, 2>              U(false, "Lyapunov U", n, n);
    BufferVector<blas::int_t> sdim(1);
    blas::gees('V', n, R.data(), R.impl().get_lda(), sdim.data(), wr.data(), wi.data(), U.data(), U.impl().get_lda());

    // Compute F = U^T * Q * U
    Tensor<T, 2> Fbuff = gemm<true, false>(1.0, U, Q);
    Tensor<T, 2> F(false, "F matrix", n, n);
    gemm<false, false>(1.0, Fbuff, U, 0.0, &F);

    // Call the Sylvester Solve
    BufferVector<T> scale(1);
    blas::trsyl('N', 'N', 1, n, n, const_cast<T const *>(R.data()), R.impl().get_lda(), const_cast<T const *>(R.data()), R.impl().get_lda(),
                F.data(), F.impl().get_lda(), scale.data());

    Tensor<T, 2> Xbuff = gemm<false, false>(scale[0], U, F);
    Tensor<T, 2> X     = gemm<false, true>(1.0, Xbuff, U);

    return X;
}

/// @todo Bring this back
/// <tt>ALIAS_TEMPLATE_FUNCTION(solve_lyapunov, solve_continuous_lyapunov)</tt>

/**
 * Perform QR decomposition.
 *
 * @f[
 * \mathbf{A} = \mathbf{QR}
 * @f]
 *
 * Here, @f$\mathbf{A}@f$ is a matrix, @f$\mathbf{Q}@f$ is a unitary matrix, and @f$\mathbf{R}@f$ is an upper triangular matrix.
 *
 * @tparam AType The matrix type.
 * @param[in] A The matrix to decompose.
 *
 * @return The Q and R matrices.
 *
 * @throws std::invalid_argument If one of the parameters passed to the internal functions is invalid.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This function no longer returns the output of geqrf. It now returns the Q and R matrices. This means that the old q function
 *      no longer needs to exist.
 * @endversion
 */
template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto qr(AType const &A) -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<typename AType::ValueType, 2>> {
    return detail::qr(A);
}

/**
 * Compute the direct product between two tensors and accumulate them into another.
 *
 * @f[
 *  c_i := \alpha a_i b_i + \beta c_i
 * @f]
 *
 * @tparam AType The type of the A tensor.
 * @tparam BType The type of the B tensor.
 * @tparam CType The type of the C tensor.
 * @tparam T The type of the scale factors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] A,B The tensors being multiplied.
 * @param[in] beta The scale factor for the output.
 * @param[out] C The output tensor.
 *
 * @versionadded{1.0.0}
 */
template <TensorConcept AType, TensorConcept BType, TensorConcept CType, typename T>
    requires(SameRank<AType, BType, CType>)
void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    LabeledSection0();

    detail::direct_product(alpha, A, B, beta, C);
}

/**
 * Compute the element-wise (Hadamard) quotient of two tensors and accumulate it
 * into another. The counterpart to @ref direct_product, for building reciprocals
 * and quotients (e.g. amplitude denominators @f$1/\Delta@f$) without a per-element
 * host callback.
 *
 * @f[
 *  c_i := \alpha \frac{a_i}{b_i} + \beta c_i
 * @f]
 *
 * @tparam AType The type of the A (numerator) tensor.
 * @tparam BType The type of the B (denominator) tensor.
 * @tparam CType The type of the C tensor.
 * @tparam T The type of the scale factors.
 * @param[in] alpha The scale factor for the quotient.
 * @param[in] A,B The numerator and denominator tensors.
 * @param[in] beta The scale factor for the output.
 * @param[out] C The output tensor.
 *
 * @versionadded{2.0.0}
 */
template <TensorConcept AType, TensorConcept BType, TensorConcept CType, typename T>
    requires(SameRank<AType, BType, CType>)
void direct_division(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    LabeledSection0();

    detail::direct_division(alpha, A, B, beta, C);
}

/**
 * Computes the determinant of a matrix.
 *
 * @tparam AType The type of the matrix.
 * @param[in] A The matrix to analyze.
 *
 * @return The determinant of the matrix.
 *
 * @throws dimension_error If the input matrix is not square.
 *
 * @versionadded{1.0.0}
 */
template <MatrixConcept AType>
typename AType::ValueType det(AType const &A) {
    using T = typename AType::ValueType;
    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only take the determinant of a square matrix.");
    }

    RemoveViewT<AType> temp = A;

    BufferVector<blas::int_t> pivots;
    int                       singular = getrf(&temp, &pivots);
    if (singular > 0) {
        return T{0.0}; // Matrix is singular, so it has a determinant of zero.
    }

    T ret{1.0};

    int parity = 0;

    // Calculate the effect of the pivots.
    for (int i = 0; i < A.dim(0); i++) {
        if (pivots[i] != i + 1) {
            parity++;
        }
    }

    // Calculate the contribution of the diagonal elements.
#pragma omp parallel for simd reduction(* : ret)
    for (int i = 0; i < A.dim(0); i++) {
        ret *= temp(i, i);
    }

    if (parity % 2 == 1) {
        ret *= T{-1.0};
    }

    return ret;
}
} // namespace einsums::linear_algebra