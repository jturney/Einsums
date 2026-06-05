//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/Print.hpp>

#include <span>
#include <stdexcept>

namespace einsums::linear_algebra::detail {

template <BlockTensorConcept AType, typename T = RemoveComplexT<typename AType::ValueType>>
void sum_square(AType const &A, T *scale, T *sumsq) {

    for (size_t i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }

        sum_square(A.block(i), scale, sumsq);
    }
}

template <BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename U>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "gemm: Tensors need the same number of blocks.");
    }

    using T = typename AType::ValueType;

    bool tA = std::tolower(transA) != 'n', tB = std::tolower(transB) != 'n';

    if (!strchr("CNTcnt", transA) || !strchr("CNTcnt", transB)) {
        EINSUMS_THROW_EXCEPTION(
            std::invalid_argument,
            "The transposition parameters were invalid! They must be either n, c, or t, case insensitive. Got transA: {} and transB: {}.",
            transA, transB);
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        gemm(transA, transB, static_cast<T>(alpha), A.block(i), B.block(i), static_cast<T>(beta), &(C->block(i)));
    }
}

template <bool TransA, bool TransB, BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename U>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    gemm((TransA) ? 't' : 'n', (TransB) ? 't' : 'n', alpha, A, B, beta, C);
}

template <bool TransA, BlockTensorConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    using T = typename AType::ValueType;
    if (beta == U(0.0)) {
        y->zero();
    } else {
        *y *= beta;
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        gemv<TransA>(static_cast<T>(alpha), A.block(i), x(A.block_range(i)), static_cast<T>(1.0), &((*y)(A.block_range(i))));
    }
}

template <BlockTensorConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    using T = typename AType::ValueType;
    if (beta == U(0.0)) {
        y->zero();
    } else {
        *y *= beta;
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        gemv(transA, static_cast<T>(alpha), A.block(i), x(A.block_range(i)), static_cast<T>(1.0), &((*y)(A.block_range(i))));
    }
}

template <bool ComputeEigenvectors = true, BlockTensorConcept AType, VectorConcept WType>
    requires requires {
        requires SameUnderlying<AType, WType>;
        requires NotComplex<AType>;
        requires MatrixConcept<AType>;
    }
void syev(AType *A, WType *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        syev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <BlockTensorConcept AType, VectorConcept WType, typename LVecPtr, typename RVecPtr>
    requires requires {
        requires std::is_same_v<AddComplexT<typename AType::ValueType>, typename WType::ValueType>;
        requires MatrixConcept<AType>;
        requires std::is_null_pointer_v<LVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<LVecPtr>> && BlockTensorConcept<std::remove_pointer_t<LVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<LVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
        requires std::is_null_pointer_v<RVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<RVecPtr>> && BlockTensorConcept<std::remove_pointer_t<RVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<RVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
    }
void geev(AType *A, WType *W, LVecPtr lvecs, RVecPtr rvecs) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        decltype(lvecs->block(0)) *lvec_block = nullptr;
        decltype(rvecs->block(0)) *rvec_block = nullptr;
        if (A->block_dim(i) == 0) {
            continue;
        }

        if (lvecs != nullptr) {
            lvec_block = &(lvecs->block(i));
        }
        if (rvecs != nullptr) {
            rvec_block = &(rvecs->block(i));
        }
        auto out_block = (*W)(A->block_range(i));
        geev(&(A->block(i)), &out_block, lvec_block, rvec_block);
    }
}

template <bool ComputeEigenvectors = true, BlockTensorConcept AType, VectorConcept WType>
    requires requires {
        requires MatrixConcept<AType>;
        requires Complex<AType>;
        requires std::is_same_v<RemoveComplexT<typename AType::ValueType>, typename WType::ValueType>;
    }
void heev(AType *A, WType *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        heev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType>;
    }
auto gesv(AType *A, BType *B) -> int {
    if (A->num_blocks() != B->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "gesv: Tensors need the same number of blocks.");
    }

    int info_out = 0;

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        int info = gesv(&(A->block(i)), &(B->block(i)));

        info_out |= info;

        if (info != 0) {
            EINSUMS_LOG_WARN("gesv: Got non-zero return: %d", info);
        }
    }

    return info_out;
}

template <BlockTensorConcept AType, MatrixConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, BType>;
    }
auto gesv(AType *A, BType *B) -> int {

    if (A->dim(0) != B->dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error,
                                "The result matrix for gesv needs to have the same number of rows as the coefficient matrix.");
    }

    int info_out = 0;

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        int info = gesv(&(A->block(i)), &((*B)(A->block_range(i), All)));

        info_out |= info;

        if (info != 0) {
            EINSUMS_LOG_WARN("gesv: Got non-zero return: %d", info);
        }
    }

    return info_out;
}

template <BlockTensorConcept AType, VectorConcept BType>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, BType>;
    }
auto gesv(AType *A, BType *B) -> int {

    if (A->dim(0) != B->dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error,
                                "The result matrix for gesv needs to have the same number of rows as the coefficient matrix.");
    }

    int info_out = 0;

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        int info = gesv(&(A->block(i)), &((*B)(A->block_range(i))));

        info_out |= info;

        if (info != 0) {
            EINSUMS_LOG_WARN("gesv: Got non-zero return: %d", info);
        }
    }

    return info_out;
}

template <BlockTensorConcept AType>
void scale(typename AType::ValueType alpha, AType *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        scale(alpha, &(A->block(i)));
    }
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
void scale_row(size_t row, typename AType::ValueType alpha, AType *A) {
    int  block_ind = A->block_of(row);
    auto temp      = A->block(block_ind)(row - A->block_range(block_ind)[0], All);
    scale(alpha, &temp);
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
void scale_column(size_t column, typename AType::ValueType alpha, AType *A) {
    int  block_ind = A->block_of(column);
    auto temp      = A->block(block_ind)(All, column - A->block_range(block_ind)[0]);
    scale(alpha, &temp);
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires SameRank<AType, BType>
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (A.num_blocks() != B.num_blocks()) {
        return dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    if (A.ranges() != B.ranges()) {
        return dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += dot(A.block(i), B.block(i));
    }

    return out;
}

template <BlockTensorConcept AType, BlockTensorConcept BType>
    requires SameRank<AType, BType>
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (A.num_blocks() != B.num_blocks()) {
        return true_dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    if (A.ranges() != B.ranges()) {
        return true_dot(typename AType::StoredType(A), typename BType::StoredType(B));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += true_dot(A.block(i), B.block(i));
    }

    return out;
}

template <BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType>
    requires SameRank<AType, BType, CType>
auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C.num_blocks() || B.num_blocks() != C.num_blocks()) {
        return dot(AType::StoredType(A), BType::StoredType(B), CType::StoredType(C));
    }

    if (A.ranges() != B.ranges() || A.ranges() != C.ranges() || B.ranges() != C.ranges()) {
        return dot(AType::StoredType(A), BType::StoredType(B), CType::StoredType(C));
    }

    using T = BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType>;

    T out{0};

#pragma omp parallel for reduction(+ : out)
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        out += dot(A.block(i), B.block(i), C.block(i));
    }

    return out;
}

template <BlockTensorConcept XType, BlockTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "axpy: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "axpy: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }

        axpy(alpha, X[i], &(Y->block(i)));
    }
}

template <BlockTensorConcept XType, BlockTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpby(typename XType::ValueType alpha, XType const &X, typename YType::ValueType beta, YType *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "axpby: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "axpby: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }
        axpby(alpha, X[i], beta, &(Y->block(i)));
    }
}

template <BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        direct_product(alpha, A.block(i), B.block(i), beta, &(C->block(i)));
    }
}

template <BlockTensorConcept AType>
    requires MatrixConcept<AType>
auto pow(AType const &a, typename AType::ValueType alpha,
         typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon()) -> RemoveViewT<AType> {
    RemoveViewT<AType> out{"pow result", a.vector_dims()};

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < a.num_blocks(); i++) {
        if (a.block_dim(i) == 0) {
            continue;
        }
        out[i] = pow(a[i], alpha, cutoff);
    }

    return out;
}

template <MatrixConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(BlockTensorConcept<TensorType>);
    }
[[nodiscard]] auto getrf(TensorType *A, Pivots *pivot) -> int {
    blas::int_t out_value = 0;

    for (size_t i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }

        std::span<blas::int_t> pivot_view(std::next(pivot->begin(), A->block_range(i)[0]), A->block_dim(i));
        int                    ret = getrf(A->block(i), &pivot_view);

        if (ret > 0 && out_value == 0) {
            out_value = ret + A->block_range(i)[0];
        }
    }

    return out_value;
}

template <MatrixConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(BlockTensorConcept<TensorType>);
    }
void getri(TensorType *A, Pivots const &pivot) {
    for (size_t i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        std::span<blas::int_t> pivot_view(std::next(pivot->begin(), A->block_range(i)[0]), A->block_dim(i));

        getri(A->block(i), pivot_view);
    }
}

template <MatrixConcept TensorType>
    requires(BlockTensorConcept<TensorType>)
void invert(TensorType *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        invert(A->block(i));
    }
}

} // namespace einsums::linear_algebra::detail