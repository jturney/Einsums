#pragma once

#include "einsums/BlockTensor.hpp"
#include "einsums/linear_algebra_imp/BaseLinearAlgebra.hpp"
#include "einsums/utility/TensorTraits.hpp"

#ifdef __HIP__
#    include "einsums/_GPUUtils.hpp"
#endif

namespace einsums::linear_algebra::detail {

template<bool TransA, bool TransB, BlockTensorConcept AType, BlockTensorConcept BType, BlockTensorConcept CType, typename U>
requires requires {
    requires SameUnderlyingAndRank<AType, BType, CType>;
    requires MatrixConcept<AType>;
    requires std::convertible_to<U, typename AType::data_type>;
}
void gemm(const U alpha, const AType &A, const BType &B, const U beta, CType *C) {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
        throw std::runtime_error("gemm: Tensors need the same number of blocks.");
    }

    using T = typename AType::data_type;

#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceTensorV<AType>) {
        using namespace einsums::gpu;

        using dev_datatype = typename AType::dev_datatype;
        dev_datatype *alpha_gpu, *beta_gpu;

        hip_catch(hipMalloc((void **)&alpha_gpu, sizeof(dev_datatype)));
        hip_catch(hipMalloc((void **)&beta_gpu, sizeof(dev_datatype)));

        hip_catch(hipMemcpy((void *)alpha_gpu, &alpha, sizeof(dev_datatype), hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)beta_gpu, &beta, sizeof(dev_datatype), hipMemcpyHostToDevice));
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>((T *)alpha_gpu, A.block(i), B.block(i), (T *)beta_gpu, &(C->block(i)));
        }

        hip_catch(hipFree((void *)alpha_gpu));
        hip_catch(hipFree((void *)beta_gpu));
    } else {
#endif

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>(static_cast<T>(alpha), A.block(i), B.block(i), static_cast<T>(1.0), &(C->block(i)));
        }
#ifdef __HIP__
    }
#endif
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T, typename U>
    requires requires {
        requires RankBlockTensor<AType<T, ARank>, 2, T>;
        requires std::convertible_to<U, T>; // Make sure the alpha and beta can be converted to T
    }
void gemv(const U alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const U beta, YType<T, XYRank> *y) {
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

template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires RankBlockTensor<AType<T, ARank>, 2, T>;
        requires RankBasicTensor<WType<T, WRank>, 1, T>;
        requires !Complex<T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        syev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires RankBlockTensor<AType<T, ARank>, 2, T>;
        requires RankBasicTensor<WType<T, WRank>, 1, T>;
        requires !Complex<T>;
    }
void geev(AType<T, ARank> *A, WType<AddComplexT<T>, WRank> *W, AType<T, ARank> *lvecs, AType<T, ARank> *rvecs) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        geev<ComputeEigenvectors>(&(A->block(i)), &out_block, lvecs, rvecs);
    }
}

template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires RankBlockTensor<AType<T, ARank>, 2, T>;
        requires RankBasicTensor<WType<RemoveComplexT<T>, WRank>, 1, WType<RemoveComplexT<T>, WRank>>;
        requires Complex<T>;
    }
void heev(AType<T, ARank> *A, WType<RemoveComplexT<T>, WRank> *W) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        auto out_block = (*W)(A->block_range(i));
        heev<ComputeEigenvectors>(&(A->block(i)), &out_block);
    }
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires RankBlockTensor<AType<T, ARank>, 2, T>;
        requires RankBlockTensor<BType<T, BRank>, 2, T>;
    }
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B) -> int {
    if (A->num_blocks() != B->num_blocks()) {
        throw std::runtime_error("gesv: Tensors need the same number of blocks.");
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
            println("gesv: Got non-zero return: %d", info);
        }
    }

    return info_out;
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires RankBlockTensor<AType<T, ARank>, ARank, T>
void scale(T alpha, AType<T, ARank> *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A->num_blocks(); i++) {
        if (A->block_dim(i) == 0) {
            continue;
        }
        scale(alpha, &(A->block(i)));
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires RankBlockTensor<AType<T, ARank>, 2, T>
void scale_row(size_t row, T alpha, AType<T, ARank> *A) {
    int  block_ind = A->block_of(row);
    auto temp      = A->block(block_ind)(row - A->block_range(block_ind)[0], AllT());
    scale(alpha, &temp);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires RankBlockTensor<AType<T, ARank>, 2, T>
void scale_column(size_t column, T alpha, AType<T, ARank> *A) {
    int  block_ind = A->block_of(column);
    auto temp      = A->block(block_ind)(AllT(), column - A->block_range(block_ind)[0]);
    scale(alpha, &temp);
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<AType<T, Rank>, Rank, T>;
        requires RankBlockTensor<BType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {
    if (A.num_blocks() != B.num_blocks()) {
        return dot(typename AType<T, Rank>::tensor_type(A), typename BType<T, Rank>::tensor_type(B));
    }

    if (A.ranges() != B.ranges()) {
        return dot(typename AType<T, Rank>::tensor_type(A), typename BType<T, Rank>::tensor_type(B));
    }

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

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<AType<T, Rank>, Rank, T>;
        requires RankBlockTensor<BType<T, Rank>, Rank, T>;
    }
auto true_dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {
    if (A.num_blocks() != B.num_blocks()) {
        return true_dot(typename AType<T, Rank>::tensor_type(A), typename BType<T, Rank>::tensor_type(B));
    }

    if (A.ranges() != B.ranges()) {
        return true_dot(typename AType<T, Rank>::tensor_type(A), typename BType<T, Rank>::tensor_type(B));
    }

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

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<AType<T, Rank>, Rank, T>;
        requires RankBlockTensor<BType<T, Rank>, Rank, T>;
        requires RankBlockTensor<CType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B, const CType<T, Rank> &C) -> T {
    if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C.num_blocks() || B.num_blocks() != C.num_blocks()) {
        return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
    }

    if (A.ranges() != B.ranges() || A.ranges() != C.ranges() || B.ranges() != C.ranges()) {
        return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
    }

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

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<XType<T, Rank>, Rank, T>;
        requires RankBlockTensor<YType<T, Rank>, Rank, T>;
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        throw std::runtime_error("axpy: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        throw std::runtime_error("axpy: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }

        axpy(alpha, X[i], &(Y->block(i)));
    }
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<XType<T, Rank>, Rank, T>;
        requires RankBlockTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {

    if (X.num_blocks() != Y->num_blocks()) {
        throw std::runtime_error("axpby: Tensors need to have the same number of blocks.");
    }

    if (X.ranges() != Y->ranges()) {
        throw std::runtime_error("axpby: Tensor blocks need to be compatible.");
    }

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < X.num_blocks(); i++) {
        if (X.block_dim(i) == 0) {
            continue;
        }
        axpby(alpha, X[i], beta, &(Y->block(i)));
    }
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires RankBlockTensor<AType<T, Rank>, Rank, T>;
        requires RankBlockTensor<BType<T, Rank>, Rank, T>;
        requires RankBlockTensor<CType<T, Rank>, Rank, T>;
    }
void direct_product(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < A.num_blocks(); i++) {
        if (A.block_dim(i) == 0) {
            continue;
        }
        direct_product(alpha, A.block(i), B.block(i), beta, &(C->block(i)));
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires RankBlockTensor<AType<T, ARank>, 2, T>
auto pow(const AType<T, ARank> &a, T alpha, T cutoff = std::numeric_limits<T>::epsilon()) -> remove_view_t<AType<T, 2>> {
    remove_view_t<AType<T, 2>> out{"pow result", a.vector_dims()};

    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < a.num_blocks(); i++) {
        if (a.block_dim(i) == 0) {
            continue;
        }
        out[i] = pow(a[i], alpha, cutoff);
    }

    return out;
}

} // namespace einsums::linear_algebra::detail