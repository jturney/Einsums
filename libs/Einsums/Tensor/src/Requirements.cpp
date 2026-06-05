//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/ArithmeticTensor.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/FunctionTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

using namespace einsums;

/**
 * @brief Checks to make sure that all tensors satisfy the concepts they are expected to and none of the ones they are not expected to.
 *
 * Contains a whole bunch of static asserts. Each assertion checks whether a tensor satisfies a concept or does not satisfy a concept.
 * This is to ensure that changes to concepts or tensors will still keep compile-time deduction as it should be expected to perform.
 * For instance, we wouldn't want a change to Tensor to make it so it could not be used with an einsum call.
 */
template <typename T, typename BadT, size_t Rank, size_t BadRank>
static void check_requirements() {
    if constexpr (std::is_same_v<T, BadT> || Rank == BadRank) {
        return;
    } else {
        if constexpr (Rank >= 2) {
            static_assert(TensorConcept<BlockTensor<T, Rank>>);
            static_assert(!NotTensorConcept<BlockTensor<T, Rank>>);
            static_assert(TypedTensorConcept<BlockTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<BlockTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<BlockTensor<T, Rank>>);
            static_assert(RankTensorConcept<BlockTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<BlockTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<BlockTensor<T, Rank>>);
            static_assert(LockableConcept<BlockTensor<T, Rank>>);
            static_assert(TRTensorConcept<BlockTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<BlockTensor<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TensorViewConcept<BlockTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<BlockTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockTensor<T, Rank>>);
            static_assert(!TiledTensorConcept<BlockTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockTensor<T, Rank>>);
            static_assert(BlockTensorConcept<BlockTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<BlockTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<BlockTensor<T, Rank>>);
            static_assert(requires(BlockTensor<T, Rank> tensor) { println(tensor); });
        }

        static_assert(TensorConcept<DiskTensor<T, Rank>>);
        static_assert(!NotTensorConcept<DiskTensor<T, Rank>>);
        static_assert(TypedTensorConcept<DiskTensor<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DiskTensor<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DiskTensor<T, Rank>>);
        static_assert(RankTensorConcept<DiskTensor<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DiskTensor<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DiskTensor<T, Rank>>);
        static_assert(LockableConcept<DiskTensor<T, Rank>>);
        static_assert(TRTensorConcept<DiskTensor<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DiskTensor<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DiskTensor<T, Rank>>);
        static_assert(DiskTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TensorViewConcept<DiskTensor<T, Rank>>);
        static_assert(!BasicTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!CollectedTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskTensor<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskTensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskTensor<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(!FunctionTensorConcept<DiskTensor<T, Rank>>);
        }
        static_assert(!AlgebraTensorConcept<DiskTensor<T, Rank>>);
        static_assert(requires(DiskTensor<T, Rank> tensor) { println(tensor); });

        static_assert(TensorConcept<DiskView<T, Rank>>);
        static_assert(!NotTensorConcept<DiskView<T, Rank>>);
        static_assert(TypedTensorConcept<DiskView<T, Rank>, T>);
        static_assert(!TypedTensorConcept<DiskView<T, Rank>, BadT>);
        static_assert(RankTensorConcept<DiskView<T, Rank>>);
        static_assert(RankTensorConcept<DiskView<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<DiskView<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<DiskView<T, Rank>>);
        static_assert(LockableConcept<DiskView<T, Rank>>);
        static_assert(TRTensorConcept<DiskView<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<DiskView<T, Rank>, Rank, T>);
        static_assert(!CoreTensorConcept<DiskView<T, Rank>>);
        static_assert(DiskTensorConcept<DiskView<T, Rank>>);
        static_assert(TensorViewConcept<DiskView<T, Rank>>);
        static_assert(!BasicTensorConcept<DiskView<T, Rank>>);
        static_assert(!CollectedTensorConcept<DiskView<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskView<T, Rank>>);
        static_assert(!TiledTensorConcept<DiskView<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskView<T, Rank>>);
        static_assert(!BlockTensorConcept<DiskView<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<DiskView<T, Rank>>);
        }
        static_assert(!AlgebraTensorConcept<DiskView<T, Rank>>);
        static_assert(requires(DiskView<T, Rank> tensor) { println(tensor); });

        if constexpr (Rank == 2) {
            static_assert(TensorConcept<KroneckerDelta<T>>);
            static_assert(!NotTensorConcept<KroneckerDelta<T>>);
            static_assert(TypedTensorConcept<KroneckerDelta<T>, T>);
            static_assert(!TypedTensorConcept<KroneckerDelta<T>, BadT>);
            static_assert(RankTensorConcept<KroneckerDelta<T>>);
            static_assert(RankTensorConcept<KroneckerDelta<T>, Rank>);
            static_assert(!RankTensorConcept<KroneckerDelta<T>, BadRank>);
            static_assert(!BasicLockableConcept<KroneckerDelta<T>>);
            static_assert(!LockableConcept<KroneckerDelta<T>>);
            static_assert(TRTensorConcept<KroneckerDelta<T>, Rank, T>);
            static_assert(!TRLTensorConcept<KroneckerDelta<T>, Rank, T>);
            static_assert(!CoreTensorConcept<KroneckerDelta<T>>);
            static_assert(!DiskTensorConcept<KroneckerDelta<T>>);
            static_assert(!TensorViewConcept<KroneckerDelta<T>>);
            static_assert(!BasicTensorConcept<KroneckerDelta<T>>);
            static_assert(!CollectedTensorConcept<KroneckerDelta<T>>);
            static_assert(!TiledTensorConcept<KroneckerDelta<T>>);
            static_assert(!TiledTensorConcept<KroneckerDelta<T>, Tensor<T, Rank>>);
            static_assert(!BlockTensorConcept<KroneckerDelta<T>>);
            static_assert(!BlockTensorConcept<KroneckerDelta<T>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<KroneckerDelta<T>>);
            static_assert(!AlgebraTensorConcept<KroneckerDelta<T>>);
            static_assert(requires(KroneckerDelta<T> tensor) { println(tensor); });
        }

        static_assert(TensorConcept<RuntimeTensor<T>>);
        static_assert(!NotTensorConcept<RuntimeTensor<T>>);
        static_assert(TypedTensorConcept<RuntimeTensor<T>, T>);
        static_assert(!TypedTensorConcept<RuntimeTensor<T>, BadT>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>, Rank>);
        static_assert(!RankTensorConcept<RuntimeTensor<T>, BadRank>);
        static_assert(BasicLockableConcept<RuntimeTensor<T>>);
        static_assert(LockableConcept<RuntimeTensor<T>>);
        static_assert(!TRTensorConcept<RuntimeTensor<T>, Rank, T>);
        static_assert(!TRLTensorConcept<RuntimeTensor<T>, Rank, T>);
        static_assert(CoreTensorConcept<RuntimeTensor<T>>);
        static_assert(!DiskTensorConcept<RuntimeTensor<T>>);
        static_assert(!TensorViewConcept<RuntimeTensor<T>>);
        static_assert(BasicTensorConcept<RuntimeTensor<T>>);
        static_assert(!CollectedTensorConcept<RuntimeTensor<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensor<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensor<T>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<RuntimeTensor<T>>);
        static_assert(!BlockTensorConcept<RuntimeTensor<T>, Tensor<T, Rank>>);
        static_assert(!FunctionTensorConcept<RuntimeTensor<T>>);
        static_assert(!AlgebraTensorConcept<RuntimeTensor<T>>);
        static_assert(requires(RuntimeTensor<T> tensor) { println(tensor); });

        static_assert(TensorConcept<RuntimeTensorView<T>>);
        static_assert(!NotTensorConcept<RuntimeTensorView<T>>);
        static_assert(TypedTensorConcept<RuntimeTensorView<T>, T>);
        static_assert(!TypedTensorConcept<RuntimeTensorView<T>, BadT>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>, Rank>);
        static_assert(!RankTensorConcept<RuntimeTensorView<T>, BadRank>);
        static_assert(BasicLockableConcept<RuntimeTensorView<T>>);
        static_assert(LockableConcept<RuntimeTensorView<T>>);
        static_assert(!TRTensorConcept<RuntimeTensorView<T>, Rank, T>);
        static_assert(!TRLTensorConcept<RuntimeTensorView<T>, Rank, T>);
        static_assert(CoreTensorConcept<RuntimeTensorView<T>>);
        static_assert(!DiskTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TensorViewConcept<RuntimeTensorView<T>>);
        static_assert(BasicTensorConcept<RuntimeTensorView<T>>);
        static_assert(!CollectedTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensorView<T>>);
        static_assert(!TiledTensorConcept<RuntimeTensorView<T>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<RuntimeTensorView<T>>);
        static_assert(!BlockTensorConcept<RuntimeTensorView<T>, Tensor<T, Rank>>);
        static_assert(!FunctionTensorConcept<RuntimeTensorView<T>>);
        static_assert(!AlgebraTensorConcept<RuntimeTensorView<T>>);
        static_assert(requires(RuntimeTensorView<T> tensor) { println(tensor); });

        static_assert(TensorConcept<Tensor<T, Rank>>);
        static_assert(!NotTensorConcept<Tensor<T, Rank>>);
        static_assert(TypedTensorConcept<Tensor<T, Rank>, T>);
        static_assert(!TypedTensorConcept<Tensor<T, Rank>, BadT>);
        static_assert(RankTensorConcept<Tensor<T, Rank>>);
        static_assert(RankTensorConcept<Tensor<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<Tensor<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<Tensor<T, Rank>>);
        static_assert(LockableConcept<Tensor<T, Rank>>);
        static_assert(TRTensorConcept<Tensor<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<Tensor<T, Rank>, Rank, T>);
        static_assert(CoreTensorConcept<Tensor<T, Rank>>);
        static_assert(!DiskTensorConcept<Tensor<T, Rank>>);
        static_assert(!TensorViewConcept<Tensor<T, Rank>>);
        static_assert(BasicTensorConcept<Tensor<T, Rank>>);
        static_assert(!CollectedTensorConcept<Tensor<T, Rank>>);
        static_assert(!TiledTensorConcept<Tensor<T, Rank>>);
        static_assert(!TiledTensorConcept<Tensor<T, Rank>, Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<Tensor<T, Rank>>);
        static_assert(!BlockTensorConcept<Tensor<T, Rank>, Tensor<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<Tensor<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<Tensor<T, Rank>>);
        static_assert(requires(Tensor<T, Rank> tensor) { println(tensor); });

        static_assert(TensorConcept<TensorView<T, Rank>>);
        static_assert(!NotTensorConcept<TensorView<T, Rank>>);
        static_assert(TypedTensorConcept<TensorView<T, Rank>, T>);
        static_assert(!TypedTensorConcept<TensorView<T, Rank>, BadT>);
        static_assert(RankTensorConcept<TensorView<T, Rank>>);
        static_assert(RankTensorConcept<TensorView<T, Rank>, Rank>);
        static_assert(!RankTensorConcept<TensorView<T, Rank>, BadRank>);
        static_assert(BasicLockableConcept<TensorView<T, Rank>>);
        static_assert(LockableConcept<TensorView<T, Rank>>);
        static_assert(TRTensorConcept<TensorView<T, Rank>, Rank, T>);
        static_assert(TRLTensorConcept<TensorView<T, Rank>, Rank, T>);
        static_assert(CoreTensorConcept<TensorView<T, Rank>>);
        static_assert(!DiskTensorConcept<TensorView<T, Rank>>);
        static_assert(TensorViewConcept<TensorView<T, Rank>>);
        static_assert(BasicTensorConcept<TensorView<T, Rank>>);
        static_assert(!CollectedTensorConcept<TensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<TensorView<T, Rank>>);
        static_assert(!TiledTensorConcept<TensorView<T, Rank>, TensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<TensorView<T, Rank>>);
        static_assert(!BlockTensorConcept<TensorView<T, Rank>, TensorView<T, Rank>>);
        if constexpr (Rank > 0) {
            static_assert(FunctionTensorConcept<TensorView<T, Rank>>);
        }
        static_assert(AlgebraTensorConcept<TensorView<T, Rank>>);
        static_assert(requires(TensorView<T, Rank> tensor) { println(tensor); });

        if constexpr (Rank > 0) {
            static_assert(TensorConcept<TiledTensor<T, Rank>>);
            static_assert(!NotTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TypedTensorConcept<TiledTensor<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledTensor<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledTensor<T, Rank>>);
            static_assert(RankTensorConcept<TiledTensor<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledTensor<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledTensor<T, Rank>>);
            static_assert(LockableConcept<TiledTensor<T, Rank>>);
            static_assert(TRTensorConcept<TiledTensor<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledTensor<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!TensorViewConcept<TiledTensor<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledTensor<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensor<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensor<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensor<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledTensor<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledTensor<T, Rank>>);
            static_assert(requires(TiledTensor<T, Rank> tensor) { println(tensor); });

            static_assert(TensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!NotTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TypedTensorConcept<TiledTensorView<T, Rank>, T>);
            static_assert(!TypedTensorConcept<TiledTensorView<T, Rank>, BadT>);
            static_assert(RankTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(RankTensorConcept<TiledTensorView<T, Rank>, Rank>);
            static_assert(!RankTensorConcept<TiledTensorView<T, Rank>, BadRank>);
            static_assert(BasicLockableConcept<TiledTensorView<T, Rank>>);
            static_assert(LockableConcept<TiledTensorView<T, Rank>>);
            static_assert(TRTensorConcept<TiledTensorView<T, Rank>, Rank, T>);
            static_assert(TRLTensorConcept<TiledTensorView<T, Rank>, Rank, T>);
            static_assert(CoreTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!DiskTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TensorViewConcept<TiledTensorView<T, Rank>>);
            static_assert(!BasicTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(CollectedTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(TiledTensorConcept<TiledTensorView<T, Rank>, TensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(!BlockTensorConcept<TiledTensorView<T, Rank>, Tensor<T, Rank>>);
            static_assert(FunctionTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(AlgebraTensorConcept<TiledTensorView<T, Rank>>);
            static_assert(requires(TiledTensorView<T, Rank> tensor) { println(tensor); });
        }
    }
}

#define CHECK_REQUIREMENTS0(t1, t2, rank1)                                                                                                 \
    check_requirements<t1, t2, rank1, 0>();                                                                                                \
    check_requirements<t1, t2, rank1, 1>();                                                                                                \
    check_requirements<t1, t2, rank1, 2>();                                                                                                \
    check_requirements<t1, t2, rank1, 3>();

#define CHECK_REQUIREMENTS1(t1, t2)                                                                                                        \
    CHECK_REQUIREMENTS0(t1, t2, 0)                                                                                                         \
    CHECK_REQUIREMENTS0(t1, t2, 1)                                                                                                         \
    CHECK_REQUIREMENTS0(t1, t2, 2)

#define CHECK_REQUIREMENTS2(t1)                                                                                                            \
    CHECK_REQUIREMENTS1(t1, float)                                                                                                         \
    CHECK_REQUIREMENTS1(t1, double)                                                                                                        \
    CHECK_REQUIREMENTS1(t1, std::complex<float>)                                                                                           \
    CHECK_REQUIREMENTS1(t1, std::complex<double>)

static void check_requirements_instantiate() {
    CHECK_REQUIREMENTS2(float)
    CHECK_REQUIREMENTS2(double)
    CHECK_REQUIREMENTS2(std::complex<float>)
    CHECK_REQUIREMENTS2(std::complex<double>)
}
