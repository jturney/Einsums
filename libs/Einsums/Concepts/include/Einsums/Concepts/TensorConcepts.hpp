//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/TensorBase.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums {

// Forward declarations
template <typename T, size_t Rank, typename Alloc>
struct GeneralTensor;

namespace disk {
template <typename T, size_t Rank>
struct Tensor;
}

/********************************
 *      Inline definitions      *
 ********************************/

/**
 * @property IsTensorV
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type defines the functions full_view_of_underlying, name, dim, and dims.
 *
 * @tparam D The type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsTensorV = requires(D tensor) {
    { tensor.full_view_of_underlying() } -> std::convertible_to<bool>;
    { tensor.name() } -> std::convertible_to<std::string>;
    tensor.dim(std::declval<int>());
    tensor.dims();
    typename std::remove_cvref_t<D>::ValueType;
};

/**
 * @property IsTypedTensorV
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename T>
constexpr inline bool IsTypedTensorV = requires {
    typename std::remove_cvref_t<D>::ValueType;
    requires std::is_same_v<typename std::remove_cvref_t<D>::ValueType, T>;
};

/**
 * @property IsRankTensorV
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have. If the rank is -1, it only checks that the tensor has a rank that is known at compile time.
 *
 * @versionadded{1.0.0}
 */
template <typename D, ptrdiff_t Rank = -1>
constexpr inline bool IsRankTensorV = requires {
    std::remove_cvref_t<D>::Rank;
    // Tensors with a dynamic-rank sentinel (RuntimeTensor / RuntimeTensorView)
    // have a ``Rank`` member but it does not represent a compile-time rank.
    // Exclude them from the static rank concept.
    requires std::remove_cvref_t<D>::Rank != dynamic_rank;
    requires(Rank == -1) || (Rank == std::remove_cvref_t<D>::Rank);
};

/**
 * @param IsScalarV
 *
 * @brief Tests to see if a value is a scalar value.
 *
 * Checks to see if a type is either a tensor with rank 0 or a scalar type such as double or complex<float>.
 *
 * @tparam D The type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsScalarV = IsRankTensorV<D, 0> || !IsTensorV<D>;

/**
 * @property IsBasicLockableV
 *
 * @brief Tests whether a given type satisfies C++'s BasicLockable requirement.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsBasicLockableV = requires(D var) {
    var.lock();
    var.unlock();
};

/**
 * @property IsLockableV
 *
 * @brief Tests whether the given type satisfies C++'s Lockable requirement.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsLockableV = requires(D var) {
    requires IsBasicLockableV<D>;
    { var.try_lock() } -> std::same_as<bool>;
};

/**
 * @property IsTRTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRTensorV = IsTypedTensorV<D, T> && IsRankTensorV<D, Rank>;

/**
 * @property IsTRLTensorV
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsTRLTensorV = IsTypedTensorV<D, T> && IsRankTensorV<D, Rank> && IsBasicLockableV<D>;

/**
 * @property IsIncoreTensorV
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against tensor_base::CoreTensor.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsIncoreTensorV = std::is_base_of_v<einsums::tensor_base::CoreTensor, D>;

namespace detail {

/// Detect if a type has a static constexpr bool IsDeviceTensor member (GeneralTensor with DeviceAllocator).
template <typename T, typename = void>
struct has_device_tensor_flag : std::false_type {};

template <typename T>
struct has_device_tensor_flag<T, std::void_t<decltype(T::IsDeviceTensor)>> : std::bool_constant<T::IsDeviceTensor> {};

} // namespace detail

/**
 * @property IsDeviceTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Detects via the ``IsDeviceTensor`` static member that GeneralTensor /
 * GeneralRuntimeTensor expose when instantiated with
 * ``gpu::DeviceAllocator<T>``.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDeviceTensorV = detail::has_device_tensor_flag<std::remove_cvref_t<D>>::value;

/**
 * @property IsDiskTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits tensor_base::DiskTensor.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDiskTensorV = std::is_base_of_v<einsums::tensor_base::DiskTensor, D>;

/**
 * @property IsTensorViewV
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks to see if the type has a typedef called <tt>underlying_type</tt>.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool IsTensorViewV = requires { typename D::underlying_type; };

/**
 * @property IsViewOfV
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename Viewed>
constexpr inline bool IsViewOfV = requires {
    requires IsTensorViewV<D>;
    requires std::is_same_v<typename std::remove_cvref_t<D>::underlying_type, Viewed>;
};

/**
 * @property IsBasicTensorV
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the type defines the functions data, stride, and strides.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsBasicTensorV =
    // A structured (tiled) tensor is NOT a basic (single-buffer) tensor, even if
    // it exposes data()/stride()/strides() for graph capture (TiledRuntimeTensor
    // returns null/advisory values). Excluding it here keeps a tiled tensor from
    // matching basic-tensor overloads that would dereference its absent whole-
    // tensor buffer, and resolves the basic-vs-tiled overload ambiguity in
    // linear_algebra. Plain TiledTensor never satisfied this (no flat data()),
    // so existing types are unaffected.
    !std::is_base_of_v<einsums::tensor_base::TiledTensorNoExtra, std::remove_cvref_t<D>> && requires(D tensor) {
        tensor.data();
        tensor.stride(0);
        tensor.strides();
    };

/**
 * @property IsCollectedTensorV
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the type defines the type <tt>StoredType</tt>.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename stored_type = void>
constexpr inline bool IsCollectedTensorV = requires {
    typename std::remove_cvref_t<D>::StoredType;
    requires std::is_void_v<stored_type> || std::is_same_v<stored_type, typename std::remove_cvref_t<D>::StoredType>;
};

/**
 * @property IsTiledTensorV
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename StoredType = void>
constexpr inline bool IsTiledTensorV = requires {
    requires std::is_base_of_v<einsums::tensor_base::TiledTensorNoExtra, std::remove_cvref_t<D>>;
    requires std::is_void_v<StoredType> || std::is_same_v<typename std::remove_cvref_t<D>::StoredType, StoredType>;
};

/**
 * @property IsBlockTensorV
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename StoredType = void>
constexpr inline bool IsBlockTensorV = requires {
    requires std::is_base_of_v<einsums::tensor_base::BlockTensorNoExtra, std::remove_cvref_t<D>>;
    requires std::is_void_v<StoredType> || std::is_same_v<typename std::remove_cvref_t<D>::StoredType, StoredType>;
};

/**
 * @property IsBufferableTensorV
 *
 * @brief Checks to see if the tensor can be buffered.
 *
 * This is used for when a tensor does not store its data in-core, but its data can brought in-core.
 * For instance, DiskTensors can have their data read into internal buffers, while FunctionTensors
 * can have their values calculated and put into a buffer on the fly.
 *
 * @versionadded{2.0.0}
 */
template <typename D>
constexpr inline bool IsBufferableTensorV = requires(D &d, D const &cd) {
    requires IsTensorV<D>;
    { d.get() };
    { cd.get() };
};

namespace detail {

template <size_t index>
int get_zero() {
    return 0;
}

template <typename T, size_t... ints>
constexpr bool test_application(std::index_sequence<ints...> const &) {
    return false;
}

template <typename T, size_t... ints>
    requires requires(T tensor, std::index_sequence<ints...> seq) {
        tensor(get_zero<ints>()...);
        tensor(std::declval<std::array<int64_t, sizeof...(ints)>>());
    }
constexpr bool test_application(std::index_sequence<ints...> const &) {
    return true;
}

template <typename T, size_t... ints>
constexpr bool test_fastsubscript(std::index_sequence<ints...> const &) {
    return false;
}

template <typename T, size_t... ints>
    requires requires(T tensor, std::index_sequence<ints...> seq) {
        tensor.subscript(get_zero<ints>()...);
        tensor.subscript(std::declval<std::array<uint64_t, sizeof...(ints)>>());
    }
constexpr bool test_fastsubscript(std::index_sequence<ints...> const &) {
    return true;
}

} // namespace detail

/**
 * @property IsFunctionTensorV
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * More specifically, checks to see if the tensor can be indexed using
 * function call syntax, and it provides a function called subscript.
 * The subscript function should not do any bounds checking or negative index checking,
 * whereas the function call operator should.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsFunctionTensorV = requires {
    requires detail::test_application<D>(std::make_index_sequence<std::remove_cvref_t<D>::Rank>());
    requires IsRankTensorV<D>;
};

/**
 * @property IsFastSubscriptableV
 *
 * @brief Checks to see if the tensor provides the subscript function.
 *
 * The subscript function should not do any bounds checking, nor should it accept negative indices as valid,
 * unless negative indices have some sort of meaning outside of index wrapping. These checks add up when iterating over
 * a tensor with millions of entries, so being able to skip them when bounds compliance is guaranteed can speed up code.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsFastSubscriptableV = requires {
    requires detail::test_fastsubscript<D>(std::make_index_sequence<std::remove_cvref_t<D>::Rank>());
    requires IsRankTensorV<D>;
};

/**
 * @property IsAlgebraTensorV
 *
 * @brief Checks to see if operations with the tensor can be optimized using libraries, indicated by deriving AlgebraOptimizedTensor.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsAlgebraTensorV = std::is_base_of_v<einsums::tensor_base::AlgebraOptimizedTensor, D>;

/**************************************
 *        Combined expressions        *
 **************************************/

/**
 * @property IsInSamePlaceV
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 *
 * @versionadded{1.0.0}
 */
template <typename... Tensors>
constexpr inline bool IsInSamePlaceV =
    (IsIncoreTensorV<Tensors> && ...) || (IsDiskTensorV<Tensors> && ...) || (IsDeviceTensorV<Tensors> && ...);

/**
 * @property IsIncoreRankTensorV
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankTensorV = IsIncoreTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsDeviceRankTensorV
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankTensorV = IsDeviceTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsDiskRankTensorV
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDiskRankTensorV = IsDiskTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankTiledTensorV
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T>;

/**
 * @property IsIncoreRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

/**
 * @property IsDeviceRankBasicTensorV
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankBasicTensorV = IsBasicTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;

/**
 * @property IsIncoreRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

/**
 * @property IsDeviceRankBlockTensorV
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankBlockTensorV = IsBlockTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;

/**
 * @property IsIncoreRankTiledTensorV
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsIncoreRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T> && IsIncoreTensorV<D>;

/**
 * @property IsDeviceRankTiledTensorV
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool IsDeviceRankTiledTensorV = IsTiledTensorV<D> && IsTRTensorV<D, Rank, T> && IsDeviceTensorV<D>;

/**
 * @property IsIncoreBasicTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsIncoreBasicTensorV = IsIncoreTensorV<D> && IsBasicTensorV<D>;

/**
 * @property IsDeviceBasicTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a basic tensor.
 *
 * Checks the tensor against DeviceTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDeviceBasicTensorV = IsDeviceTensorV<D> && IsBasicTensorV<D>;

/**
 * @property IsDiskBasicTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a basic tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDiskBasicTensorV = IsDiskTensorV<D> && IsBasicTensorV<D>;

/**
 * @property IsIncoreTiledTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsIncoreTiledTensorV = IsIncoreTensorV<D> && IsTiledTensorV<D>;

/**
 * @property IsDeviceTiledTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a tiled tensor.
 *
 * Checks the tensor against DeviceTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDeviceTiledTensorV = IsDeviceTensorV<D> && IsTiledTensorV<D>;

/**
 * @property IsDiskTiledTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a tiled tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDiskTiledTensorV = IsDiskTensorV<D> && IsTiledTensorV<D>;

/**
 * @property IsIncoreBlockTensorV
 *
 * @brief Checks to see if the tensor is available in-core and is a block tensor.
 *
 * Checks the tensor against CoreTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsIncoreBlockTensorV = IsIncoreTensorV<D> && IsBlockTensorV<D>;

/**
 * @property IsDeviceBlockTensorV
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a block tensor.
 *
 * Checks the tensor against DeviceTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      Fixed a typo. Was @c IsDeviceBLockTensorV , now it is @c IsDeviceBlockTensorV .
 * @endversion
 */
template <typename D>
constexpr inline bool IsDeviceBlockTensorV = IsDeviceTensorV<D> && IsBlockTensorV<D>;

/**
 * @property IsDiskBlockTensorV
 *
 * @brief Checks to see if the tensor is stored on-disk and is a block tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr inline bool IsDiskBlockTensorV = IsDiskTensorV<D> && IsBlockTensorV<D>;

/**
 * @property IsSameUnderlyingV
 *
 * @brief Checks to see if the tensors have the same storage type, but without specifying that type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameUnderlyingV = (std::is_same_v<typename First::ValueType, typename Rest::ValueType> && ...);

/**
 * @property IsSameRankV
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * Treats any operand whose @c Rank equals @ref einsums::dynamic_rank as a
 * compile-time wildcard (no compile-time mismatch can be inferred against
 * it). Runtime-rank checks remain the caller's responsibility.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameRankV = ((First::Rank == dynamic_rank || Rest::Rank == dynamic_rank || First::Rank == Rest::Rank) && ...);

/**
 * @property IsSameUnderlyingAndRankV
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
constexpr inline bool IsSameUnderlyingAndRankV = IsSameUnderlyingV<First, Rest...> && IsSameRankV<First, Rest...>;

/**
 * \concept TensorConcept
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept TensorConcept = IsTensorV<D>;

/**
 * \concept TensorConcept
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept NotTensorConcept = !IsTensorV<D>;

/**
 * \concept TypedTensorConcept
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename T>
concept TypedTensorConcept = IsTypedTensorV<D, T>;

/**
 * \concept RankTensorConcept
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 *
 * @versionadded{1.0.0}
 */
template <typename D, ptrdiff_t Rank = -1>
concept RankTensorConcept = IsRankTensorV<D, Rank>;

/**
 * \concept BasicLockableConcept
 *
 * @brief Tests whether the given type statisfies the C++ BasicLockable requirement.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept BasicLockableConcept = IsBasicLockableV<D>;

/**
 * \concept LockableConcept
 *
 * @brief Tests whether the given type satisfies the C++ Lockable requirement.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept LockableConcept = IsLockableV<D>;

/**
 * \concept TRTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept TRTensorConcept = IsTRTensorV<D, Rank, T>;

/**
 * \concept TRLTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept TRLTensorConcept = IsTRLTensorV<D, Rank, T>;

/**
 * \concept CoreTensorConcept
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept CoreTensorConcept = IsIncoreTensorV<D>;

/**
 * \concept DeviceTensorConcept
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept DeviceTensorConcept = IsDeviceTensorV<D>;

/**
 * \concept DiskTensorConcept
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept DiskTensorConcept = IsDiskTensorV<D>;

/**
 * \concept TensorViewConcept
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept TensorViewConcept = IsTensorViewV<D>;

/**
 * \concept ViewOfConcept
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename Viewed>
concept ViewOfConcept = IsViewOfV<D, Viewed>;

/**
 * \concept BasicTensorConcept
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept BasicTensorConcept = IsBasicTensorV<D>;

/**
 * \concept CollectedTensorConcept
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename StoredType = void>
concept CollectedTensorConcept = IsCollectedTensorV<D, StoredType>;

/**
 * \concept TiledTensorConcept
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename StoredType = void>
concept TiledTensorConcept = IsTiledTensorV<D, StoredType>;

/**
 * \concept BlockTensorConcept
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 *
 * @versionadded{1.0.0}
 */
template <typename D, typename StoredType = void>
concept BlockTensorConcept = IsBlockTensorV<D, StoredType>;

/**
 * @concept BufferableTensorConcept
 *
 * @brief Checks to see if a tensor's data can be buffered.
 *
 * This checks for tensors that don't store their data in-core, but can have their data be brought in-core.
 * For instance, DiskTensors can have their data read into an internal buffer for use, while a FunctionTensor
 * can be evaluated at several points and the results put into a buffer.
 *
 * @versionadded{2.0.0}
 */
template <typename D>
concept BufferableTensorConcept = IsBufferableTensorV<D>;

/**
 * \concept FunctionTensorConcept
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept FunctionTensorConcept = IsFunctionTensorV<D>;

/**
 * \concept FastSubscriptableConcept
 *
 * @brief Checks to see if the tensor has a faster method of subscripting.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept FastSubscriptableConcept = IsFastSubscriptableV<D>;

/**
 * \concept AlgebraTensorConcept
 *
 * @brief Checks to see if operations with the tensor can be optimized with libraries.
 *
 * @tparam D The tensor type to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept AlgebraTensorConcept = IsAlgebraTensorV<D>;

/**
 * \concept CoreRankTensor
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTensor = IsIncoreRankTensorV<D, Rank, T>;

/**
 * \concept DeviceRankTensor
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTensor = IsDeviceRankTensorV<D, Rank, T>;

/**
 * \concept DiskRankTensor
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept DiskRankTensor = IsDiskRankTensorV<D, Rank, T>;

/**
 * \concept RankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept RankBasicTensor = IsRankBasicTensorV<D, Rank, T>;

/**
 * \concept RankTiledTensor
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept RankTiledTensor = IsRankTiledTensorV<D, Rank, T>;

/**
 * \concept RankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept RankBlockTensor = IsRankBlockTensorV<D, Rank, T>;

/**
 * \concept CoreRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBasicTensor = IsIncoreRankBasicTensorV<D, Rank, T>;

/**
 * \concept DeviceRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBasicTensor = IsDeviceRankBasicTensorV<D, Rank, T>;

/**
 * \concept CoreRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBlockTensor = IsIncoreRankBlockTensorV<D, Rank, T>;

/**
 * \concept DeviceRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBlockTensor = IsDeviceRankBlockTensorV<D, Rank, T>;

/**
 * \concept CoreRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTiledTensor = IsIncoreRankTiledTensorV<D, Rank, T>;

/**
 * \concept DeviceRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 *
 * @versionadded{1.0.0}
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTiledTensor = IsDeviceRankTiledTensorV<D, Rank, T>;

/**
 * \concept CoreBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor stored in-core.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept CoreBasicTensorConcept = IsIncoreBasicTensorV<D>;

/**
 * \concept DeviceBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept DeviceBasicTensorConcept = IsDeviceBasicTensorV<D>;

/**
 * \concept CoreBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor stored in-core.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept CoreBlockTensorConcept = IsIncoreBlockTensorV<D>;

/**
 * \concept DeviceBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept DeviceBlockTensorConcept = IsDeviceBlockTensorV<D>;

/**
 * \concept CoreTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept CoreTiledTensorConcept = IsIncoreTiledTensorV<D>;

/**
 * \concept DeviceTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept DeviceTiledTensorConcept = IsDeviceTiledTensorV<D>;

/**
 * \concept InSamePlace
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 *
 * @versionadded{1.0.0}
 */
template <typename... Tensors>
concept InSamePlace = IsInSamePlaceV<Tensors...>;

/**
 * \concept MatrixConcept
 *
 * @brief Alias of RankTensorConcept<D, 2>.
 *
 * Shorthand for requiring that a tensor be a matrix.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept MatrixConcept = RankTensorConcept<D, 2>;

/**
 * \concept VectorConcept
 *
 * @brief Alias of RankTensorConcept<D, 1>.
 *
 * Shorthand for requiring that a tensor be a vector.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept VectorConcept = RankTensorConcept<D, 1>;

/**
 * \concept ScalarConcept
 *
 * @brief Alias of RankTensorConcept<D, 0>.
 *
 * Shorthand for requiring that a tensor be a scalar. That is, a tensor with zero rank or a variable with a type such as double or
 * std::complex<float>.
 *
 * @tparam D The tensor to check.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
concept ScalarConcept = IsScalarV<D>;

/**
 * \concept SameUnderlying
 *
 * @brief Checks that several tensors store the same type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
concept SameUnderlying = IsSameUnderlyingV<First, Rest...>;

/**
 * \concept SameRank
 *
 * @brief Checks that several tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
concept SameRank = IsSameRankV<First, Rest...>;

/**
 * \concept SameUnderlyingAndRank
 *
 * @brief Checks that several tensors have the same rank and underlying type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
concept SameUnderlyingAndRank = IsSameUnderlyingAndRankV<First, Rest...>;

/**
 * @struct RemoveView
 *
 * @brief Gets the underlying type of view.
 *
 * @tparam D The tensor type to strip.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
struct RemoveView {
    /**
     * @typedef base_type
     *
     * This will contain the type of tensor being viewed by the tensor view, if given a view.
     * Otherwise, it will be the same as the tensor being passed.
     *
     * @versionadded{1.0.0}
     */
    using base_type = D;
};

template <TensorViewConcept D>
struct RemoveView<D> {
    using base_type = typename D::underlying_type;
};

/**
 * @typedef RemoveViewT
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
using RemoveViewT = typename RemoveView<D>::base_type;

namespace detail {
/**
 * @brief Creates a new tensor with the same type as the input but with a different rank or storage type.
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the
 *
 * @versionadded{1.0.0}
 */
template <typename NewT, size_t NewRank, template <typename, size_t> typename TensorType, typename T, size_t Rank>
    requires(TensorConcept<TensorType<T, Rank>>)
TensorType<NewT, NewRank> create_tensor_of_same_type(TensorType<T, Rank> const &tensor) {
    return TensorType<NewT, NewRank>();
}

/**
 * @brief Creates a new basic tensor in the same place as the input, but with a different rank and storage type.
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the tensor to be copied.
 *
 * @versionadded{1.0.0}
 */
template <typename NewT, size_t NewRank, CoreTensorConcept TensorType>
GeneralTensor<NewT, NewRank, std::allocator<NewT>> create_basic_tensor_like(TensorType const &tensor) {
    return GeneralTensor<NewT, NewRank, std::allocator<NewT>>();
}

template <typename NewT, size_t NewRank, DiskTensorConcept TensorType>
disk::Tensor<NewT, NewRank> create_basic_tensor_like(TensorType const &) {
    return disk::Tensor<NewT, NewRank>();
}

} // namespace detail

/**
 * @typedef TensorLike
 *
 * @brief Gets the type of tensor, but with a new rank and type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 *
 * @versionadded{1.0.0}
 */
template <TensorConcept D, typename T, size_t Rank>
using TensorLike = decltype(detail::create_tensor_of_same_type<T, Rank>(D()));

/**
 * @typedef BasicTensorLike
 *
 * @brief Gets the type of basic tensor with the same storage location, but with different rank and underlying type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 *
 * @versionadded{1.0.0}
 */
template <TensorConcept D, typename T, size_t Rank>
using BasicTensorLike = decltype(detail::create_basic_tensor_like<T, Rank>(D()));

/**
 * @struct ValueType
 *
 * @brief Gets the data type of tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::ValueType. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 */
template <typename D>
struct ValueType {
    /**
     * @typedef type
     *
     * This will contain the type of data stored by the tensor.
     */
    using type = D;
};

template <TensorConcept D>
struct ValueType<D> {
    using type = typename D::ValueType;
};

/**
 * @typedef ValueTypeT
 *
 * @brief Gets the data type of tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::ValueType. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
using ValueTypeT = typename ValueType<D>::type;

/**
 * @property TensorRank
 *
 * @brief Gets the rank of a tensor/scalar.
 *
 * Normally, you can get the rank using an expression such as AType::Rank. However,
 * if you want to support both zero-rank tensors and scalars, then this constant can help with brevity.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
constexpr size_t TensorRank = 0;

template <TensorConcept D>
constexpr size_t TensorRank<D> = D::Rank;

/**
 * @struct BiggestType
 *
 * @brief Gets the type with the biggest storage specification.
 *
 * @versionadded{1.0.0}
 */
template <typename First, typename... Rest>
struct BiggestType {
    /**
     * @typedef type
     *
     * The result of the operation. This will be the biggest type passed to this struct.
     *
     * @versionadded{1.0.0}
     */
    using type =
        std::conditional_t<(sizeof(First) > sizeof(typename BiggestType<Rest...>::type)), First, typename BiggestType<Rest...>::type>;
};

template <typename First>
struct BiggestType<First> {
    using type = First;
};

/**
 * @typedef BiggestTypeT
 *
 * @brief Gets the type with the biggest storage specification.
 *
 * @versionadded{1.0.0}
 */
template <typename... Args>
using BiggestTypeT = typename BiggestType<Args...>::type;

/**
 * @struct LocationTensorBaseOf
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * @tparam D The tensor type to query.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
struct LocationTensorBaseOf {};

template <CoreTensorConcept D>
struct LocationTensorBaseOf<D> {
    using type = tensor_base::CoreTensor;
};

template <DiskTensorConcept D>
struct LocationTensorBaseOf<D> {
    using type = tensor_base::DiskTensor;
};

/**
 * @typedef LocationTensorBaseOfT
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * This typedef can be used as a base class for tensors.
 *
 * @tparam D The tensor type to query.
 *
 * @versionadded{1.0.0}
 */
template <typename D>
using LocationTensorBaseOfT = typename LocationTensorBaseOf<D>::type;

namespace detail {
template <typename T>
constexpr size_t count_of_type() {
    return 0;
}

template <typename T, typename First, typename... Args>
constexpr size_t count_of_type(/*Args... args*/) {
    if constexpr (std::is_same_v<T, First>) {
        return 1 + count_of_type<T, Args...>();
    } else {
        return count_of_type<T, Args...>();
    }
}
} // namespace detail

/**
 * @concept NoneOfType
 *
 * Checks each type against a key. If any type matches, returns false.
 *
 * @tparam T The type to check against.
 * @tparam Args The list of types.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename... Args>
concept NoneOfType = !(std::is_same_v<T, std::remove_cvref_t<Args>> || ... || false);

/**
 * @concept AtLeastOneOfType
 *
 * Checks each type against a key. If any type matches, returns true.
 *
 * @tparam T The type to check against.
 * @tparam Args The list of types.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename... Args>
concept AtLeastOneOfType = (std::is_same_v<T, std::remove_cvref_t<Args>> || ... || false);

/**
 * @concept NumOfType
 *
 * Checks each type against a key. If the number of matches matches the number, returns true.
 *
 * @tparam T The type to check against.
 * @tparam Num The expected number of matches.
 * @tparam Args The list of types.
 *
 * @versionadded{1.0.0}
 */
template <typename T, size_t Num, typename... Args>
concept NumOfType = detail::count_of_type<T, std::remove_cvref_t<Args>...>() == Num;

/**
 * @concept AllOfType
 *
 * Checks each type against a key. If all types match, returns true.
 *
 * @tparam T The type to check against.
 * @tparam Args The list of types.
 *
 * @versionadded{1.0.0}
 */
template <typename T, typename... Args>
concept AllOfType = (std::is_same_v<T, std::remove_cvref_t<Args>> && ... && true);

} // namespace einsums
