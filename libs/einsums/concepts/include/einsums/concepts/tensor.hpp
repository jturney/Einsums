//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/tensor_base/tensor_base.hpp>

#include <cstddef>
#include <type_traits>

namespace einsums {

// Forward declarations
template <typename T, size_t Rank>
struct tensor;

namespace disk {
template <typename T, size_t Rank>
struct tensor;
}

#if defined(__HIP__)
template <typename T, size_t Rank>
struct device_tensor;
#endif

namespace detail {
/**************************************
 *               Structs              *
 **************************************/

/**********************
 *    Basic traits.   *
 **********************/

/**
 * @struct is_tensor
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_base::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
struct is_tensor : public std::is_base_of<tensor_base::tensor_no_extra, D> {};

/**
 * @struct is_typed_tensor
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
struct is_typed_tensor : std::is_base_of<tensor_base::typed_tensor<T>, D> {};

/**
 * @struct is_rank_tensor
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
struct is_rank_tensor : public std::is_base_of<tensor_base::rank_tensor<Rank>, D> {};

/**
 * @struct is_scalar
 *
 * @brief Tests to see if a value is a scalar value.
 *
 * Checks to see if a type is either a tensor with rank 0 or a scalar type such as double or complex<float>.
 *
 * @tparam D The type to check.
 */
template <typename D>
struct is_scalar : std::bool_constant<is_rank_tensor<D, 0>::value || !is_tensor<D>::value> {};

/**
 * @struct is_lockable_tensor
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct is_lockable_tensor : std::is_base_of<tensor_base::lockable_tensor, D> {};

/**
 * @struct is_tr_tensor
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
struct is_tr_tensor : std::bool_constant<is_rank_tensor<D, Rank>::value && is_typed_tensor<D, T>::value> {};

/**
 * @struct is_trl_tensor
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
struct is_trl_tensor : std::bool_constant<is_rank_tensor<D, Rank>::value && is_typed_tensor<D, T>::value && is_lockable_tensor<D>::value> {
};

/**
 * @struct is_incore_tensor
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct is_incore_tensor : std::is_base_of<tensor_base::core_tensor, D> {};

#ifdef __HIP__
/**
 * @struct is_device_tensor
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct is_device_tensor : public std::is_base_of<tensor_base::device_tensor_base, D> {};
#endif

/**
 * @struct is_disk_tensor
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct is_disk_tensor : std::is_base_of<tensor_base::disk_tensor, D> {};

/**
 * @struct is_tensor_view
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct is_tensor_view : std::is_base_of<tensor_base::tensor_view_no_extra, D> {};

/**
 * @struct is_view_of
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
struct is_view_of : std::is_base_of<tensor_base::tensor_view_only_viewed<Viewed>, D> {};

/**
 * @struct is_basic_tensor
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct is_basic_tensor : std::is_base_of<tensor_base::basic_tensor_no_extra, D> {};

/**
 * @struct is_collected_tensor
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct is_collected_tensor
    : std::bool_constant<std::is_void_v<StoredType> ? std::is_base_of_v<tensor_base::collected_tensor_no_extra, D>
                                                    : std::is_base_of_v<tensor_base::collected_tensor_only_stored<StoredType>, D>> {};

/**
 * @struct is_tiled_tensor
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct is_tiled_tensor
    : std::bool_constant<std::is_base_of_v<tensor_base::tiled_tensor_no_extra, D> &&
                         (std::is_void_v<StoredType> || std::is_base_of_v<tensor_base::collected_tensor_only_stored<StoredType>, D>)> {};

/**
 * @struct is_block_tensor
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
struct is_block_tensor
    : std::bool_constant<std::is_base_of_v<tensor_base::block_tensor_no_extra, D> &&
                         (std::is_void_v<StoredType> || std::is_base_of_v<tensor_base::collected_tensor_only_stored<StoredType>, D>)> {};

/**
 * @struct is_function_tensor
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
struct is_function_tensor : public std::is_base_of<tensor_base::function_tensor_no_extra, D> {};

/**
 * @struct is_algebra_tensor
 *
 * @brief Checks to see if operations with the tensor can be optimized using libraries, indicated by deriving AlgebraOptimizedTensor.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
struct is_algebra_tensor : public std::is_base_of<tensor_base::algebra_optimized_tensor, D> {};
} // namespace detail

/********************************
 *      Inline definitions      *
 ********************************/

/**
 * @property is_tensor_v
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_base::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
constexpr inline bool is_tensor_v = detail::is_tensor<D>::value;

/**
 * @property is_typed_tensor_v
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
constexpr inline bool is_typed_tensor_v = detail::is_typed_tensor<D, T>::value;

/**
 * @property is_rank_tensor_v
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
constexpr inline bool is_rank_tensor_v = detail::is_rank_tensor<D, Rank>::value;

/**
 * @param IsScalarV
 *
 * @brief Tests to see if a value is a scalar value.
 *
 * Checks to see if a type is either a tensor with rank 0 or a scalar type such as double or complex<float>.
 *
 * @tparam D The type to check.
 */
template <typename D>
constexpr inline bool is_scalar_v = detail::is_scalar<D>::value;

/**
 * @property is_lockable_tensor_v
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_lockable_tensor_v = detail::is_lockable_tensor<D>::value;

/**
 * @property is_tr_tensor_v
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_tr_tensor_v = detail::is_tr_tensor<D, Rank, T>::value;

/**
 * @property is_trl_tensor_v
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_trl_tensor_v = detail::is_trl_tensor<D, Rank, T>::value;

/**
 * @property IsIncoreTensorV
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_incore_tensor_v = detail::is_incore_tensor<D>::value;

#ifdef __HIP__
/**
 * @property is_device_tensor_v
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_device_tensor_v = detail::is_device_tensor<D>::value;
#endif

/**
 * @property is_disk_tensor_v
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_disk_tensor_v = detail::is_disk_tensor<D>::value;

/**
 * @property is_tensor_view_v
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_tensor_view_v = detail::is_tensor_view<D>::value;

/**
 * @property is_view_of_v
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
constexpr inline bool is_view_of_v = detail::is_view_of<D, Viewed>::value;

/**
 * @property is_basic_tensor_v
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_basic_tensor_v = detail::is_basic_tensor<D>::value;

/**
 * @property is_collected_tensor_v
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool is_collected_tensor_v = detail::is_collected_tensor<D, StoredType>::value;

/**
 * @property is_tiled_tensor_v
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool is_tiled_tensor_v = detail::is_tiled_tensor<D, StoredType>::value;

/**
 * @property is_block_tensor_v
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
constexpr inline bool is_block_tensor_v = detail::is_block_tensor<D, StoredType>::value;

/**
 * @property is_function_tensor_v
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_function_tensor_v = detail::is_function_tensor<D>::value;

/**
 * @property is_algebra_tensor_v
 *
 * @brief Checks to see if operations with the tensor can be optimized using libraries, indicated by deriving AlgebraOptimizedTensor.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_algebra_tensor_v = detail::is_algebra_tensor<D>::value;

/**************************************
 *        Combined expressions        *
 **************************************/

/**
 * @property IsSamePlaceV
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
constexpr inline bool is_in_same_place_v = (is_incore_tensor_v<Tensors> && ...) || (is_disk_tensor_v<Tensors> && ...)
#if defined(__HIP__)
                                           || (is_device_tensor_v<Tensors> && ...)
#endif
    ;

/**
 * @property is_incore_rank_tensor_v
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_incore_rank_tensor_v = is_incore_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;

#ifdef __HIP__
/**
 * @property is_device_rank_tensor_v
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_device_rank_tensor_v = is_device_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;
#endif

/**
 * @property is_disk_rank_tensor_v
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_disk_rank_tensor_v = is_disk_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;

/**
 * @property is_rank_basic_tensor_v
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_rank_basic_tensor_v = is_basic_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;

/**
 * @property is_rank_tiled_tensor_v
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_rank_tiled_tensor_v = is_tiled_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;

/**
 * @property is_rank_block_tensor_v
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_rank_block_tensor_v = is_block_tensor_v<D> && is_tr_tensor_v<D, Rank, T>;

/**
 * @property is_incore_rank_basic_tensor_v
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_incore_rank_basic_tensor_v = is_basic_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_incore_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_rank_basic_tensor_v
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_device_rank_basic_tensor_v = is_basic_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_device_tensor_v<D>;
#endif

/**
 * @property is_incore_rank_block_tensor_v
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_incore_rank_block_tensor_v = is_block_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_incore_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_rank_block_tensor_v
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_device_rank_block_tensor_v = is_block_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_device_tensor_v<D>;
#endif

/**
 * @property is_incore_rank_tiled_tensor_v
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_incore_rank_tiled_tensor_v = is_tiled_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_incore_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_rank_tiled_tensor_v
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
constexpr inline bool is_device_rank_tiled_tensor_v = is_tiled_tensor_v<D> && is_tr_tensor_v<D, Rank, T> && is_device_tensor_v<D>;
#endif

/**
 * @property is_incore_basic_tensor_v
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_incore_basic_tensor_v = is_incore_tensor_v<D> && is_basic_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_basic_tensor_v
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a basic tensor.
 *
 * Checks the tensor against DeviceTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_device_basic_tensor_v = is_device_tensor_v<D> && is_basic_tensor_v<D>;
#endif

/**
 * @property is_disk_basic_tensor_v
 *
 * @brief Checks to see if the tensor is stored on-disk and is a basic tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BasicTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_disk_basic_tensor_v = is_disk_tensor_v<D> && is_basic_tensor_v<D>;

/**
 * @property is_incore_tiled_tensor_v
 *
 * @brief Checks to see if the tensor is available in-core and is a basic tensor.
 *
 * Checks the tensor against CoreTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_incore_tiled_tensor_v = is_incore_tensor_v<D> && is_tiled_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_tiled_tensor_v
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a tiled tensor.
 *
 * Checks the tensor against DeviceTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_device_tiled_tensor_v = is_device_tensor_v<D> && is_tiled_tensor_v<D>;
#endif

/**
 * @property is_disk_tiled_tensor_v
 *
 * @brief Checks to see if the tensor is stored on-disk and is a tiled tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and TiledTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_disk_tiled_tensor_v = is_disk_tensor_v<D> && is_tiled_tensor_v<D>;

/**
 * @property is_incore_block_tensor_v
 *
 * @brief Checks to see if the tensor is available in-core and is a block tensor.
 *
 * Checks the tensor against CoreTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_incore_block_tensor_v = is_incore_tensor_v<D> && is_block_tensor_v<D>;

#ifdef __HIP__
/**
 * @property is_device_block_tensor_v
 *
 * @brief Checks to see if the tensor is available to graphics hardware and is a block tensor.
 *
 * Checks the tensor against DeviceTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
constexpr inline bool is_device_block_tensor_v = is_device_tensor_v<D> && is_block_tensor_v<D>;
#endif

/**
 * @property is_disk_block_tensor_v
 *
 * @brief Checks to see if the tensor is stored on-disk and is a block tensor.
 *
 * Checks whether the tensor inherits DiskTensorBase and BlockTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
constexpr inline bool is_disk_block_tensor_v = is_disk_tensor_v<D> && is_block_tensor_v<D>;

/**
 * @property is_same_underlying_v
 *
 * @brief Checks to see if the tensors have the same storage type, but without specifying that type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
constexpr inline bool is_same_underlying_v = (std::is_same_v<typename First::value_type, typename Rest::value_type> && ...);

/**
 * @property is_same_rank_v
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 */
template <typename First, typename... Rest>
constexpr inline bool is_same_rank_v = ((First::rank == Rest::rank) && ...);

/**
 * @property is_same_underlying_and_rank_v
 *
 * @brief Checks to see if the tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors
 */
template <typename First, typename... Rest>
constexpr inline bool is_same_underlying_and_rank_v = is_same_underlying_v<First, Rest...> && is_same_rank_v<First, Rest...>;

/**
 * @concept TensorConcept
 *
 * @brief Tests whether the given type is a tensor or not.
 *
 * Checks to see if the given type is derived from einsums::tensor_props::TensorBase.
 *
 * @tparam D The type to check.
 */
template <typename D>
concept TensorConcept = is_tensor_v<D>;

template <typename D>
concept NotTensorConcept = !is_tensor_v<D>;

/**
 * @concept TypedTensorConcept
 *
 * @brief Tests whether the given type is a tensor with an underlying type.
 *
 * @tparam D The tensor type to check.
 * @tparam T The type the tensor should store.
 */
template <typename D, typename T>
concept TypedTensorConcept = is_typed_tensor_v<D, T>;

/**
 * @concept RankTensorConcept
 *
 * @brief Tests whether the given type is a tensor with the given rank.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The rank the tensor should have.
 */
template <typename D, size_t Rank>
concept RankTensorConcept = is_rank_tensor_v<D, Rank>;

/**
 * @concept LockableTensorConcept
 *
 * @brief Tests whether the given tensor type can be locked.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept LockableTensorConcept = is_lockable_tensor_v<D>;

/**
 * @concept TRTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank.
 *
 * This checks to see if the tensor derives RankTensorBase and TypedTensorBase.
 * Try not to rely on a tensor deriving TRTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam T The storage type stored by the tensor.
 * @tparam Rank The expected rank of the tensor.
 */
template <typename D, size_t Rank, typename T>
concept TRTensorConcept = is_tr_tensor_v<D, Rank, T>;

/**
 * @concept TRLTensorConcept
 *
 * @brief Tests whether the given tensor type has a storage type and rank and can be locked.
 *
 * This checks to see if the tensor derives RankTensorBase, TypedTensorBase, and LockableTensorBase.
 * Try not to rely on a tensor deriving TRLTensorBase, as this may not always be the case.
 *
 * @tparam D The tensor type to check.
 * @tparam Rank The expected rank of the tensor.
 * @tparam T The expected storage type stored by the tensor.
 */
template <typename D, size_t Rank, typename T>
concept TRLTensorConcept = is_trl_tensor_v<D, Rank, T>;

/**
 * @concept CoreTensorConcept
 *
 * @brief Checks to see if the tensor is available in-core.
 *
 * Checks the tensor against CoreTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreTensorConcept = is_incore_tensor_v<D>;

#ifdef __HIP__
/**
 * @concept DeviceTensorConcept
 *
 * @brief Checks to see if the tensor is available to graphics hardware.
 *
 * Checks the tensor against DeviceTensorBase.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceTensorConcept = is_device_tensor_v<D>;
#endif

/**
 * @concept DiskTensorConcept
 *
 * @brief Checks to see if the tensor is stored on-disk.
 *
 * Checks whether the tensor inherits DiskTensorBase.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept DiskTensorConcept = is_disk_tensor_v<D>;

/**
 * @concept TensorViewConcept
 *
 * @brief Checks to see if the tensor is a view of another.
 *
 * Checks whether the tensor inherits TensorViewBaseNoExtra.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept TensorViewConcept = is_tensor_view_v<D>;

/**
 * @concept ViewOfConcept
 *
 * @brief Checks to see if the tensor is a view of another tensor with the kind of tensor specified.
 *
 * Checks whether the tensor inherits the appropriate TensorViewBase.
 *
 * @tparam D The tensor type to check.
 * @tparam Viewed The type of tensor expected to be viewed.
 */
template <typename D, typename Viewed>
concept ViewOfConcept = is_view_of_v<D, Viewed>;

/**
 * @concept BasicTensorConcept
 *
 * @brief Checks to see if the tensor is a basic tensor.
 *
 * Checks to see if the tensor inherits BasicTensorBaseNoExtra.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept BasicTensorConcept = is_basic_tensor_v<D>;

/**
 * @concept CollectedTensorConcept
 *
 * @brief Checks to see if the tensor is a tensor collection with the given storage type.
 *
 * Checks to see if the tensor inherits CollectedTensorBaseOnlyStored if a type is given, or CollectedTensorBaseNoExtra if type is not
 * given.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept CollectedTensorConcept = is_collected_tensor_v<D, StoredType>;

/**
 * @concept TiledTensorConcept
 *
 * @brief Checks to see if the tensor is a tiled tensor with the given storage type.
 *
 * Checks to see if the tensor inherits TiledTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept TiledTensorConcept = is_tiled_tensor_v<D, StoredType>;

/**
 * @concept BlockTensorConcept
 *
 * @brief Checks to see if the tensor is a block tensor with the given storage type.
 *
 * Checks to see if the tensor inherits BlockTensorBaseNoExtra. If a type is given, also check to see if it inherits
 * the appropriate CollectedTensorBaseOnlyStored.
 *
 * @tparam D The tensor to check.
 * @tparam StoredType The type of the tensors stored in the collection, or void if you don't care.
 */
template <typename D, typename StoredType = void>
concept BlockTensorConcept = is_block_tensor_v<D, StoredType>;

/**
 * @concept FunctionTensorConcept
 *
 * @brief Checks to see if the tensor is a function tensor.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept FunctionTensorConcept = is_function_tensor_v<D>;

/**
 * @concept AlgebraTensorConcept
 *
 * @brief Checks to see if operations with the tensor can be optimized with libraries.
 *
 * @tparam D The tensor type to check.
 */
template <typename D>
concept AlgebraTensorConcept = is_algebra_tensor_v<D>;

/**
 * @concept CoreRankTensor
 *
 * @brief Requires that a tensor is in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTensor = is_incore_rank_tensor_v<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankTensor
 *
 * @brief Requires that a tensor is available to the graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTensor = is_device_rank_tensor_v<D, Rank, T>;
#endif

/**
 * @concept DiskRankTensor
 *
 * @brief Requires that a tensor is stored on disk, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DiskRankTensor = is_disk_rank_tensor_v<D, Rank, T>;

/**
 * @concept RankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankBasicTensor = is_rank_basic_tensor_v<D, Rank, T>;

/**
 * @concept RankTiledTensor
 *
 * @brief Requires that a tensor is a Tiled tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankTiledTensor = is_rank_tiled_tensor_v<D, Rank, T>;

/**
 * @concept RankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept RankBlockTensor = is_rank_block_tensor_v<D, Rank, T>;

/**
 * @concept CoreRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBasicTensor = is_incore_rank_basic_tensor_v<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankBasicTensor
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBasicTensor = is_device_rank_basic_tensor_v<D, Rank, T>;
#endif

/**
 * @concept CoreRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankBlockTensor = is_incore_rank_block_tensor_v<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankBlockTensor
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankBlockTensor = is_device_rank_block_tensor_v<D, Rank, T>;
#endif

/**
 * @concept CoreRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept CoreRankTiledTensor = is_incore_rank_tiled_tensor_v<D, Rank, T>;

#ifdef __HIP__
/**
 * @concept DeviceRankTiledTensor
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware, stores the required type, and has the required rank.
 *
 * @tparam D The tensor to check.
 * @tparam Rank The rank of the tensor.
 * @tparam T The type that should be stored.
 */
template <typename D, size_t Rank, typename T>
concept DeviceRankTiledTensor = is_device_rank_tiled_tensor_v<D, Rank, T>;
#endif

/**
 * @concept CoreBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreBasicTensorConcept = is_incore_basic_tensor_v<D>;

#ifdef __HIP__
/**
 * @concept DeviceBasicTensorConcept
 *
 * @brief Requires that a tensor is a basic tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBasicTensorConcept = is_device_basic_tensor_v<D>;
#endif

/**
 * @concept CoreBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreBlockTensorConcept = is_incore_block_tensor_v<D>;

#ifdef __HIP__
/**
 * @concept DeviceBlockTensorConcept
 *
 * @brief Requires that a tensor is a block tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceBlockTensorConcept = is_device_block_tensor_v<D>;
#endif

/**
 * @concept CoreTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor stored in-core.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept CoreTiledTensorConcept = is_incore_tiled_tensor_v<D>;

#ifdef __HIP__
/**
 * @concept DeviceTiledTensorConcept
 *
 * @brief Requires that a tensor is a tiled tensor available to graphics hardware.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept DeviceTiledTensorConcept = is_device_tiled_tensor_v<D>;
#endif

/**
 * @concept InSamePlace
 *
 * @brief Requires that all tensors are in the same storage place.
 *
 * @tparam Tensors The tensors to check.
 */
template <typename... Tensors>
concept InSamePlace = is_in_same_place_v<Tensors...>;

/**
 * @concept MatrixConcept
 *
 * @brief Alias of RankTensorConcept<D, 2>.
 *
 * Shorthand for requiring that a tensor be a matrix.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept MatrixConcept = RankTensorConcept<D, 2>;

/**
 * @concept VectorConcept
 *
 * @brief Alias of RankTensorConcept<D, 1>.
 *
 * Shorthand for requiring that a tensor be a vector.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept VectorConcept = RankTensorConcept<D, 1>;

/**
 * @concept ScalarConcept
 *
 * @brief Alias of RankTensorConcept<D, 0>.
 *
 * Shorthand for requiring that a tensor be a scalar. That is, a tensor with zero rank or a variable with a type such as double or
 * std::complex<float>.
 *
 * @tparam D The tensor to check.
 */
template <typename D>
concept ScalarConcept = is_scalar_v<D>;

/**
 * @concept SameUnderlying
 *
 * @brief Checks that several tensors store the same type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlying = is_same_underlying_v<First, Rest...>;

/**
 * @concept SameRank
 *
 * @brief Checks that several tensors have the same rank.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameRank = is_same_rank_v<First, Rest...>;

/**
 * @concept SameUnderlyingAndRank
 *
 * @brief Checks that several tensors have the same rank and underlying type.
 *
 * @tparam First The first tensor.
 * @tparam Rest The rest of the tensors.
 */
template <typename First, typename... Rest>
concept SameUnderlyingAndRank = is_same_underlying_and_rank_v<First, Rest...>;

/**
 * @struct remove_view
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
struct remove_view {
    using base_type = D;
};

template <TensorViewConcept D>
struct remove_view<D> {
    using base_type = typename D::underlying_type;
};

/**
 * @typedef remove_view_t
 *
 * @brief Gets the underlying type of a view.
 *
 * @tparam D The tensor type to strip.
 */
template <typename D>
using remove_view_t = typename remove_view<D>::base_type;

namespace detail {
/**
 * @brief Creates a new tensor with the same type as the input but with a different rank or storage type.
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the
 */
template <typename NewT, size_t NewRank, template <typename, size_t> typename TensorType, typename T, size_t Rank>
    requires(TensorConcept<TensorType<T, Rank>>)
TensorType<NewT, NewRank> create_tensor_of_same_type(const TensorType<T, Rank> &tensor) {
    return TensorType<NewT, NewRank>();
}

/**
 * @brief Creates a new basic tensor in the same place as the input, but with a different rank and storage type..
 *
 * This does not initialize the new tensor and more or less is used to get the return type with a decltype.
 *
 * @param tensor The tensor whose type is being copied.
 * @tparam TensorType The type of the
 */
template <typename NewT, size_t NewRank, CoreTensorConcept TensorType>
tensor<NewT, NewRank> create_basic_tensor_like(const TensorType &) {
    return tensor<NewT, NewRank>();
}

#ifdef __HIP__
template <typename NewT, size_t NewRank, DeviceTensorConcept TensorType>
DeviceTensor<NewT, NewRank> create_basic_tensor_like(const TensorType &tensor) {
    return DeviceTensor<NewT, NewRank>();
}
#endif

template <typename NewT, size_t NewRank, DiskTensorConcept TensorType>
disk::tensor<NewT, NewRank> create_basic_tensor_like(const TensorType &) {
    return disk::tensor<NewT, NewRank>();
}

} // namespace detail

/**
 * @typedef TensorLike
 *
 * @brief Gets the type of a tensor, but with a new rank and type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 */
template <TensorConcept D, typename T, size_t Rank>
using tensor_like = decltype(detail::create_tensor_of_same_type<T, Rank>(D()));

/**
 * @typedef BasicTensorLike
 *
 * @brief Gets the type of basic tensor with the same storage location, but with different rank and underlying type.
 *
 * @tparam D The underlying tensor type.
 * @tparam T The new type.
 * @tparam Rank The new rank.
 */
template <TensorConcept D, typename T, size_t Rank>
using basic_tensor_like = decltype(detail::create_basic_tensor_like<T, Rank>(D()));

/**
 * @struct value_type
 *
 * @brief Gets the data type of a tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::value_type. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 */
template <typename D>
struct value_type {
    using type = D;
};

template <TensorConcept D>
struct value_type<D> {
    using type = typename D::value_type;
};

/**
 * @typedef DataTypeT
 *
 * @brief Gets the data type of a tensor/scalar.
 *
 * Normally, you can get the data type using an expression such as typename AType::value_type. However, if you want
 * to support both zero-rank tensors and scalars, then this typedef can help with brevity.
 */
template <typename D>
using value_type_t = typename value_type<D>::type;

/**
 * @property TensorRank
 *
 * @brief Gets the rank of a tensor/scalar.
 *
 * Normally, you can get the rank using an expression such as AType::rank. However,
 * if you want to support both zero-rank tensors and scalars, then this constant can help with brevity.
 */
template <typename D>
constexpr size_t tensor_rank = 0;

template <TensorConcept D>
constexpr size_t tensor_rank<D> = D::rank;

/**
 * @struct biggest_type
 *
 * @brief Gets the type with the biggest storage specification.
 */
template <typename First, typename... Rest>
struct biggest_type {
    using type =
        std::conditional_t<(sizeof(First) > sizeof(typename biggest_type<Rest...>::type)), First, typename biggest_type<Rest...>::type>;
};

template <typename First>
struct biggest_type<First> {
    using type = First;
};

/**
 * @typedef BiggestTypeT
 *
 * @brief Gets the type with the biggest storage specification.
 */
template <typename... Args>
using biggest_type_t = typename biggest_type<Args...>::type;

/**
 * @struct location_tensor_base_of
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * @tparam D The tensor type to query.
 */
template <typename D>
struct location_tensor_base_of {};

template <CoreTensorConcept D>
struct location_tensor_base_of<D> {
    using type = tensor_base::core_tensor;
};

template <DiskTensorConcept D>
struct location_tensor_base_of<D> {
    using type = tensor_base::disk_tensor;
};

#ifdef __HIP__
template <DeviceTensorConcept D>
struct location_tensor_base_of<D> {
    using type = tensor_base::device_tensor_base;
};
#endif

/**
 * @typedef LocationTensorBaseOfT
 *
 * @brief Gets the location base (CoreTensorBase, DiskTensorBase, etc.) of the argument.
 *
 * This typedef can be used as a base class for tensors.
 *
 * @tparam D The tensor type to query.
 */
template <typename D>
using location_tensor_base_of_t = typename location_tensor_base_of<D>::type;

namespace detail {

template <typename T, typename... Args>
constexpr auto count_of_type(/*Args... args*/) {
    return (std::is_convertible_v<Args, T> + ... + 0);
}

} // namespace detail

template <typename T, typename... Args>
concept NoneOfType = detail::count_of_type<T, Args...>() == 0;

template <typename T, typename... Args>
concept AtLeastOneOfType = detail::count_of_type<T, Args...>() >= 1;
template <typename T, size_t Num, typename... Args>
concept NumOfType = detail::count_of_type<T, Args...>() == Num;

} // namespace einsums
