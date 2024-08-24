//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <type_traits>

namespace einsums {

// TODO: Move these forward declarations into a separate header under a tensor_base module or something.

// Forward declarations
template <typename T, size_t Rank>
struct tensor;

template <typename T, size_t Rank>
struct tensor_view;

template <typename T, size_t Rank>
struct block_tensor;

template <typename T, size_t Rank>
struct tiled_tensor;

template <typename T, size_t Rank>
struct tiled_tensor_view;

template <typename T, size_t Rank>
struct disk_tensor;

template <typename T, size_t ViewRank, size_t Rank>
struct disk_view;

#if defined(__HIP__)
template <typename T, size_t Rank>
struct device_tensor;

template <typename T, size_t Rank>
struct device_tensor_view;

template <typename T, size_t Rank>
struct block_device_tensor;

template <typename T, size_t Rank>
struct tiled_device_tensor;

template <typename T, size_t Rank>
struct tiled_device_tensor_view;
#endif

namespace detail {

template <typename D, typename T, size_t Rank>
struct is_incore_rank_tensor
    : std::bool_constant<std::is_same_v<std::decay_t<D>, tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, tensor_view<T, Rank>> ||
                         std::is_same_v<std::decay_t<D>, block_tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, tiled_tensor<T, Rank>> ||
                         std::is_same_v<std::decay_t<D>, tiled_tensor_view<T, Rank>>> {};

template <typename D, typename T, size_t Rank>
inline constexpr bool is_incore_rank_tensor_v = is_incore_rank_tensor<D, T, Rank>::value;

template <typename D, typename T, size_t Rank>
struct is_basic_tensor
    : std::bool_constant<std::is_same_v<std::decay_t<D>, tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, tensor_view<T, Rank>>
#if defined(__HIP__)
                         || std::is_same_v<std::decay_t<D>, device_tensor<T, Rank>> ||
                         std::is_same_v<std::decay_t<D>, device_tensor_view<T, Rank>>
#endif
                         > {
};

template <typename D, typename T, size_t Rank>
constexpr inline bool is_basic_tensor_v = is_basic_tensor<D, T, Rank>::value;

template <typename D, typename T, size_t Rank>
struct is_incore_rank_basic_tensor : std::bool_constant<is_basic_tensor_v<D, T, Rank> && is_incore_rank_tensor_v<D, T, Rank>> {};

template <typename D, typename T, size_t Rank>
constexpr inline bool is_incore_rank_basic_tensor_v = is_incore_rank_basic_tensor<D, T, Rank>::value;

// Block tensor tests.
template <typename D, typename T, size_t Rank>
struct is_block_tensor : std::bool_constant<std::is_same_v<std::decay_t<D>, block_tensor<T, Rank>>
#if defined(__HIP__)
                                            || std::is_same_v<std::decay_t<D>, block_device_tensor<T, Rank>>
#endif
                                            > {
};

template <typename D, typename T, size_t Rank>
inline constexpr bool is_block_tensor_v = is_block_tensor<D, T, Rank>::value;

// In-core and block.
template <typename D, typename T, size_t Rank>
struct is_incore_rank_block_tensor : std::bool_constant<is_block_tensor_v<D, T, Rank> && is_incore_rank_basic_tensor_v<D, T, Rank>> {};

template <typename D, typename T, size_t Rank>
inline constexpr bool is_incore_rank_block_tensor_v = is_incore_rank_block_tensor<D, T, Rank>::value;

// Tiled tensor tests.
template <typename D, typename T, size_t Rank>
struct is_tiled_tensor : std::bool_constant<std::is_same_v<std::decay_t<D>, tiled_tensor<T, Rank>> ||
                                            std::is_same_v<std::decay_t<D>, tiled_tensor_view<T, Rank>>
#if defined(__HIP__)
                                            || std::is_same_v<std::decay_t<D>, tiled_device_tensor<T, Rank>> ||
                                            std::is_same_v<std::decay_t<D>, tiled_device_tensor_view<T, Rank>>
#endif
                                            > {
};

template <typename D, typename T, size_t Rank>
inline constexpr bool is_tiled_tensor_v = is_tiled_tensor<D, T, Rank>::value;

template <typename D, typename T, size_t Rank>
struct is_incore_rank_tiled_tensor : std::bool_constant<is_tiled_tensor_v<D, T, Rank> && is_incore_rank_tensor_v<D, T, Rank>> {};

template <typename D, typename T, size_t Rank>
inline constexpr bool is_incore_rank_tiled_tensor_v = is_incore_rank_tiled_tensor<D, T, Rank>::value;

template <typename D, typename T, size_t Rank, size_t ViewRank = Rank>
struct is_ondisk_tensor : std::bool_constant<std::is_same_v<D, disk_tensor<T, Rank>> || std::is_same_v<D, disk_view<T, ViewRank, Rank>>> {};

template <typename D, typename T, size_t Rank, size_t ViewRank = Rank>
inline constexpr bool is_ondisk_tensor_v = is_ondisk_tensor<D, T, Rank, ViewRank>::value;

#if defined(__HIP__)
/**
 * @struct is_device_rank_tensor
 *
 * @brief Struct for specifying that a tensor is device compatible.
 */
template <typename D, typename T, size_t Rank>
struct is_device_rank_tensor
    : std::bool_constant<
          std::is_same_v<std::decay_t<D>, device_tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, device_tensor_view<T, Rank>> ||
          std::is_same_v<std::decay_t<D>, block_bevice_tensor<T, Rank>> || std::is_same_v<std::decay_t<D>, tiled_device_tensor<T, Rank>> ||
          std::is_same_v<std::decay_t<D>, tiled_device_tensor_view<T, Rank>>> {};

/**
 * @property is_device_rank_tensor_v
 *
 * @brief True if the tensor is device compatible.
 */
template <typename D, typename T, size_t Rank>
inline constexpr bool is_device_rank_tensor_v = is_device_rank_tensor<D, T, Rank>::value;

/**
 * @struct is_device_rank_block_tensor
 *
 * @brief Struct for specifying that a tensor is device compatible, and is block diagonal.
 */
template <typename D, typename T, size_t Rank>
struct is_device_rank_block_tensor : std::bool_constant<is_device_rank_tensor_v<D, T, Rank> && is_block_tensor_v<D, T, Rank>> {};

/**
 * @property is_device_rank_block_tensor_v
 *
 * @brief True if the tensor is device compatible and is block diagonal.
 */
template <typename D, typename T, size_t Rank>
inline constexpr bool is_device_rank_block_tensor_v = is_device_rank_block_tensor<D, T, Rank>::value;

/**
 * @struct is_device_rank_tiled_tensor
 *
 * @brief Struct for specifying that a tensor is device compatible, and is tiled.
 */
template <typename D, typename T, size_t Rank>
struct is_device_rank_tiled_tensor : std::bool_constant<is_device_rank_tensor_v<D, T, Rank> && is_tiled_tensor_v<D, T, Rank>> {};

/**
 * @property IsDeviceRankTiledTensorV
 *
 * @brief True if the tensor is device compatible and is tiled.
 */
template <typename D, typename T, size_t Rank>
inline constexpr bool is_device_rank_tiled_tensor_v = is_device_rank_tiled_tensor<D, T, Rank>::value;

template <typename D, typename T, size_t Rank>
struct is_device_rank_basic_tensor : std::bool_constant<is_basic_tensor_v<D, T, Rank> && is_device_rank_tensor_v<D, T, Rank>> {};

template <typename D, typename T, size_t Rank>
constexpr inline bool is_device_rank_basic_tensor_v = is_device_rank_basic_tensor<D, T, Rank>::value;
#endif

/**
 * @struct in_same_place
 *
 * Determines whether the tensors are all in the same place, either in core, on disk, or on the GPU.
 */
template <typename AType, typename BType, size_t ARank, size_t BRank, typename ADataType, typename BDataType = ADataType>
struct in_same_place
    : std::bool_constant<(is_incore_rank_tensor_v<AType, ADataType, ARank> && is_incore_rank_tensor_v<BType, BDataType, BRank>) ||
                         (is_ondisk_tensor_v<AType, ADataType, ARank, ARank> && is_ondisk_tensor_v<BType, BDataType, BRank, BRank>)
#if defined(__HIP__)
                         || (is_device_rank_tensor_v<AType, ADataType, ARank> && is_Device_rank_tensor_v<BType, BDataType, BRank>)
#endif
                         > {
};

template <typename AType, typename BType, size_t ARank, size_t BRank, typename ADataType, typename BDataType = ADataType>
constexpr inline bool in_same_place_v = in_same_place<AType, BType, ARank, BRank, ADataType, BDataType>::value;

} // namespace detail

} // namespace einsums
