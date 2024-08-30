//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/concepts/tensor.hpp>
#include <einsums/tensor/tensor_fwd.hpp>

// TODO: Would like to move these routines to the tensor_utilities module

namespace einsums {

/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like(a);                  // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &t) -> tensor<DataType, Rank> {
    return tensor<DataType, Rank>{t.name(), t.dims()};
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the tensor. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    return device_tensor<DataType, Rank>{tensor.dims(), mode};
}
#    endif

/**
 * @brief Creates a new tensor with the same rank, dimensions, and block sizes of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> block_tensor<DataType, Rank> {
    return block_tensor<DataType, Rank>{"(unnamed)", tensor.vector_dims()};
}

#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank, dimensions, and block sizes of the provided tensor.
 *
 * The tensor name will not be copied from the provided tensor. Be sure to call set_name on the new tensor.
 *
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the new tensor. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    return BlockDeviceTensor<DataType, Rank>{"(unnamed)", mode, tensor.vector_dims()};
}
#    endif
#endif

/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like("b", a);             // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &t) -> tensor<DataType, Rank> {
    auto result = tensor<DataType, Rank>{t.dims()};
    result.set_name(name);
    return result;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same rank and dimensions of the provided tensor.
 *
 * @code
 * auto a = create_ones_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_tensor_like("b", a);             // auto -> Tensor<double, 2>
 * @endcode
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBasicTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> DeviceTensor<DataType, Rank> {
    auto result = DeviceTensor<DataType, Rank>{tensor.dims(), mode};
    result.set_name(name);
    return result;
}
#    endif

/**
 * @brief Creates a new tensor with the same, rank, dimensions, and block parameters of the provided tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires CoreRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> block_tensor<DataType, Rank> {
    return block_tensor<DataType, Rank>{name, tensor.vector_dims()};
}

#    ifdef __HIP__
/**
 * @brief Creates a new tensor with the same, rank, dimensions, and block parameters of the provided tensor.
 *
 * @tparam TensorType The basic type of the provided tensor.
 * @tparam DataType The underlying datatype of the provided tensor.
 * @tparam Rank The rank of the provided tensor.
 * @param name The name of the new tensor.
 * @param tensor The provided tensor to copy the dimensions from.
 * @param mode The storage mode for the blocks. Defaults to device memory.
 * @return A new tensor with the same rank and dimensions as the provided tensor.
 */
template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
    requires DeviceRankBlockTensor<TensorType<DataType, Rank>, Rank, DataType>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor,
                        einsums::detail::HostToDeviceMode mode = einsums::detail::DEV_ONLY) -> BlockDeviceTensor<DataType, Rank> {
    auto result = BlockDeviceTensor<DataType, Rank>{"(unnamed)", tensor.vector_dims(), mode};
    result.set_name(name);
    return result;
}
#    endif
#endif

} // namespace einsums