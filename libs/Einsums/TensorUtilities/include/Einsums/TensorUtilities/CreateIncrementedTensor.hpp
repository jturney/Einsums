//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <complex>
#include <string>

namespace einsums {

/**
 * @brief Create a new tensor with \p name and \p index filled with incremental data.
 *
 * Just a simple factory function for creating new tensor with initial data. The first element of the tensor is 0 and then each subsequent
 * entry is +1.0 of the prior element. Defaults to using double for the underlying data and automatically determines the rank of the tensor
 * from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2> with data ranging from 0.0 to 8.0
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3> with dat ranging from 0.0f to 119.0f
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with incremented data
 *
 * @versionadded{1.0.0}
 */
template <typename T = double, typename... MultiIndex>
auto create_incremented_tensor(bool row_major, std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    Tensor<T, sizeof...(MultiIndex)> A(row_major, name, std::forward<MultiIndex>(index)...);

    Stride<sizeof...(MultiIndex)> index_strides;
    size_t                        elements = dims_to_strides(A.dims(), index_strides, true);

    for (size_t item = 0; item < elements; item++) {
        size_t sentinel;
        sentinel_to_sentinels(item, index_strides, A.strides(), sentinel);
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            A.data()[sentinel] = T(item, item);
        } else {
            A.data()[sentinel] = T(item);
        }
    }

    return A;
}

/**
 * @brief Create a new tensor with \p name and \p index filled with incremental data.
 *
 * Just a simple factory function for creating new tensor with initial data. The first element of the tensor is 0 and then each subsequent
 * entry is +1.0 of the prior element. Defaults to using double for the underlying data and automatically determines the rank of the tensor
 * from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_incremented_tensor("a", 3, 3);          // auto -> Tensor<double, 2> with data ranging from 0.0 to 8.0
 * auto b = create_incremented_tensor<float>("b" 4, 5, 6); // auto -> Tensor<float, 3> with dat ranging from 0.0f to 119.0f
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with incremented data
 *
 * @versionadded{1.0.0}
 */
template <typename T = double, typename... MultiIndex>
auto create_incremented_tensor(std::string const &name, MultiIndex &&...index) -> Tensor<T, sizeof...(MultiIndex)> {
    return create_incremented_tensor<T>(GlobalConfigMap::get_singleton().get_bool("row-major"), name, std::forward<MultiIndex>(index)...);
}

} // namespace einsums