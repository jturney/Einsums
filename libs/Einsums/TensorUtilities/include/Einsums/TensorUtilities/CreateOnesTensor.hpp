//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include <complex>
#include <string>

namespace einsums {

/**
 * @brief Create a tensor and zero its  data.
 *
 * @tparam T The type to be stored by the tensor.
 * @tparam MultiIndex The types fo the indices.
 * @param name The name of the new tensor.
 * @param index The dimensions for the new tensor.
 * @return A new tensor whose elements have been zeroed.
 */
template <typename T = double, typename... MultiIndex>
auto create_ones_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating ones tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);
    A.set_all(T{1.0});

    return A;
}
} // namespace einsums
