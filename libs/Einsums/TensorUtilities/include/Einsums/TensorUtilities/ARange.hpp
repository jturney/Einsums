//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Creates a new rank-1 tensor filled with digits from \p start to \p stop in \p step increments.
 *
 * @code
 * // auto -> Tensor<double, 1> with data ranging from 0.0 to 9.0
 * auto a = arange<double>(0, 10);
 * @endcode
 *
 * @tparam T Underlying datatype of the tensor
 * @param[in] start Value to start the tensor with
 * @param[in] stop Value to stop the tensor with
 * @param[in] step Increment value
 * @return new rank-1 tensor filled with digits from \p start to \p stop in \p step increments
 *
 * @versionadded{1.0.0}
 */
template <NotComplex T>
auto arange(T start, T stop, T step = T{1}) -> Tensor<T, 1> {
    if (stop < start) {
        EINSUMS_THROW_EXCEPTION(bad_logic, "arange: stop ({}) < start ({})", stop, start);
    }

    int nelem = static_cast<int>((stop - start) / step);

    auto result = create_zero_tensor<T>("arange created tensor", nelem);

    int index{0};
    for (T value = start; value < stop; value += step) {
        result(index++) = value;
    }
    return result;
}

/**
 * @brief Creates a new rank-1 tensor filled with digits from 0 to \p stop .
 *
 * @code
 * // auto -> Tensor<double, 1> with data ranging from 0.0 to 9.0
 * auto a = arange<double>(10);
 * @endcode
 *
 * @tparam T Underlying datatype of the tensor
 * @param[in] stop Value to stop the tensor with
 * @return new rank-1 tensor filled with digits from 0 to \p stop
 *
 * @versionadded{1.0.0}
 */
template <NotComplex T>
auto arange(T stop) -> Tensor<T, 1> {
    return arange(T{0}, stop);
}

EINSUMS_NAMESPACE_END()