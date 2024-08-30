//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/tensor/tensor_fwd.hpp>

namespace einsums {

/**
 * @brief Creates a diagonal matrix from a vector.
 *
 * @tparam T The datatype of the underlying data.
 * @param v The input vector.
 * @return A new rank-2 tensor with the diagonal elements set to \p v .
 */
template <typename T>
auto diagonal(const tensor<T, 1> &v) -> tensor<T, 2> {
    auto result = create_tensor<T>(v.name(), v.dim(0), v.dim(0));
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

template <typename T>
auto diagonal_like(const tensor<T, 1> &v, const tensor<T, 2> &like) -> tensor<T, 2> {
    auto result = create_tensor_like(v.name(), like);
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

} // namespace einsums