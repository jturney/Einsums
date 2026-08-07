//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Creates a diagonal matrix from a vector.
 *
 * @tparam T The datatype of the underlying data.
 * @param[in] v The input vector.
 * @return A new rank-2 tensor with the diagonal elements set to \p v .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto diagonal(Tensor<T, 1> const &v) -> Tensor<T, 2> {
    EINSUMS_LOG_TRACE("creating diagonal tensor from {}", v.name());

    auto result = create_tensor<T>(v.name(), v.dim(0), v.dim(0));
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

/**
 * @brief Creates a diagonal matrix from a vector using a template.
 *
 * @tparam T The datatype of the underlying data.
 * @param[in] v The input vector.
 * @param[in] like The template to use. Its dimensions and name will be used for the new tensor.
 * @return A new rank-2 tensor with the diagonal elements set to @p v .
 *
 * @versionadded{1.0.0}
 */
template <typename T>
auto diagonal_like(Tensor<T, 1> const &v, Tensor<T, 2> const &like) -> Tensor<T, 2> {
    auto result = create_tensor_like(v.name(), like);
    result.zero();
    for (size_t i = 0; i < v.dim(0); i++) {
        result(i, i) = v(i);
    }
    return result;
}

EINSUMS_NAMESPACE_END()