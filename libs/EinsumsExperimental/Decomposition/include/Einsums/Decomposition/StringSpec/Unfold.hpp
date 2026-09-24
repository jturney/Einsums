//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorPermute/Unfold.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(decomposition::string_spec::detail)

/// The mode-@p mode unfolding of @p tensor as a new matrix. See tensor_permute::unfold.
template <typename TType, size_t TRank>
auto unfolded(std::size_t mode, Tensor<TType, TRank> const &tensor) -> Tensor<TType, 2> {
    std::size_t columns = 1;
    for (std::size_t axis = 0; axis < TRank; ++axis) {
        if (axis != mode) {
            columns *= tensor.dim(axis);
        }
    }
    Tensor<TType, 2> result{"unfolding", tensor.dim(mode), columns};
    tensor_permute::unfold(mode, &result, tensor);
    return result;
}

EINSUMS_NAMESPACE_END(decomposition::string_spec::detail)
