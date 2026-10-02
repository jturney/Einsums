//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/Options.hpp>

#include <concepts>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

// None of these call zero(): a self-allocating tensor's storage is value-initialized by resize, and
// a second strided write cost 16 ms against 5 ms on 97.7 MiB. The CreateZeroTensor test checks the
// constructor's guarantee.

/**
 * @brief Create a tensor and zero its  data.
 *
 * @tparam T The type to be stored by the tensor.
 * @tparam MultiIndex The types fo the indices.
 * @param[in] name The name of the new tensor.
 * @param[in] index The dimensions for the new tensor.
 * @return A new tensor whose elements have been zeroed.
 *
 * @versionadded{1.0.0}
 */
template <typename T = double, typename... MultiIndex>
auto create_zero_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating zero tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(default_row_major(), name, std::forward<MultiIndex>(index)...);
    return A;
}

// The flag is deduced, not `bool`, so a zero extent cannot match as a null-pointer name
// (create_zero_tensor<T>("out", size_t{0}, size_t{6}) would be ambiguous).
template <typename T = double, std::same_as<bool> RowMajor = bool, typename... MultiIndex>
auto create_zero_tensor(RowMajor row_major, std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating zero tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(row_major, name, std::forward<MultiIndex>(index)...);
    return A;
}

/**
 * @brief Create a runtime-rank zero tensor from a runtime shape vector.
 *
 * The runtime-rank form, exposed to Python as ``create_zero_tensor``.
 */
template <typename T = double>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("create_zero_tensor", double) APIARY_INSTANTIATE_AS("create_zero_tensor", float)
APIARY_INSTANTIATE_AS("create_zero_tensor", std::complex<double>) APIARY_INSTANTIATE_AS("create_zero_tensor", std::complex<float>) auto
create_zero_tensor(std::string const &name, std::vector<size_t> const &dims) -> RuntimeTensor<T> {
    EINSUMS_LOG_TRACE("creating zero runtime tensor {} (rank {})", name, dims.size());
    RuntimeTensor<T> A(name, dims);
    return A;
}

EINSUMS_NAMESPACE_END()
