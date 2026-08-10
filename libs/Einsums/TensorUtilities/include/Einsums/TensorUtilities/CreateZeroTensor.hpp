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

#include <concepts>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

// Why none of these call zero() on what they just constructed.
//
// A tensor that allocates its own storage is ALREADY zero when the constructor
// returns: storage is a `std::vector<T>` grown with `resize`, which
// value-initializes, and every constructor here takes that path - external and
// aliased storage arrive through `materialize_into`/`alias_to`, never through a
// dimensioned constructor.
//
// Calling zero() on top wrote every element a second time, and the second write
// is the expensive one: the vector's is a memset, while zero() goes through the
// impl's strided writer. Measured on a 97.7 MiB (Q|mn) that was 16 ms against
// 5 ms, and DLPNO's PNO overlap stage spent 19 of its 79 ms allocating.
//
// This is load-bearing on the constructor's guarantee. TensorUtilities'
// CreateZeroTensor test asserts a freshly created tensor reads back zero, so a
// change to storage that broke it would fail there rather than silently here.

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

    Tensor<T, sizeof...(MultiIndex)> A(GlobalConfigMap::get_singleton().get_bool("row-major"), name, std::forward<MultiIndex>(index)...);
    return A;
}

// Deducing the flag rather than taking a plain `bool` keeps this overload out of
// the candidate set unless the caller really passed a bool. GCC and the
// MSVC-compatible front ends still treat any integer constant expression of value
// zero as a null pointer constant, so a zero-extent call such as
// create_zero_tensor<T>("out", size_t{0}, size_t{6}) would otherwise match here
// too, with name = (char const *)0, and be ambiguous.
template <typename T = double, std::same_as<bool> RowMajor = bool, typename... MultiIndex>
auto create_zero_tensor(RowMajor row_major, std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    EINSUMS_LOG_TRACE("creating zero tensor {}, {}", name, std::forward_as_tuple(index...));

    Tensor<T, sizeof...(MultiIndex)> A(row_major, name, std::forward<MultiIndex>(index)...);
    return A;
}

/**
 * @brief Create a runtime-rank zero tensor from a runtime shape vector.
 *
 * RuntimeTensor-returning overload mirroring the typed family above.
 * Lets Python callers, and any C++ caller with a runtime shape, avoid
 * the typed-rank cross-product. Annotated for the einsums-pybind
 * codegen and exposed to Python as ``create_zero_tensor``, overloaded
 * across the four bound dtypes.
 */
template <typename T = double>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("create_zero_tensor", double) APIARY_INSTANTIATE_AS("create_zero_tensor", float)
    APIARY_INSTANTIATE_AS("create_zero_tensor", std::complex<double>)
        APIARY_INSTANTIATE_AS("create_zero_tensor", std::complex<float>) auto create_zero_tensor(std::string const         &name,
                                                                                                 std::vector<size_t> const &dims)
            -> RuntimeTensor<T> {
    EINSUMS_LOG_TRACE("creating zero runtime tensor {} (rank {})", name, dims.size());
    RuntimeTensor<T> A(name, dims);
    return A;
}

EINSUMS_NAMESPACE_END()
