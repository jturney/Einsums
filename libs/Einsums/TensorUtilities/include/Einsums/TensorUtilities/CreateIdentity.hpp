//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/Detail/SetTo.hpp>
#include <Einsums/Utilities/Tuple.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

namespace detail {}

/**
 * @brief Create a new tensor with \p name and \p index with ones on the diagonal. Defaults to using double for the underlying data and
 * automatically determines the rank of the tensor from \p index .
 *
 * A \p name is required for the tensor. \p name is used when printing and performing disk operations.
 *
 * @code
 * auto a = create_identity_tensor("a", 3, 3);          // auto -> Tensor<double, 2>
 * auto b = create_identity_tensor<float>("b", 4, 5, 6); // auto -> Tensor<float, 3>
 *
 * // Ones sit where every index agrees, so the diagonal is as long as the shortest axis.
 * auto c = create_identity_tensor("c", 6, 3);          // ones at (0,0), (1,1), (2,2)
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @tparam MultiIndex The datatype of the calling parameters. In almost all cases you should just ignore this parameter.
 * @param[in] name The name of the new tensor.
 * @param[in] index The arguments needed to construct the tensor.
 * @return A new tensor filled with random data
 *
 * @versionadded{1.0.0}
 */
template <typename T = double, typename... MultiIndex>
auto create_identity_tensor(std::string const &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    static_assert(sizeof...(MultiIndex) >= 1, "Rank parameter doesn't make sense.");

    Tensor<T, sizeof...(MultiIndex)> A{name, std::forward<MultiIndex>(index)...};
    A.zero();

    // The diagonal runs to the SHORTEST axis, not to the first one. Using dim(0) worked only
    // when it happened to be the smallest: create_identity_tensor("c", 6, 3) walked to 6 and
    // threw out of range on a tensor whose second axis stops at 3. A rectangular identity is
    // perfectly well defined, and this is what makes it so for every ordering of the extents.
    size_t const extent = std::min({static_cast<size_t>(index)...});

    for (size_t dim = 0; dim < extent; dim++) {
        detail::set_to(A, T{1.0}, create_tuple<sizeof...(MultiIndex)>(dim), std::make_index_sequence<sizeof...(MultiIndex)>());
    }

    return A;
}

/**
 * @brief Create a runtime-rank identity tensor from a runtime shape vector.
 *
 * RuntimeTensor-returning overload mirroring the typed form above, and matching
 * @ref create_zero_tensor and @ref create_random_tensor, which already take a shape this way.
 * Without it, an identity was the one creator in that group that handed back a statically ranked
 * tensor, so a caller working in runtime ranks had to convert.
 *
 * Ones sit where every index agrees, so the diagonal is as long as the shortest axis and a
 * rectangular shape is as valid as a square one.
 *
 * @code
 * auto a = create_identity_tensor<double>("a", {3, 3});     // 3x3 identity
 * auto b = create_identity_tensor<double>("b", {6, 3});     // ones at (0,0), (1,1), (2,2)
 * @endcode
 *
 * @tparam T The datatype of the underlying tensor. Defaults to double.
 * @param[in] name The name of the new tensor.
 * @param[in] dims The extent of each dimension.
 * @return A new tensor with ones along its diagonal and zeros elsewhere.
 *
 * @versionadded{2.0.0}
 */
template <typename T = double>
APIARY_EXPOSE APIARY_INSTANTIATE_AS("create_identity_tensor", double) APIARY_INSTANTIATE_AS("create_identity_tensor", float)
APIARY_INSTANTIATE_AS("create_identity_tensor", std::complex<double>)
APIARY_INSTANTIATE_AS("create_identity_tensor", std::complex<float>) auto
create_identity_tensor(std::string const &name, std::vector<size_t> const &dims) -> RuntimeTensor<T> {
    RuntimeTensor<T> A(name, dims);

    if (dims.empty()) {
        return A;
    }

    // A zero extent leaves nothing to set, and min() would still report it, so the loop below
    // simply does not run.
    size_t const              extent = *std::ranges::min_element(dims);
    std::vector<std::int64_t> index(dims.size());

    for (size_t dim = 0; dim < extent; dim++) {
        std::ranges::fill(index, static_cast<std::int64_t>(dim));
        A.set_element(index, T{1.0});
    }

    return A;
}

EINSUMS_NAMESPACE_END()