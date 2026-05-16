//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#ifndef DOXYGEN

#    include <Einsums/Concepts/TensorConcepts.hpp>
#    include <Einsums/Profile.hpp>
#    include <Einsums/Tensor/Tensor.hpp>
#    include <Einsums/TensorAlgebra/Detail/Utilities.hpp>

#    include <algorithm>
#    include <cstddef>
#    include <tuple>
#    include <utility>

namespace einsums::tensor_algebra {
template <bool ConjA, bool ConjB, TensorConcept AType, TensorConcept BType, typename... AIndices, typename... BIndices>
    requires requires {
        requires InSamePlace<AType, BType>;
        requires AType::Rank == sizeof...(AIndices);
        requires BType::Rank == sizeof...(BIndices);
    }
auto khatri_rao(std::tuple<AIndices...> const &, AType const &A, std::tuple<BIndices...> const &, BType const &B)
    -> BasicTensorLike<AType, typename AType::ValueType, 2> {
    using OutType = BasicTensorLike<AType, typename AType::ValueType, 2>;
    using T       = typename AType::ValueType;
    LabeledSection0();

    constexpr auto A_indices = std::tuple<AIndices...>();
    constexpr auto B_indices = std::tuple<BIndices...>();

    // Determine the common indices between A and B
    constexpr auto common = IntersectT<std::tuple<AIndices...>, std::tuple<BIndices...>>();
    // Determine unique indices in A
    constexpr auto A_only = DifferenceT<std::tuple<AIndices...>, decltype(common)>();
    // Determine unique indices in B
    constexpr auto B_only = DifferenceT<std::tuple<BIndices...>, decltype(common)>();

    // Record the positions of each types.
    constexpr auto A_common_position = detail::find_type_with_position(common, A_indices);
    constexpr auto B_common_position = detail::find_type_with_position(common, B_indices);
    constexpr auto A_only_position   = detail::find_type_with_position(A_only, A_indices);
    constexpr auto B_only_position   = detail::find_type_with_position(B_only, B_indices);

    // Obtain dimensions of the indices discovered above
    auto A_common_dims = detail::get_dim_for(A, A_common_position);
    auto B_common_dims = detail::get_dim_for(B, B_common_position);
    auto A_only_dims   = detail::get_dim_for(A, A_only_position);
    auto B_only_dims   = detail::get_dim_for(B, B_only_position);

    // Sanity check - ensure the common dims between A and B are the same size.
    for_sequence<std::tuple_size_v<decltype(common)>>([&](auto i) {
        if (std::get<i>(A_common_dims) != std::get<i>(B_common_dims)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Common dimensions for index {} of A and B do not match.", std::get<i>(common));
        }
    });

    auto result_dims = std::tuple_cat(std::make_tuple("KR product"), A_only_dims, B_only_dims, A_common_dims);
    // Construct resulting tensor
    auto result = std::make_from_tuple<Tensor<T, std::tuple_size_v<decltype(result_dims)> - 1>>(result_dims);
    // Perform the actual Khatri-Rao product using our einsum routine.
    einsum<ConjA, ConjB>(std::tuple_cat(A_only, B_only, common), &result, std::tuple_cat(A_only, common), A, std::tuple_cat(B_only, common),
                         B);

    // Return a reconstruction of the result tensor ... this can be considered as a simple reshape of the tensor.

    return OutType{std::move(result), "KR product", -1, detail::product_dims(A_common_position, A)};
}
} // namespace einsums::tensor_algebra

#endif