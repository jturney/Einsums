//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/Detail/Index.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorBase/Common.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#if defined(EINSUMS_USE_CATCH2)
#    include <catch2/catch_all.hpp>
#endif

EINSUMS_NAMESPACE_BEGIN(tensor_algebra)
namespace detail {
template <int Remaining, typename Skip, typename Head, typename... Args>
constexpr auto _get_n_skip() {
    // They are the same, skip it.
    if constexpr (std::is_same_v<std::decay_t<Skip>, std::decay_t<Head>> && Remaining > 0) {
        return _get_n_skip<Remaining, Skip, Args...>();
    } else if constexpr (Remaining > 0) {
        // They are not the same, add it.
        return std::tuple_cat(std::make_tuple(Head()), _get_n_skip<Remaining - 1, Skip, Args...>());
    } else {
        return std::tuple{};
    }
}

template <unsigned int N, typename Skip, typename... List>
constexpr auto get_n_skip(Skip const &, std::tuple<List...> const &) {
    return _get_n_skip<N, Skip, List...>();
}

template <int Remaining, typename Head, typename... Args>
constexpr auto _get_n() {
    if constexpr (Remaining > 0) {
        // They are not the same, add it.
        return std::tuple_cat(std::make_tuple(Head()), _get_n<Remaining - 1, Args...>());
    } else {
        return std::tuple{};
    }
}
} // namespace detail

template <unsigned int N, typename... List>
constexpr auto get_n(std::tuple<List...> const &) {
    return detail::_get_n<N, List...>();
}

/**
 * Returns the mode-`mode` unfolding of `tensor` with modes startng at `0`
 *
 * @returns unfolded_tensor of shape ``(tensor.dim(mode), -1)``
 */
template <unsigned int mode, template <typename, size_t> typename CType, size_t CRank, typename T>
    requires(std::is_same_v<Tensor<T, CRank>, CType<T, CRank>>)
Tensor<T, 2> unfold(CType<T, CRank> const &source) {
    LabeledSection("mode-{} unfold", mode);

    Dim<2> target_dims;
    target_dims[0] = source.dim(mode);
    target_dims[1] = 1;
    for (int i = 0; i < CRank; i++) {
        if (i == mode)
            continue;
        target_dims[1] *= source.dim(i);
    }

    auto target         = Tensor<T, 2>{fmt::format("mode-{} unfolding of {}", mode, source.name()), target_dims[0], target_dims[1]};
    auto target_indices = std::make_tuple(std::get<mode>(index::list), index::Z);
    auto source_indices = get_n<CRank>(index::list);

    // Use similar logic found in einsums:
    auto link        = IntersectT<decltype(target_indices), decltype(source_indices)>();
    auto target_only = DifferenceT<decltype(target_indices), decltype(link)>();
    auto source_only = DifferenceT<decltype(source_indices), decltype(link)>();

    auto source_position_in_source = detail::find_type_with_position(source_only, source_indices);
    auto link_position_in_source   = detail::find_type_with_position(link, source_indices);

    auto link_dims   = detail::get_dim_for(target, detail::find_type_with_position(link, target_indices));
    auto source_dims = detail::get_dim_for(source, source_position_in_source);

    std::array<size_t, std::tuple_size_v<decltype(link_dims)>> link_strides;
    size_t                                                     link_elems = dims_to_strides(link_dims, link_strides);

    Stride<std::tuple_size_v<decltype(source_dims)>> source_strides;
    size_t                                           source_elems = dims_to_strides(source_dims, source_strides);

#pragma omp parallel for collapse(2)
    for (size_t link_item = 0; link_item < link_elems; link_item++) {
        for (size_t source_item = 0; source_item < source_elems; source_item++) {
            thread_local std::array<uint64_t, 2>                                        link_it;
            thread_local std::array<uint64_t, std::tuple_size_v<decltype(source_dims)>> source_it;
            sentinel_to_indices(link_item, link_strides, link_it);
            sentinel_to_indices(source_item, source_strides, source_it);

            auto target_order = std::array<size_t, 2>{std::get<0>(link_it), source_item};

            auto source_order =
                detail::construct_indices(source_indices, source_it, source_position_in_source, link_it, link_position_in_source);

            T source_value                 = subscript_tensor(source, source_order);
            target.subscript(target_order) = source_value;
        }
    }

    return target;
}

EINSUMS_NAMESPACE_END(tensor_algebra)