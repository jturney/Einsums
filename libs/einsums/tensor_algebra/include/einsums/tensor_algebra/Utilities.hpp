//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/preprocessor/namespace.hpp>

// #include "einsums/STL.hpp"
#include "range/v3/view/iota.hpp"

#include <tuple>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace detail {

template <size_t Rank, typename... Args, std::size_t... I>
auto order_indices(std::tuple<Args...> const &combination, std::array<size_t, Rank> const &order,
                   std::index_sequence<I...>) {
    return std::tuple{get_from_tuple<size_t>(combination, order[I])...};
}

} // namespace detail
#endif

template <size_t Rank, typename... Args>
auto order_indices(std::tuple<Args...> const &combination, std::array<size_t, Rank> const &order) {
    return detail::order_indices(combination, order, std::make_index_sequence<Rank>{});
}

namespace detail {

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position),
                              _find_type_with_position<T, Position + 1, Args...>());
    } else {
        return _find_type_with_position<T, Position + 1, Args...>();
    }
}

template <typename T, int Position>
constexpr auto _unique_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _unique_find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position));
    } else {
        return _unique_find_type_with_position<T, Position + 1, Args...>();
    }
}
#endif

template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, std::size_t... I,
          typename T = double>
auto get_dim_ranges_for(TensorType<T, Rank> const &tensor, std::tuple<Args...> const &args, std::index_sequence<I...>) {
    return std::tuple{ranges::views::ints(0, (int)tensor.dim(std::get<2 * I + 1>(args)))...};
}

template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, std::size_t... I,
          typename T = double>
auto get_dim_for(TensorType<T, Rank> const &tensor, std::tuple<Args...> const &args, std::index_sequence<I...>) {
    return std::tuple{tensor.dim(std::get<2 * I + 1>(args))...};
}

template <typename T, int Position>
constexpr auto find_position() {
    return -1;
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto find_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        // Found it
        return Position;
    } else {
        return find_position<T, Position + 1, Args...>();
    }
}

template <typename AIndex, typename... Args>
constexpr auto find_position() {
    return find_position<AIndex, 0, Args...>();
}

template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(std::tuple<TargetCombination...> const &) {
    return detail::find_position<AIndex, TargetCombination...>();
}

template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(std::tuple<Ts...> const &, std::tuple<Us...> const &) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _unique_find_type_with_position(std::index_sequence<Is...>) {
    return std::tuple_cat(detail::_unique_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}

template <typename... Ts, typename... Us>
constexpr auto unique_find_type_with_position(std::tuple<Ts...> const &, std::tuple<Us...> const &) {
    return _unique_find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, typename T = double>
auto get_dim_ranges_for(TensorType<T, Rank> const &tensor, std::tuple<Args...> const &args) {
    return detail::get_dim_ranges_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

template <template <typename, size_t> typename TensorType, size_t Rank, typename... Args, typename T = double>
auto get_dim_for(TensorType<T, Rank> const &tensor, std::tuple<Args...> const &args) {
    return detail::get_dim_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

template <typename AIndex, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
auto construct_index(std::tuple<TargetCombination...> const &target_combination,
                     std::tuple<TargetPositionInC...> const &, std::tuple<LinkCombination...> const &link_combination,
                     std::tuple<LinkPositionInLink...> const &) {

    constexpr auto IsAIndexInC    = detail::find_position<AIndex, TargetPositionInC...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, LinkPositionInLink...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1,
                  "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC / 2>(target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink / 2>(link_combination);
    } else {
        return -1;
    }
}

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC,
          typename... LinkCombination, typename... LinkPositionInLink>
constexpr auto construct_indices(std::tuple<TargetCombination...> const  &target_combination,
                                 std::tuple<TargetPositionInC...> const  &target_position_in_C,
                                 std::tuple<LinkCombination...> const    &link_combination,
                                 std::tuple<LinkPositionInLink...> const &link_position_in_link) {
    return std::make_tuple(construct_index<AIndices>(target_combination, target_position_in_C, link_combination,
                                                     link_position_in_link)...);
}

template <typename AIndex, typename... UniqueTargetIndices, typename... UniqueTargetCombination,
          typename... TargetPositionInC, typename... UniqueLinkIndices, typename... UniqueLinkCombination,
          typename... LinkPositionInLink>
auto construct_index_from_unique_target_combination(
    std::tuple<UniqueTargetIndices...> const & /*unique_target_indices*/,
    std::tuple<UniqueTargetCombination...> const &unique_target_combination, std::tuple<TargetPositionInC...> const &,
    std::tuple<UniqueLinkIndices...> const & /*unique_link_indices*/,
    std::tuple<UniqueLinkCombination...> const &unique_link_combination, std::tuple<LinkPositionInLink...> const &) {

    constexpr auto IsAIndexInC    = detail::find_position<AIndex, UniqueTargetIndices...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, UniqueLinkIndices...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1,
                  "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC>(unique_target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink>(unique_link_combination);
    } else {
        return -1;
    }
}
template <typename... AIndices, typename... UniqueTargetIndices, typename... UniqueTargetCombination,
          typename... TargetPositionInC, typename... UniqueLinkIndices, typename... UniqueLinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices_from_unique_combination(std::tuple<UniqueTargetIndices...> const     &unique_target_indices,
                                          std::tuple<UniqueTargetCombination...> const &unique_target_combination,
                                          std::tuple<TargetPositionInC...> const       &target_position_in_C,
                                          std::tuple<UniqueLinkIndices...> const       &unique_link_indices,
                                          std::tuple<UniqueLinkCombination...> const   &unique_link_combination,
                                          std::tuple<LinkPositionInLink...> const      &link_position_in_link) {
    return std::make_tuple(construct_index_from_unique_target_combination<AIndices>(
        unique_target_indices, unique_target_combination, target_position_in_C, unique_link_indices,
        unique_link_combination, link_position_in_link)...);
}

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC,
          typename... LinkCombination, typename... LinkPositionInLink>
constexpr auto construct_indices(std::tuple<AIndices...> const &,
                                 std::tuple<TargetCombination...> const  &target_combination,
                                 std::tuple<TargetPositionInC...> const  &target_position_in_C,
                                 std::tuple<LinkCombination...> const    &link_combination,
                                 std::tuple<LinkPositionInLink...> const &link_position_in_link) {
    return construct_indices<AIndices...>(target_combination, target_position_in_C, link_combination,
                                          link_position_in_link);
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <typename... PositionsInX, std::size_t... I>
constexpr auto _contiguous_positions(std::tuple<PositionsInX...> const &x, std::index_sequence<I...>) -> bool {
    return ((std::get<2 * I + 1>(x) == std::get<2 * I + 3>(x) - 1) && ... && true);
}
#endif

/**
 * @brief Determines in the indices are contiguous.
 *
 * The tuple that is passed in resembles the following:
 *
 * @verbatim
 * {i, 0, j, 1, k, 2}
 * @verbatim
 *
 * or
 *
 * @verbatim
 * {i, 0, j, 2, k 1}
 * @verbatim
 *
 * In the first case, the function will return true because the indices of the labels are contiguous.
 * And in the second case, the function will return false because the indices of the labels are not contiguous.
 *
 * @tparam PositionsInX
 * @tparam I
 * @param x
 * @return true
 * @return false
 */
template <typename... PositionsInX>
constexpr auto contiguous_positions(std::tuple<PositionsInX...> const &x) -> bool {
    if constexpr (sizeof...(PositionsInX) <= 2) {
        return true;
    } else {
        return _contiguous_positions(x, std::make_index_sequence<sizeof...(PositionsInX) / 2 - 1>{});
    }
}

template <typename... PositionsInX, typename... PositionsInY, std::size_t... I>
constexpr auto _is_same_ordering(std::tuple<PositionsInX...> const &positions_in_x,
                                 std::tuple<PositionsInY...> const &positions_in_y, std::index_sequence<I...>) {
    return (std::is_same_v<decltype(std::get<2 * I>(positions_in_x)), decltype(std::get<2 * I>(positions_in_y))> &&
            ...);
}

template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(std::tuple<PositionsInX...> const &positions_in_x,
                                std::tuple<PositionsInY...> const &positions_in_y) {
    // static_assert(sizeof...(PositionsInX) == sizeof...(PositionsInY) && sizeof...(PositionsInX) > 0);
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0)
        return false; // NOLINT
    else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY))
        return false;
    else
        return _is_same_ordering(positions_in_x, positions_in_y,
                                 std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
}

template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I,
          typename T = double>
constexpr auto product_dims(std::tuple<PositionsInX...> const &indices, XType<T, XRank> const &X,
                            std::index_sequence<I...>) -> size_t {
    return (X.dim(std::get<2 * I + 1>(indices)) * ... * 1);
}

template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, std::size_t... I,
          typename T = double>
constexpr auto is_same_dims(std::tuple<PositionsInX...> const &indices, XType<T, XRank> const &X,
                            std::index_sequence<I...>) -> bool {
    return ((X.dim(std::get<1>(indices)) == X.dim(std::get<2 * I + 1>(indices))) && ... && 1);
}

template <typename LHS, typename RHS, std::size_t... I>
constexpr auto same_indices(std::index_sequence<I...>) {
    return (std::is_same_v<std::tuple_element_t<I, LHS>, std::tuple_element_t<I, RHS>> && ...);
}

template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto product_dims(std::tuple<PositionsInX...> const &indices, XType<T, XRank> const &X) -> size_t {
    return detail::product_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto is_same_dims(std::tuple<PositionsInX...> const &indices, XType<T, XRank> const &X) -> size_t {
    return detail::is_same_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <template <typename, size_t> typename XType, size_t XRank, typename... PositionsInX, typename T = double>
constexpr auto last_stride(std::tuple<PositionsInX...> const &indices, XType<T, XRank> const &X) -> size_t {
    return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
}

template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>)
        return false;
    else
        return detail::same_indices<LHS, RHS>(std::make_index_sequence<std::tuple_size_v<LHS>>());
}

} // namespace detail

END_EINSUMS_NAMESPACE_HPP(einsums::tensor_algebra)