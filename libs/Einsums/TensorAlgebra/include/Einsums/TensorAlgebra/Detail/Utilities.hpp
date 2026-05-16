//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <tuple>

namespace einsums::tensor_algebra {

namespace detail {
/**
 * @enum AlgorithmChoice
 *
 * @brief Indicates which algorithm the einsum call used.
 *
 * This is an enum for testing purposes only.
 */
enum AlgorithmChoice {
    GENERIC,      /// Generic algorithm
    DOT,          /// Dot product
    DIRECT,       /// Direct product
    GER,          /// Outer product
    GEMV,         /// Matrix-vector product
    GEMM,         /// Matrix-matrix product
    PACKED_GEMM,  /// Used packed GEMM backend
    SORT_GEMM,    /// Auto sort (permute) + GEMM
    INDETERMINATE /// Something happened and the einsum call failed.
};
} // namespace detail

#if !defined(DOXYGEN)
namespace detail {

template <size_t Rank, typename... Args, std::size_t... __I>
auto order_indices(std::tuple<Args...> const &combination, std::array<size_t, Rank> const &order, std::index_sequence<__I...> /*seq*/) {
    return std::tuple{arguments::get_from_tuple<size_t>(combination, order[__I])...};
}

} // namespace detail
#endif

/**
 * Swap around the indices to the desired order. For instance, if we have a list of indices
 * <tt>{a,b,c}</tt> and we want to reverse it, we would give the order <tt>{2, 1, 0}</tt>.
 * There is no requirement that all the orders be different, so if we were to pass <tt>{1, 1, 0}</tt>
 * instead, we would get <tt>{b,b,a}</tt>.
 *
 * @param combination The indices to reorder.
 * @param order The array containing the new order.
 */
template <size_t Rank, typename... Args>
auto order_indices(std::tuple<Args...> const &combination, std::array<size_t, Rank> const &order) {
    return detail::order_indices(combination, order, std::make_index_sequence<Rank>{});
}

namespace detail {

#if !defined(DOXYGEN)
template <typename T, int Position>
constexpr auto _find_type_with_position() {
    return std::make_tuple();
}

template <typename T, int Position, typename Head, typename... Args>
constexpr auto _find_type_with_position() {
    if constexpr (std::is_same_v<std::decay_t<Head>, std::decay_t<T>>) {
        return std::tuple_cat(std::make_pair(std::decay_t<T>(), Position), _find_type_with_position<T, Position + 1, Args...>());
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

template <TensorConcept TensorType, typename... Args, size_t... __I>
auto get_dim_for(TensorType const &tensor, std::tuple<Args...> const &args, std::index_sequence<__I...> /*seq*/) {
    return std::tuple{tensor.dim(std::get<2 * __I + 1>(args))...};
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

#endif

/**
 * Find the position of an index within a tuple of indices.
 *
 * @tparam AIndex The index to find.
 * @tparam TargetCombination The indices to search through.
 *
 * @return The position of the index in the tuple, or -1 if not found.
 */
template <typename AIndex, typename... TargetCombination>
constexpr auto find_position(std::tuple<TargetCombination...> const & /*indices*/) {
    return detail::find_position<AIndex, TargetCombination...>();
}

#ifndef DOXYGEN
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _find_type_with_position(std::index_sequence<Is...> /*seq*/) {
    return std::tuple_cat(detail::_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}
#endif

/**
 * Find the positions of several types in a tuple. The type will be in the even elements of the output
 * and the positions will be in the odd elements.
 *
 * @tparam Ts The types to find.
 * @tparam Us The indices to search through.
 *
 * @return A tuple containing the types found and their positions.
 */
template <typename... Ts, typename... Us>
constexpr auto find_type_with_position(std::tuple<Ts...> const & /*unused*/, std::tuple<Us...> const & /*unused*/) {
    return _find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

#ifndef DOXYGEN
template <typename S1, typename... S2, std::size_t... Is>
constexpr auto _unique_find_type_with_position(std::index_sequence<Is...> /*seq*/) {
    return std::tuple_cat(detail::_unique_find_type_with_position<std::tuple_element_t<Is, S1>, 0, S2...>()...);
}
#endif

template <typename... Ts, typename... Us>
constexpr auto unique_find_type_with_position(std::tuple<Ts...> const & /*unused*/, std::tuple<Us...> const & /*unused*/) {
    return _unique_find_type_with_position<std::tuple<Ts...>, Us...>(std::make_index_sequence<sizeof...(Ts)>{});
}

/**
 * Create a tuple of ranges that move along each of the tensor's axes.
 *
 * @param tensor The tensor we want to iterate over.
 * @param args A tuple containing the axis indices in the odd positions and the index objects in the even positions.
 *
 * @return A tuple of ranges to be used to iterate over a tensor.
 */
// template <TensorConcept TensorType, typename... Args>
// auto get_dim_ranges_for(TensorType const &tensor, std::tuple<Args...> const &args) {
//     return detail::get_dim_ranges_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
// }

/**
 * Create a tuple containing the dimensions of a tensor.
 *
 * @param tensor The tensor to query.
 * @param args A tuple containing the indices and positions to use to query the axes of the tensor.
 */
template <TensorConcept TensorType, typename... Args>
auto get_dim_for(TensorType const &tensor, std::tuple<Args...> const &args) {
    return detail::get_dim_for(tensor, args, std::make_index_sequence<sizeof...(Args) / 2>{});
}

#ifndef DOXYGEN
// template <typename ScalarType>
//     requires(!TensorConcept<ScalarType>)
// auto get_dim_ranges_for(ScalarType const &tensor, std::tuple<> const &args) {
//     return std::tuple{};
// }

template <typename ScalarType>
auto get_dim_for(ScalarType const &tensor, std::tuple<> const &args) {
    return std::tuple{};
}
#endif

template <typename AIndex, typename TargetCombination, typename LinkCombination, typename... TargetPositionInC,
          typename... LinkPositionInLink>
ptrdiff_t construct_index(TargetCombination const &target_combination, std::tuple<TargetPositionInC...> const & /*unused*/,
                          LinkCombination const   &link_combination, std::tuple<LinkPositionInLink...> const   &/*unused*/) {

    constexpr auto IsAIndexInC    = detail::find_position<AIndex, TargetPositionInC...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, LinkPositionInLink...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1, "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC / 2>(target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink / 2>(link_combination);
    } else {
        return -1;
    }
}

template <typename... AIndices, typename... TargetCombination, typename... TargetPositionInC, typename... LinkCombination,
          typename... LinkPositionInLink>
constexpr auto
construct_indices(std::tuple<TargetCombination...> const &target_combination, std::tuple<TargetPositionInC...> const &target_position_in_C,
                  std::tuple<LinkCombination...> const &link_combination, std::tuple<LinkPositionInLink...> const &link_position_in_link) {
    return std::array<ptrdiff_t, sizeof...(AIndices)>{
        construct_index<AIndices>(target_combination, target_position_in_C, link_combination, link_position_in_link)...};
}

template <typename AIndex, typename UniqueTargetCombination, typename UniqueLinkCombination, typename... UniqueTargetIndices,
          typename... TargetPositionInC, typename... UniqueLinkIndices, typename... LinkPositionInLink>
ptrdiff_t construct_index_from_unique_target_combination(std::tuple<UniqueTargetIndices...> const & /*unique_target_indices*/,
                                                         UniqueTargetCombination const &unique_target_combination,
                                                         std::tuple<TargetPositionInC...> const & /*unused*/,
                                                         std::tuple<UniqueLinkIndices...> const & /*unique_link_indices*/,
                                                         UniqueLinkCombination const &unique_link_combination,
                                                         std::tuple<LinkPositionInLink...> const & /*unused*/) {

    constexpr auto IsAIndexInC    = detail::find_position<AIndex, UniqueTargetIndices...>();
    constexpr auto IsAIndexInLink = detail::find_position<AIndex, UniqueLinkIndices...>();

    static_assert(IsAIndexInC != -1 || IsAIndexInLink != -1, "Looks like the indices in your einsum are not quite right! :(");

    if constexpr (IsAIndexInC != -1) {
        return std::get<IsAIndexInC>(unique_target_combination);
    } else if constexpr (IsAIndexInLink != -1) {
        return std::get<IsAIndexInLink>(unique_link_combination);
    } else {
        return -1;
    }
}
template <typename... AIndices, typename... UniqueTargetIndices, typename... UniqueTargetCombination, typename... TargetPositionInC,
          typename... UniqueLinkIndices, typename... UniqueLinkCombination, typename... LinkPositionInLink>
constexpr auto construct_indices_from_unique_combination(std::tuple<UniqueTargetIndices...> const     &unique_target_indices,
                                                         std::tuple<UniqueTargetCombination...> const &unique_target_combination,
                                                         std::tuple<TargetPositionInC...> const       &target_position_in_C,
                                                         std::tuple<UniqueLinkIndices...> const       &unique_link_indices,
                                                         std::tuple<UniqueLinkCombination...> const   &unique_link_combination,
                                                         std::tuple<LinkPositionInLink...> const      &link_position_in_link) {
    return std::array<ptrdiff_t, sizeof...(AIndices)>{
        construct_index_from_unique_target_combination<AIndices>(unique_target_indices, unique_target_combination, target_position_in_C,
                                                                 unique_link_indices, unique_link_combination, link_position_in_link)...};
}

template <typename UniqueTargetCombination, typename UniqueLinkCombination, typename... AIndices, typename... UniqueTargetIndices,
          typename... TargetPositionInC, typename... UniqueLinkIndices, typename... LinkPositionInLink>
constexpr auto construct_indices_from_unique_combination(std::tuple<AIndices...> const            &A_indices,
                                                         std::tuple<UniqueTargetIndices...> const &unique_target_indices,
                                                         UniqueTargetCombination const            &unique_target_combination,
                                                         std::tuple<TargetPositionInC...> const   &target_position_in_C,
                                                         std::tuple<UniqueLinkIndices...> const   &unique_link_indices,
                                                         UniqueLinkCombination const              &unique_link_combination,
                                                         std::tuple<LinkPositionInLink...> const  &link_position_in_link) {
    return std::array<ptrdiff_t, sizeof...(AIndices)>{
        construct_index_from_unique_target_combination<AIndices>(unique_target_indices, unique_target_combination, target_position_in_C,
                                                                 unique_link_indices, unique_link_combination, link_position_in_link)...};
}

template <typename TargetCombination, typename LinkCombination, typename... AIndices, typename... TargetPositionInC,
          typename... LinkPositionInLink>
constexpr auto construct_indices(std::tuple<AIndices...> const & /*unused*/, TargetCombination const &target_combination,
                                 std::tuple<TargetPositionInC...> const &target_position_in_C, LinkCombination const &link_combination,
                                 std::tuple<LinkPositionInLink...> const &link_position_in_link) {
    return std::array<ptrdiff_t, sizeof...(AIndices)>{
        construct_index<AIndices>(target_combination, target_position_in_C, link_combination, link_position_in_link)...};
}

#if !defined(DOXYGEN)
template <typename... PositionsInX, std::size_t... __I>
constexpr auto _contiguous_positions(std::tuple<PositionsInX...> const &x, std::index_sequence<__I...> /*unused*/) -> bool {
    return ((std::get<2 * __I + 1>(x) == std::get<2 * __I + 3>(x) - 1) && ... && true);
}
#endif

/**
 * @brief Determines in the indices are contiguous.
 *
 * The tuple that is passed in resembles the following:
 *
 * @verbatim
 * {i, 0, j, 1, k, 2}
 * @endverbatim
 *
 * or
 *
 * @verbatim
 * {i, 0, j, 2, k 1}
 * @endverbatim
 *
 * In the first case, the function will return true because the indices of the labels are contiguous.
 * And in the second case, the function will return false because the indices of the labels are not contiguous.
 *
 * @tparam PositionsInX
 * @tparam __I
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

template <typename... PositionsInX, typename... PositionsInY, std::size_t... __I>
constexpr auto _is_same_ordering(std::tuple<PositionsInX...> const &positions_in_x, std::tuple<PositionsInY...> const &positions_in_y,
                                 std::index_sequence<__I...> /*unused*/) {
    return (std::is_same_v<decltype(std::get<2 * __I>(positions_in_x)), decltype(std::get<2 * __I>(positions_in_y))> && ...);
}

template <typename... PositionsInX, typename... PositionsInY>
constexpr auto is_same_ordering(std::tuple<PositionsInX...> const &positions_in_x, std::tuple<PositionsInY...> const &positions_in_y) {
    // static_assert(sizeof...(PositionsInX) == sizeof...(PositionsInY) && sizeof...(PositionsInX) > 0);
    if constexpr (sizeof...(PositionsInX) == 0 || sizeof...(PositionsInY) == 0) {
        return false; // NOLINT
    } else if constexpr (sizeof...(PositionsInX) != sizeof...(PositionsInY)) {
        return false;
    } else {
        return _is_same_ordering(positions_in_x, positions_in_y, std::make_index_sequence<sizeof...(PositionsInX) / 2>{});
    }
}

template <TensorConcept XType, typename... PositionsInX, size_t... __I>
constexpr auto product_dims(std::tuple<PositionsInX...> const &indices, XType const &X, std::index_sequence<__I...> /*unused*/) -> size_t {
    return (X.dim(std::get<2 * __I + 1>(indices)) * ... * 1);
}

template <TensorConcept XType, typename... PositionsInX, size_t... __I>
constexpr auto is_same_dims(std::tuple<PositionsInX...> const &indices, XType const &X, std::index_sequence<__I...> /*unused*/) -> bool {
    return ((X.dim(std::get<1>(indices)) == X.dim(std::get<2 * __I + 1>(indices))) && ... && 1);
}

template <typename LHS, typename RHS, std::size_t... __I>
constexpr auto same_indices(std::index_sequence<__I...> /*unused*/) {
    return (std::is_same_v<std::tuple_element_t<__I, LHS>, std::tuple_element_t<__I, RHS>> && ...);
}

template <TensorConcept XType, typename... PositionsInX>
constexpr auto product_dims(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    return detail::product_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <typename XType, typename... PositionsInX>
    requires(!TensorConcept<XType>)
constexpr auto product_dims(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    return 0UL;
}

template <TensorConcept XType, typename... PositionsInX>
constexpr auto is_same_dims(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    return detail::is_same_dims(indices, X, std::make_index_sequence<sizeof...(PositionsInX) / 2>());
}

template <typename XType, typename... PositionsInX>
    requires(!TensorConcept<XType>)
constexpr auto is_same_dims(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    return true;
}

template <CoreTensorConcept XType, typename... PositionsInX>
constexpr auto last_stride(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    if (X.impl().is_row_major()) {
        return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
    } else {
        return X.stride(std::get<1>(indices));
    }
}

template <typename XType, typename... PositionsInX>
constexpr auto last_stride(std::tuple<PositionsInX...> const &indices, einsums::detail::TensorImpl<XType> const &X) -> size_t {
    if (X.is_row_major()) {
        return X.stride(std::get<sizeof...(PositionsInX) - 1>(indices));
    } else {
        return X.stride(std::get<1>(indices));
    }
}

template <typename XType, typename... PositionsInX>
    requires(!TensorConcept<XType>)
constexpr auto last_stride(std::tuple<PositionsInX...> const &indices, XType const &X) -> size_t {
    return 0UL;
}

template <typename LHS, typename RHS>
constexpr auto same_indices() {
    if constexpr (std::tuple_size_v<LHS> != std::tuple_size_v<RHS>) {
        return false;
    } else {
        return detail::same_indices<LHS, RHS>(std::make_index_sequence<std::tuple_size_v<LHS>>());
    }
}

template <typename UniqueIndex, int BDim, TensorConcept BType>
size_t get_grid_ranges_for_many_b(BType const &B, std::tuple<> const &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead>
auto get_grid_ranges_for_many_b(BType const &B, std::tuple<BHead> const &B_indices)
    -> std::enable_if<std::is_same_v<BHead, UniqueIndex>, size_t> {
    if constexpr (IsTiledTensorV<BType>) {
        return B.grid_size(BDim);
    } else if constexpr (IsBlockTensorV<BType>) {
        return B.num_blocks();
    } else {
        return 1;
    }
}

template <typename UniqueIndex, int BDim, TensorConcept BType, typename BHead, typename... BIndices>
size_t get_grid_ranges_for_many_b(BType const &B, std::tuple<BHead, BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<BHead, UniqueIndex>) {
        if constexpr (IsTiledTensorV<BType>) {
            return B.grid_size(BDim);
        } else if constexpr (IsBlockTensorV<BType>) {
            return B.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, BDim + 1>(B, std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType, TensorConcept BType, typename... BIndices>
size_t get_grid_ranges_for_many_a(AType const &A, std::tuple<> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices) {
    return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, TensorConcept AType, TensorConcept BType, typename AHead, typename... BIndices>
size_t get_grid_ranges_for_many_a(AType const &A, std::tuple<AHead> const &A_indices, BType const &B,
                                  std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, TensorConcept AType, TensorConcept BType, typename AHead, typename... AIndices,
          typename... BIndices>
auto get_grid_ranges_for_many_a(AType const &A, std::tuple<AHead, AIndices...> const &A_indices, BType const &B,
                                std::tuple<BIndices...> const &B_indices) -> std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        if constexpr (IsTiledTensorV<AType>) {
            return A.grid_size(ADim);
        } else if constexpr (IsBlockTensorV<AType>) {
            return A.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, ADim + 1>(A, std::tuple<AIndices...>(), B, B_indices);
    }
}

// In these functions, leave CType as typename to allow for scalar types and tensor types.
template <typename UniqueIndex, int CDim, typename CType, TensorConcept AType, TensorConcept BType, typename... AIndices,
          typename... BIndices>
size_t get_grid_ranges_for_many_c(CType const &C, std::tuple<> const &C_indices, AType const &A, std::tuple<AIndices...> const &A_indices,
                                  BType const &B, std::tuple<BIndices...> const &B_indices) {
    return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, typename CType, TensorConcept AType, TensorConcept BType, typename CHead, typename... AIndices,
          typename... BIndices>
size_t get_grid_ranges_for_many_c(CType const &C, std::tuple<CHead> const &C_indices, AType const &A,
                                  std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (IsBlockTensorV<CType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, TensorConcept AType, TensorConcept BType, typename CHead, typename... CIndices,
          typename... AIndices, typename... BIndices>
auto get_grid_ranges_for_many_c(CType const &C, std::tuple<CHead, CIndices...> const &C_indices, AType const &A,
                                std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices)
    -> std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        if constexpr (IsTiledTensorV<CType>) {
            return C.grid_size(CDim);
        } else if constexpr (IsBlockTensorV<CType>) {
            return C.num_blocks();
        } else {
            return 1;
        }
    } else {
        return get_grid_ranges_for_many_c<UniqueIndex, CDim + 1>(C, std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

/**
 * @brief Finds the tile grid dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param B The B tensor.
 * @param B_indices The indices for the B tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, TensorConcept AType, TensorConcept BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
auto get_grid_ranges_for_many(CType const &C, std::tuple<CIndices...> const &C_indices, AType const &A,
                              std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices,
                              std::tuple<AllUniqueIndices...> const &All_unique_indices) {
    return std::array{get_grid_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

} // namespace detail

namespace detail {

template <typename CheckType1, typename CheckType2, TensorConcept TensorType>
auto get_stride_for_index(TensorType const &tensor, size_t index) {
    if constexpr (std::is_same_v<std::remove_cvref_t<CheckType1>, std::remove_cvref_t<CheckType2>>) {
        return tensor.stride(index);
    } else {
        return 0;
    }
}

template <size_t N, TensorConcept TensorType, typename... TensIndexWithPos, typename... OutIndex, size_t... __I>
auto get_stride_for(TensorType const &tensor, std::tuple<TensIndexWithPos...> const &args, std::tuple<OutIndex...> const &out_indices,
                    std::array<size_t, sizeof...(OutIndex)> &out, std::index_sequence<__I...> const &seq) {
    if constexpr (N >= sizeof...(OutIndex)) {
        return;
    } else {
        out[N] =
            (get_stride_for_index<std::tuple_element_t<N, std::remove_cvref_t<decltype(out_indices)>>,
                                  std::tuple_element_t<2 * __I, std::remove_cvref_t<decltype(args)>>>(tensor, std::get<2 * __I + 1>(args)) +
             ... + 0);
        get_stride_for<N + 1>(tensor, args, out_indices, out, seq);
    }
}
} // namespace detail

template <TensorConcept TensorType, typename... TensIndexWithPos, typename... OutIndex>
auto get_stride_for(TensorType const &tensor, std::tuple<TensIndexWithPos...> const &args, std::tuple<OutIndex...> const &out_indices) {
    std::array<size_t, sizeof...(OutIndex)> out;
    out.fill(0);

    detail::get_stride_for<0>(tensor, args, out_indices, out, std::make_index_sequence<sizeof...(TensIndexWithPos) / 2>{});

    return out;
}

namespace detail {
template <typename T, typename Tuple>
struct HasType;

template <typename T>
struct HasType<T, std::tuple<>> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct HasType<T, std::tuple<U, Ts...>> : HasType<T, std::tuple<Ts...>> {};

template <typename T, typename... Ts>
struct HasType<T, std::tuple<T, Ts...>> : std::true_type {};
} // namespace detail

/**
 * @struct Intersect
 *
 * Find the intersection between two tuples.
 */
template <typename S1, typename S2>
struct Intersect {
    /**
     * Make the compiler perform the difference operation. The return type of this function
     * is the desired result.
     */
    template <std::size_t... Indices>
    static auto make_intersection(std::index_sequence<Indices...>) {

        return std::tuple_cat(std::conditional_t<detail::HasType<std::decay_t<std::tuple_element_t<Indices, S1>>, std::decay_t<S2>>::value,
                                                 std::tuple<std::tuple_element_t<Indices, S1>>, std::tuple<>>{}...);
    }

    /**
     * @typedef type
     *
     * Holds the result of this operation.
     */
    using type = decltype(make_intersection(std::make_index_sequence<std::tuple_size<S1>::value>{}));
};

/**
 * @typedef IntersectT
 *
 * Gives the result of an intersection operation. @sa Intersect
 */
template <typename S1, typename S2>
using IntersectT = typename Intersect<S1, S2>::type;

/**
 * @struct Difference
 *
 * Find the elements in \p S1 that are not in \p S2 .
 */
template <typename S1, typename S2>
struct Difference {

    /**
     * Make the compiler perform the difference operation. The return type of this function
     * is the desired result.
     */
    template <std::size_t... Indices>
    static auto make_difference(std::index_sequence<Indices...>) {

        return std::tuple_cat(std::conditional_t<detail::HasType<std::decay_t<std::tuple_element_t<Indices, S1>>, std::decay_t<S2>>::value,
                                                 std::tuple<>, std::tuple<std::tuple_element_t<Indices, S1>>>{}...);
    }

    /**
     * @typedef type
     *
     * Holds the result of this operation.
     */
    using type = decltype(make_difference(std::make_index_sequence<std::tuple_size<S1>::value>{}));
};

/**
 * @typedef DifferenceT
 *
 * The result of a difference operation. @sa Difference
 */
template <typename S1, typename S2>
using DifferenceT = typename Difference<S1, S2>::type;

/**
 * @struct Contains
 *
 * Check whether a tuple contains an element.
 *
 * @tparam Haystack The tuple to search through.
 * @tparam Needle The type to search for.
 */
template <class Haystack, class Needle>
struct Contains;

#ifndef DOXYGEN
template <class Car, class... Cdr, class Needle>
struct Contains<std::tuple<Car, Cdr...>, Needle> : Contains<std::tuple<Cdr...>, Needle> {};

template <class... Cdr, class Needle>
struct Contains<std::tuple<Needle, Cdr...>, Needle> : std::true_type {};

template <class Needle>
struct Contains<std::tuple<>, Needle> : std::false_type {};
#endif

/**
 * @struct Filter
 *
 * Removes duplicate values from a tuple. To use, you should start with an empty tuple for the \p Out parameter.
 *
 * @tparam In The tuple to filter.
 * @tparam Out The output of the filter.
 */
template <class Out, class In>
struct Filter;

#ifndef DOXYGEN
template <class... Out, class InCar, class... InCdr>
struct Filter<std::tuple<Out...>, std::tuple<InCar, InCdr...>> {
    using type =
        std::conditional_t<Contains<std::tuple<Out...>, InCar>::value, typename Filter<std::tuple<Out...>, std::tuple<InCdr...>>::type,
                           typename Filter<std::tuple<Out..., InCar>, std::tuple<InCdr...>>::type>;
};

template <class Out>
struct Filter<Out, std::tuple<>> {
    using type = Out;
};
#endif

template <class T>
using UniqueT = typename Filter<std::tuple<>, T>::type;

// template <class T>
// using c_unique_t = typename filter<std::tuple<>, const T>::type;

/**
 * @struct CUnique
 *
 * Looks for a unique type in a tuple, but decays away all cvref modifiers on the tuple.
 */
template <class T>
struct CUnique {
    /**
     * @typedef type
     *
     * Holds the result of this operation
     */
    using type = UniqueT<std::decay_t<T>>;
};

/**
 * @typedef CUniqueT
 *
 * Same as UniqueT, but it decays away all cvref modifiers on the tuple.
 */
template <class T>
using CUniqueT = typename CUnique<T>::type;

/**
 * @struct DecayElements
 *
 * Applies std::decay_t to each element type in a tuple.
 * Useful for stripping const/ref qualifiers added by DifferenceT/IntersectT
 * when tuple_cat produces const-qualified element types.
 */
template <typename Tuple>
struct DecayElements;

template <typename... Ts>
struct DecayElements<std::tuple<Ts...>> {
    using type = std::tuple<std::decay_t<Ts>...>;
};

template <typename Tuple>
using DecayElementsT = typename DecayElements<Tuple>::type;

/**
 * @struct Reverse
 *
 * Reverses the elements of a tuple.
 */
template <typename First, typename... Args>
struct Reverse {
    /**
     * @typedef type
     *
     * The tuple type with the elements in reverse.
     */
    using type = decltype(std::tuple_cat(std::declval<typename Reverse<Args...>::type>(), std::declval<std::tuple<First>>()));
};

#ifndef DOXYGEN
template <typename First>
struct Reverse<First> {
    using type = std::tuple<First>;
};
#endif

/**
 * @typedef ReverseT
 *
 * Reverses the elements of a tuple.
 */
template <typename... Args>
using ReverseT = Reverse<Args...>::type;

} // namespace einsums::tensor_algebra