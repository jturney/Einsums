//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file IndexTable.hpp
/// @brief Mapping compile-time index packs onto a unique index list, for the templated engine.
///
/// Only the compile-time-index engine uses this, so it lives with that engine rather than in
/// TensorBase, whose headers ComputeGraph includes.

#include <Einsums/Config/Namespace.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

EINSUMS_NAMESPACE_BEGIN()

template <int __I, typename Head, typename Index>
int compile_index_table(std::tuple<Head> const &, Index const &, int &out) {
    if constexpr (std::is_same_v<Head, Index>) {
        out = __I;
    } else {
        out = -1;
    }
    return 0;
}

template <int __I, typename Head, typename... UniqueIndices, typename Index>
auto compile_index_table(std::tuple<Head, UniqueIndices...> const &, Index const &index, int &out) -> int
    requires(sizeof...(UniqueIndices) != 0)
{
    if constexpr (std::is_same_v<Head, Index>) {
        out = __I;
    } else {
        compile_index_table<__I + 1>(std::tuple<UniqueIndices...>(), index, out);
    }
    return 0;
}

template <typename... UniqueIndices, typename... Indices, size_t... __I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out,
                         std::index_sequence<__I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<__I>(to_inds), out[__I])...};
}

/**
 * @brief Turn a list of indices into a link table.
 *
 * Takes a list of indices and creates a mapping so that an index list for a tensor can reference the unique index list.
 */
template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

template <typename... UniqueIndices, typename... Indices, size_t... __I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out, std::index_sequence<__I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<__I>(to_inds), out[__I])...};
}

template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

EINSUMS_NAMESPACE_END()
