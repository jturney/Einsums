//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorAlgebra/Backends/Dispatch.hpp>
#include <Einsums/TensorAlgebra/Detail/IndexTable.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(tensor_algebra::detail)

template <bool OnlyUseGenericAlgorithm, bool ConjA, bool ConjB, TiledTensorConcept AType, TiledTensorConcept BType,
          TiledTensorConcept CType, typename... CIndices, typename... AIndices, typename... BIndices>
    requires(CType::Rank != 0)
auto einsum_special_dispatch(typename CType::ValueType const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = CType::Rank;

    using ADataType = typename AType::ValueType;
    using BDataType = typename BType::ValueType;
    using CDataType = typename CType::ValueType;

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides, true);

    std::array<int, ARank> A_index_table;
    std::array<int, BRank> B_index_table;
    std::array<int, CRank> C_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);
    compile_index_table(unique_indices, C_indices, C_index_table);

    if (C_prefactor == CDataType(0.0)) {
        if constexpr (!IsTensorViewV<CType>) {
            C->zero_no_clear();
        } else {
            C->zero();
        }
    } else if (C_prefactor != CDataType(1.0)) {
        *C *= C_prefactor;
    }

    // The unique indices list C's first, and the grid is row-major over them, so a sentinel splits
    // into an output tile (the leading CRank indices) and a step over the contracted ones. Threads
    // divide the output tiles: each tile has one owner, which sums its contributions in grid order,
    // so no tile is locked and every run gives the same result. Only creating a tile, an insertion
    // into C's container, is locked.
    std::size_t const total     = unique_grid[0] * unique_strides[0];
    std::size_t const per_tile  = unique_strides[CRank - 1];
    std::size_t const out_tiles = per_tile == 0 ? 0 : total / per_tile;

    // Thread only a contraction worth a region: the dense extents bound the work from above.
    std::size_t link_elements = 1;
    {
        std::array<std::size_t, std::tuple_size<decltype(unique_indices)>::value> extent{};
        for (int i = 0; i < ARank; i++) {
            extent[A_index_table[i]] = A.dim(i);
        }
        for (int i = 0; i < BRank; i++) {
            extent[B_index_table[i]] = B.dim(i);
        }
        for (std::size_t u = CRank; u < extent.size(); ++u) {
            link_elements *= extent[u];
        }
    }
    bool const threaded = generic_walk_wants_threads(C->size(), link_elements);

    auto const contract_into = [&](std::size_t out_tile) {
        std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;
        std::array<int, ARank>                                               A_tile_index;
        std::array<int, BRank>                                               B_tile_index;
        std::array<int, CRank>                                               C_tile_index;
        std::remove_reference_t<decltype(C->tile(C_tile_index))>            *C_tile = nullptr;
        for (std::size_t step = 0; step < per_tile; ++step) {
            sentinel_to_indices(out_tile * per_tile + step, unique_strides, unique_index_table);
            for (int i = 0; i < ARank; i++) {
                A_tile_index[i] = unique_index_table[A_index_table[i]];
            }
            for (int i = 0; i < BRank; i++) {
                B_tile_index[i] = unique_index_table[B_index_table[i]];
            }
            if (step == 0) {
                for (int i = 0; i < CRank; i++) {
                    C_tile_index[i] = unique_index_table[C_index_table[i]];
                }
                if (C->has_zero_size(C_tile_index)) {
                    return;
                }
            }
            if (!A.has_tile(A_tile_index) || !B.has_tile(B_tile_index) || A.has_zero_size(A_tile_index) || B.has_zero_size(B_tile_index)) {
                continue;
            }
            if (C_tile == nullptr) {
                C->lock();
                C_tile = &C->tile(C_tile_index);
                C->unlock();
            }
            einsum<OnlyUseGenericAlgorithm, false, ConjA, ConjB>(CDataType{1.0}, C_indices, C_tile, AB_prefactor, A_indices,
                                                                 A.tile(A_tile_index), B_indices, B.tile(B_tile_index));
        }
    };

    if (threaded) {
#pragma omp parallel for schedule(dynamic)
        for (std::size_t out_tile = 0; out_tile < out_tiles; ++out_tile) {
            contract_into(out_tile);
        }
    } else {
        for (std::size_t out_tile = 0; out_tile < out_tiles; ++out_tile) {
            contract_into(out_tile);
        }
    }
}

template <bool OnlyUseGenericAlgorithm, bool ConjA, bool ConjB, TiledTensorConcept AType, TiledTensorConcept BType, ScalarConcept CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum_special_dispatch(ValueTypeT<CType> const C_prefactor, std::tuple<CIndices...> const &C_indices, CType *C,
                             BiggestTypeT<typename AType::ValueType, typename BType::ValueType> const AB_prefactor,
                             std::tuple<AIndices...> const &A_indices, AType const &A, std::tuple<BIndices...> const &B_indices,
                             BType const &B) -> void {

    constexpr size_t ARank = AType::Rank;
    constexpr size_t BRank = BType::Rank;
    constexpr size_t CRank = 0;

    using ADataType = typename AType::ValueType;
    using BDataType = typename BType::ValueType;
    using CDataType = ValueTypeT<CType>;

    constexpr auto unique_indices = UniqueT<std::tuple<CIndices..., AIndices..., BIndices...>>();
    auto           unique_grid    = get_grid_ranges_for_many(*C, C_indices, A, A_indices, B, B_indices, unique_indices);

    auto unique_strides = std::array<size_t, std::tuple_size<decltype(unique_indices)>::value>();

    dims_to_strides(unique_grid, unique_strides, true);

    std::array<int, ARank> A_index_table;
    std::array<int, BRank> B_index_table;

    compile_index_table(unique_indices, A_indices, A_index_table);
    compile_index_table(unique_indices, B_indices, B_index_table);

    if (C_prefactor == CDataType(0.0)) {
        *C = CDataType{0.0};
    } else if (C_prefactor != CDataType(1.0)) {
        *C *= C_prefactor;
    }

    CDataType out{0.0};

    // Every index is contracted: thread only a reduction worth a region, bounded by the dense extents.
    std::size_t link_elements = 1;
    for (int i = 0; i < ARank; i++) {
        link_elements *= A.dim(i);
    }
    bool const threaded = generic_walk_wants_threads(1, link_elements);

#pragma omp parallel for reduction(+ : out) if (threaded)
    for (size_t sentinel = 0; sentinel < unique_grid[0] * unique_strides[0]; sentinel++) {
        std::array<size_t, std::tuple_size<decltype(unique_indices)>::value> unique_index_table;

        sentinel_to_indices(sentinel, unique_strides, unique_index_table);
        std::array<int, ARank> A_tile_index;
        std::array<int, BRank> B_tile_index;

        for (int i = 0; i < ARank; i++) {
            A_tile_index[i] = unique_index_table[A_index_table[i]];
        }

        for (int i = 0; i < BRank; i++) {
            B_tile_index[i] = unique_index_table[B_index_table[i]];
        }

        if (!A.has_tile(A_tile_index) || !B.has_tile(B_tile_index) || A.has_zero_size(A_tile_index) || B.has_zero_size(B_tile_index)) {
            continue;
        }

        CDataType C_tile{0.0};
        einsum<OnlyUseGenericAlgorithm, false, ConjA, ConjB>(CDataType{0.0}, C_indices, &C_tile, AB_prefactor, A_indices,
                                                             A.tile(A_tile_index), B_indices, B.tile(B_tile_index));
        out += C_tile;
    }

    *C += out;
}
EINSUMS_NAMESPACE_END(tensor_algebra::detail)
