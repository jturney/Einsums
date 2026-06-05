//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/StringDispatch.hpp> // pulls ComputeGraph/EinsumSpec.hpp (ParsedEinsumSpec) + dispatch::string_einsum
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace einsums::compute_graph::detail {

/// True when any of the three operand types is a tiled tensor.
template <typename AType, typename BType, typename CType>
inline constexpr bool any_tiled_v =
    IsTiledTensorV<std::remove_cvref_t<AType>> || IsTiledTensorV<std::remove_cvref_t<BType>> || IsTiledTensorV<std::remove_cvref_t<CType>>;

/**
 * @brief Tier-B (B1) tiled einsum: compose dense per-tile contractions over a tile grid.
 *
 * Runtime-rank analogue of TensorAlgebra's compile-time `TileAlgebra` dispatch.
 * It walks the grid of all unique einsum indices and, for every combination of
 * tiles where the A and B operands are both populated, calls the dense
 * `dispatch::string_einsum` on the underlying `RuntimeTensor` tiles, accumulating
 * into the matching output tile.
 *
 * @par Semantics
 * `C = c_pf * C + ab_pf * contract(A, B)`. Existing output tiles are scaled by
 * `c_pf` once up front; output tiles are then created on demand wherever a valid
 * contraction path exists (infer-and-create), and each per-tile contribution
 * accumulates with `beta == 1` (a freshly created tile starts zeroed).
 *
 * @par Requirements / limitations (B1)
 * - All three operands are `TiledRuntimeTensor<T>`; mixed tiled/dense is rejected
 *   by the caller's overload constraint.
 * - Indices shared across operands must share an identical tile partition; a
 *   mismatch throws (general re-tiling is out of scope).
 * - Scalar (full-reduction) output is not yet supported.
 * - Input (A, B) tiles must already be materialized; output tiles are
 *   materialized here on demand. Serial over tiles (each dense tile contraction
 *   is itself BLAS/threaded).
 */
template <typename T>
void tiled_runtime_einsum(ParsedEinsumSpec const &parsed, T c_pf, TiledRuntimeTensor<T> *C, T ab_pf, TiledRuntimeTensor<T> const &A,
                          TiledRuntimeTensor<T> const &B) {
    auto const &cidx = parsed.c_indices;
    auto const &aidx = parsed.a_indices;
    auto const &bidx = parsed.b_indices;

    if (cidx.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::einsum (tiled): scalar-output / full-reduction over tiled tensors is not "
                                                       "supported yet");
    }

    // Unique index letters in a stable order (C, then new from A, then new from B).
    std::vector<std::string> unique;
    auto                     add_unique = [&](std::vector<std::string> const &v) {
        for (auto const &s : v) {
            if (std::ranges::find(unique, s) == unique.end()) {
                unique.push_back(s);
            }
        }
    };
    add_unique(cidx);
    add_unique(aidx);
    add_unique(bidx);

    size_t const nu  = unique.size();
    auto         pos = [&](std::string const &s) { return static_cast<int>(std::ranges::find(unique, s) - unique.begin()); };

    // Per-unique-index grid range (#tiles) + partition, validated for alignment
    // across every operand that carries the index.
    std::vector<int>              grid(nu, -1);
    std::vector<std::vector<int>> part(nu);
    auto absorb = [&](std::vector<std::string> const &idx, std::vector<std::vector<int>> const &sizes, char const *who) {
        for (size_t ax = 0; ax < idx.size(); ++ax) {
            int         u = pos(idx[ax]);
            auto const &p = sizes[ax];
            if (grid[u] < 0) {
                grid[u] = static_cast<int>(p.size());
                part[u] = p;
            } else if (grid[u] != static_cast<int>(p.size()) || part[u] != p) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "cg::einsum (tiled): tile partition for index '{}' on operand {} does not match the other "
                                        "operands; aligned tile partitions are required",
                                        idx[ax], who);
            }
        }
    };
    absorb(aidx, A.tile_sizes(), "A");
    absorb(bidx, B.tile_sizes(), "B");
    absorb(cidx, C->tile_sizes(), "C");

    // Up-front output scaling on existing tiles only (created tiles start zeroed).
    {
        std::vector<std::vector<int>> existing;
        existing.reserve(C->tiles().size());
        for (auto const &kv : C->tiles()) {
            existing.push_back(kv.first);
        }
        for (auto const &coord : existing) {
            auto &tile = C->tile(coord);
            tile.materialize();
            if (c_pf == T{0}) {
                tile.zero();
            } else {
                tile *= c_pf;
            }
        }
    }

    // Row-major strides over the unique grid so a single sentinel enumerates
    // every tile combination.
    std::vector<size_t> stride(nu, 1);
    for (int i = static_cast<int>(nu) - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * static_cast<size_t>(grid[i + 1]);
    }
    size_t const total = (nu == 0) ? 0 : stride[0] * static_cast<size_t>(grid[0]);

    // Precompute operand-axis -> unique-index maps.
    std::vector<int> a_tab(aidx.size()), b_tab(bidx.size()), c_tab(cidx.size());
    for (size_t ax = 0; ax < aidx.size(); ++ax) {
        a_tab[ax] = pos(aidx[ax]);
    }
    for (size_t ax = 0; ax < bidx.size(); ++ax) {
        b_tab[ax] = pos(bidx[ax]);
    }
    for (size_t ax = 0; ax < cidx.size(); ++ax) {
        c_tab[ax] = pos(cidx[ax]);
    }

    std::vector<int> ucoord(nu), acoord(aidx.size()), bcoord(bidx.size()), ccoord(cidx.size());
    for (size_t s = 0; s < total; ++s) {
        size_t rem = s;
        for (size_t u = 0; u < nu; ++u) {
            ucoord[u] = static_cast<int>(rem / stride[u]);
            rem %= stride[u];
        }
        for (size_t ax = 0; ax < aidx.size(); ++ax) {
            acoord[ax] = ucoord[a_tab[ax]];
        }
        for (size_t ax = 0; ax < bidx.size(); ++ax) {
            bcoord[ax] = ucoord[b_tab[ax]];
        }
        if (!A.has_tile(acoord) || !B.has_tile(bcoord)) {
            continue;
        }
        for (size_t ax = 0; ax < cidx.size(); ++ax) {
            ccoord[ax] = ucoord[c_tab[ax]];
        }
        auto &ctile = C->tile(ccoord); // infer-and-create
        ctile.materialize();
        // beta == 1: existing C tiles were pre-scaled, created tiles start zeroed.
        dispatch::string_einsum(parsed, T{1}, &ctile, ab_pf, A.tile(acoord), B.tile(bcoord));
    }
}

} // namespace einsums::compute_graph::detail
