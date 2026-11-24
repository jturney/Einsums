//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp> // ParsedEinsumSpec + dispatch::erased_string_einsum
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// True when any of the three operand types is a tiled tensor.
template <typename AType, typename BType, typename CType>
inline constexpr bool any_tiled_v = TiledTensorConcept<AType> || TiledTensorConcept<BType> || TiledTensorConcept<CType>;

/**
 * @brief Tiled einsum: compose dense per-tile contractions over a tile grid.
 *
 * Runtime-rank analogue of TensorAlgebra's compile-time `TileAlgebra` dispatch.
 * For every pair of stored A and B tiles that agree on their shared index letters,
 * it calls the dense `dispatch::erased_string_einsum` on the underlying `RuntimeTensor`
 * tiles, accumulating into the matching output tile. Only stored pairs are visited,
 * not the whole tile grid, and they run in grid order, so each output tile sums its
 * contributions in a fixed order. A pair touching a tile with no elements creates
 * its output tile but makes no dense call.
 *
 * @par Semantics
 * `C = c_pf * C + ab_pf * contract(A, B)`. Existing output tiles are scaled by
 * `c_pf` once up front; output tiles are then created on demand wherever a valid
 * contraction path exists (infer-and-create), and each per-tile contribution
 * accumulates with `beta == 1` (a freshly created tile starts zeroed).
 *
 * @par Requirements / limitations
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

    // Up-front output scaling on existing tiles only (created tiles start zeroed). A prefactor of one
    // leaves the tile as it is, but the tile is still materialized, as every existing tile always was.
    for (auto &[coord, tile] : C->tiles()) {
        tile.materialize();
        if (c_pf == T{0}) {
            tile.zero();
        } else if (c_pf != T{1}) {
            tile *= c_pf;
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

    // One contribution: C(c) += ab_pf * A(a) * B(b) for the tile coordinates @p ucoord names. The C
    // tile is created whenever the pair exists, which is what decides the output's tile structure;
    // the dense call is skipped when an operand tile has no elements, since it would add nothing.
    std::vector<int> ccoord(cidx.size());
    auto const       contribute = [&](std::vector<int> const &ucoord, RuntimeTensor<T> const &a_tile, RuntimeTensor<T> const &b_tile) {
        for (size_t ax = 0; ax < cidx.size(); ++ax) {
            ccoord[ax] = ucoord[c_tab[ax]];
        }
        auto &ctile = C->tile(ccoord); // infer-and-create
        ctile.materialize();
        if (a_tile.size() == 0 || b_tile.size() == 0 || ctile.size() == 0) {
            return;
        }
        // beta == 1: existing C tiles were pre-scaled, created tiles start zeroed.
        dispatch::erased_string_einsum<T>(parsed, T{1}, ctile.impl(), ab_pf, a_tile.impl(), b_tile.impl(), false, false);
    };

    // Every C letter must come from A or B for the pairs of stored tiles to cover the grid; the
    // string engine broadcasts nothing, so that is every spec it accepts, and the grid walk below
    // remains only for one it would not.
    std::vector<bool> in_ab(nu, false);
    for (int const u : a_tab) {
        in_ab[u] = true;
    }
    for (int const u : b_tab) {
        in_ab[u] = true;
    }
    bool const pairs_cover_grid = std::ranges::all_of(c_tab, [&](int u) { return in_ab[u]; });

    std::vector<int> ucoord(nu);
    if (!pairs_cover_grid) {
        std::vector<int> acoord(aidx.size()), bcoord(bidx.size());
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
            if (A.has_tile(acoord) && B.has_tile(bcoord)) {
                contribute(ucoord, A.tile(acoord), B.tile(bcoord));
            }
        }
        return;
    }

    // Visit only pairs of stored tiles. B's tiles are indexed by their coordinates on the letters B
    // shares with A, so each A tile meets exactly the B tiles that agree with it; a tile whose own
    // repeated letters disagree (a diagonal 'ii' spec off the tile diagonal) pairs with nothing. The
    // pairs then run in grid order, the order the full walk used, so every C tile accumulates its
    // contributions in the same order and the result is bitwise what it was.
    std::vector<int> shared; // unique positions in both A and B, in order
    {
        std::vector<bool> in_a(nu, false);
        for (int const u : a_tab) {
            in_a[u] = true;
        }
        for (size_t u = 0; u < nu; ++u) {
            if (in_a[u] && std::ranges::find(b_tab, static_cast<int>(u)) != b_tab.end()) {
                shared.push_back(static_cast<int>(u));
            }
        }
    }
    // Assign @p coord's tile indices to their letters in @p uc; false when a repeated letter disagrees.
    auto const assign = [](std::vector<int> const &tab, std::vector<int> const &coord, std::vector<int> &uc, std::vector<bool> &set) {
        for (size_t ax = 0; ax < tab.size(); ++ax) {
            int const u = tab[ax];
            if (set[u] && uc[u] != coord[ax]) {
                return false;
            }
            uc[u]  = coord[ax];
            set[u] = true;
        }
        return true;
    };

    using BTile = std::pair<std::vector<int> const *, RuntimeTensor<T> const *>;
    std::unordered_map<std::vector<int>, std::vector<BTile>, einsums::hashes::ContainerHash<std::vector<int>>> b_by_shared;
    {
        std::vector<int>  uc(nu);
        std::vector<bool> set(nu);
        std::vector<int>  key(shared.size());
        for (auto const &[coord, tile] : B.tiles()) {
            std::fill(set.begin(), set.end(), false);
            if (!assign(b_tab, coord, uc, set)) {
                continue;
            }
            for (size_t k = 0; k < shared.size(); ++k) {
                key[k] = uc[shared[k]];
            }
            b_by_shared[key].emplace_back(&coord, &tile);
        }
    }

    struct Pair {
        size_t                  sentinel;
        RuntimeTensor<T> const *a;
        RuntimeTensor<T> const *b;
    };
    std::vector<Pair> pairs;
    {
        std::vector<bool> set(nu);
        std::vector<int>  key(shared.size());
        for (auto const &[a_coord, a_tile] : A.tiles()) {
            std::fill(set.begin(), set.end(), false);
            if (!assign(a_tab, a_coord, ucoord, set)) {
                continue;
            }
            for (size_t k = 0; k < shared.size(); ++k) {
                key[k] = ucoord[shared[k]];
            }
            auto const it = b_by_shared.find(key);
            if (it == b_by_shared.end()) {
                continue;
            }
            std::vector<bool> const a_set = set;
            std::vector<int> const  a_uc  = ucoord;
            for (auto const &[b_coord, b_tile] : it->second) {
                set    = a_set;
                ucoord = a_uc;
                if (!assign(b_tab, *b_coord, ucoord, set)) {
                    continue;
                }
                size_t sentinel = 0;
                for (size_t u = 0; u < nu; ++u) {
                    sentinel += static_cast<size_t>(ucoord[u]) * stride[u];
                }
                pairs.push_back({sentinel, &a_tile, b_tile});
            }
        }
    }
    std::ranges::sort(pairs, {}, &Pair::sentinel);

    for (auto const &pair : pairs) {
        size_t rem = pair.sentinel;
        for (size_t u = 0; u < nu; ++u) {
            ucoord[u] = static_cast<int>(rem / stride[u]);
            rem %= stride[u];
        }
        contribute(ucoord, *pair.a, *pair.b);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
