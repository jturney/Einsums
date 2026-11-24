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

/// Uniform access to a tiled operand, a @ref TiledRuntimeTensor or a @ref TiledRuntimeTensorView of
/// one. A view sits on its parent's tile grid with each tile cut to its index spaces, so it stores
/// exactly the parent's tiles, and a tile of the view is a slice of the parent's tile.
namespace tiled_access {

template <typename T>
std::vector<std::vector<int>> const &sizes(TiledRuntimeTensor<T> const &x) noexcept {
    return x.tile_sizes();
}
template <typename T>
std::vector<std::vector<int>> const &sizes(TiledRuntimeTensorView<T> const &x) noexcept {
    return x.tile_sizes();
}

/// The stored tiles, keyed by grid coordinate; for a view, its parent's, uncut.
template <typename T>
auto const &stored(TiledRuntimeTensor<T> const &x) noexcept {
    return x.tiles();
}
template <typename T>
auto const &stored(TiledRuntimeTensorView<T> const &x) noexcept {
    return x.parent().tiles();
}

/// Call @p f with the operand's tile at @p coord, given its stored tile @p tile: the tile itself, or
/// the view's slice of it. Nothing is created or materialized.
template <typename T, typename F>
decltype(auto) read(TiledRuntimeTensor<T> const &, std::vector<int> const &, RuntimeTensor<T> const &tile, F &&f) {
    return f(tile);
}
template <typename T, typename F>
decltype(auto) read(TiledRuntimeTensorView<T> const &x, std::vector<int> const &coord, RuntimeTensor<T> const &tile, F &&f) {
    RuntimeTensorView<T> const slice = tile.at_view(x.slice_specs(coord));
    return f(slice);
}

/// Call @p f with the output's tile at @p coord, creating and materializing it when absent: the tile
/// itself, or the view's slice of the parent's tile, so a write through a view touches only its
/// sub-block.
template <typename T, typename F>
decltype(auto) write(TiledRuntimeTensor<T> &x, std::vector<int> const &coord, F &&f) {
    auto &tile = x.tile(coord);
    tile.materialize();
    return f(tile);
}
template <typename T, typename F>
decltype(auto) write(TiledRuntimeTensorView<T> &x, std::vector<int> const &coord, F &&f) {
    auto &tile = x.parent().tile(coord);
    tile.materialize();
    RuntimeTensorView<T> slice = tile.at_view(x.slice_specs(coord));
    return f(slice);
}

} // namespace tiled_access

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
 * - Each operand is a `TiledRuntimeTensor<T>` or a `TiledRuntimeTensorView<T>` of one;
 *   mixed tiled/dense is rejected by the caller's overload constraint. Through a view,
 *   C's existing tiles are scaled and written only within the view's sub-blocks, and an
 *   absent C tile is created whole in the parent.
 * - Indices shared across operands must share an identical tile partition; a
 *   mismatch throws (general re-tiling is out of scope).
 * - Scalar (full-reduction) output is not yet supported.
 * - Input (A, B) tiles must already be materialized; output tiles are
 *   materialized here on demand. Serial over tiles (each dense tile contraction
 *   is itself BLAS/threaded).
 */
template <typename T, typename CType, typename AType, typename BType>
void tiled_runtime_einsum(ParsedEinsumSpec const &parsed, T c_pf, CType *C, T ab_pf, AType const &A, BType const &B) {
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
    absorb(aidx, tiled_access::sizes(A), "A");
    absorb(bidx, tiled_access::sizes(B), "B");
    absorb(cidx, tiled_access::sizes(*C), "C");

    // Up-front output scaling on existing tiles only (created tiles start zeroed). A prefactor of one
    // leaves the tile as it is, but the tile is still materialized, as every existing tile always was.
    {
        std::vector<std::vector<int>> existing;
        existing.reserve(tiled_access::stored(*C).size());
        for (auto const &[coord, tile] : tiled_access::stored(*C)) {
            existing.push_back(coord);
        }
        for (auto const &coord : existing) {
            tiled_access::write(*C, coord, [&](auto &tile) {
                if (c_pf == T{0}) {
                    tile.zero();
                } else if (c_pf != T{1}) {
                    tile *= c_pf;
                }
            });
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
    auto const       contribute = [&](std::vector<int> const &ucoord, std::vector<int> const &a_coord, RuntimeTensor<T> const &a_stored,
                                      std::vector<int> const &b_coord, RuntimeTensor<T> const &b_stored) {
        for (size_t ax = 0; ax < cidx.size(); ++ax) {
            ccoord[ax] = ucoord[c_tab[ax]];
        }
        tiled_access::read(A, a_coord, a_stored, [&](auto const &a_tile) {
            tiled_access::read(B, b_coord, b_stored, [&](auto const &b_tile) {
                tiled_access::write(*C, ccoord, [&](auto &c_tile) { // infer-and-create
                    if (a_tile.size() == 0 || b_tile.size() == 0 || c_tile.size() == 0) {
                        return;
                    }
                    // beta == 1: existing C tiles were pre-scaled, created tiles start zeroed.
                    dispatch::erased_string_einsum<T>(parsed, T{1}, c_tile.impl(), ab_pf, a_tile.impl(), b_tile.impl(), false, false);
                });
            });
        });
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
            auto const a_it = tiled_access::stored(A).find(acoord);
            auto const b_it = tiled_access::stored(B).find(bcoord);
            if (a_it != tiled_access::stored(A).end() && b_it != tiled_access::stored(B).end()) {
                contribute(ucoord, acoord, a_it->second, bcoord, b_it->second);
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
        for (auto const &[coord, tile] : tiled_access::stored(B)) {
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
        std::vector<int> const *a_coord;
        RuntimeTensor<T> const *a;
        std::vector<int> const *b_coord;
        RuntimeTensor<T> const *b;
    };
    std::vector<Pair> pairs;
    {
        std::vector<bool> set(nu);
        std::vector<int>  key(shared.size());
        for (auto const &[a_coord, a_tile] : tiled_access::stored(A)) {
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
                pairs.push_back({sentinel, &a_coord, &a_tile, b_coord, b_tile});
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
        contribute(ucoord, *pair.a_coord, *pair.a, *pair.b_coord, *pair.b);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
