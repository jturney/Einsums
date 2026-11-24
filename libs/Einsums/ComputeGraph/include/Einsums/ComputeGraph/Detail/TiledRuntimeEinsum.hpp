//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp> // ParsedEinsumSpec + dispatch::erased_string_einsum
#include <Einsums/ComputeGraph/Detail/TiledPricing.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorBase/HashFunctions.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
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

/// Highest rank the densified lowering copies tiles at directly; a rank above it takes the slower
/// copy through tensor views. A tiled tensor of rank 12 is far outside anything the library sees.
inline constexpr std::size_t kMaxBlockRank = 12;

/// A strided block of elements: a tile, a view's slice of one, or a tile's place in a dense buffer.
/// Built from the tile's own extents and strides with no allocation, since the densified lowering
/// copies every stored tile on every call.
template <typename P>
struct TileBlock {
    P                                     *data{nullptr};
    std::size_t                            rank{0};
    std::array<std::size_t, kMaxBlockRank> dims{};
    std::array<std::size_t, kMaxBlockRank> strides{};

    [[nodiscard]] bool empty() const noexcept {
        for (std::size_t ax = 0; ax < rank; ++ax) {
            if (dims[ax] == 0) {
                return true;
            }
        }
        return false;
    }
};

/// The block of @p impl, cut to @p range(axis) on each axis when @p range is given.
template <typename P, typename Impl, typename Range>
TileBlock<P> block_of(Impl &impl, std::size_t rank, Range &&range) {
    TileBlock<P> b;
    b.data = impl.data();
    b.rank = rank;
    for (std::size_t ax = 0; ax < rank; ++ax) {
        auto const [first, last] = range(ax, impl.dim(ax));
        b.dims[ax]               = static_cast<std::size_t>(last - first);
        b.strides[ax]            = impl.stride(ax);
        b.data += static_cast<std::ptrdiff_t>(first) * static_cast<std::ptrdiff_t>(b.strides[ax]);
    }
    return b;
}

/// Copy @p src into @p dst, or add it when @p Add, element by element in order: no thread team and
/// no temporaries, since a tile here is a few hundred elements and a region's entry outweighs it.
template <bool Add, typename T>
void block_copy(TileBlock<T const> const &src, TileBlock<T> const &dst) {
    if (src.empty()) {
        return;
    }
    std::size_t const rank = src.rank;
    if (rank == 0) {
        Add ? (*dst.data += *src.data) : (*dst.data = *src.data);
        return;
    }
    std::size_t const                      inner = src.dims[rank - 1];
    std::size_t const                      is    = src.strides[rank - 1];
    std::size_t const                      ds    = dst.strides[rank - 1];
    std::array<std::size_t, kMaxBlockRank> idx{};
    while (true) {
        std::ptrdiff_t so = 0, d_o = 0;
        for (std::size_t ax = 0; ax + 1 < rank; ++ax) {
            so += static_cast<std::ptrdiff_t>(idx[ax] * src.strides[ax]);
            d_o += static_cast<std::ptrdiff_t>(idx[ax] * dst.strides[ax]);
        }
        T const *s = src.data + so;
        T       *d = dst.data + d_o;
        for (std::size_t i = 0; i < inner; ++i) {
            if constexpr (Add) {
                d[i * ds] += s[i * is];
            } else {
                d[i * ds] = s[i * is];
            }
        }
        std::size_t ax = rank - 1;
        while (ax > 0) {
            --ax;
            if (++idx[ax] < src.dims[ax]) {
                break;
            }
            idx[ax] = 0;
            if (ax == 0) {
                return;
            }
        }
        if (rank == 1) {
            return;
        }
    }
}

namespace tiled_access {

/// The block an operand reads at @p coord from its stored tile @p tile: the whole tile, or the view's slice.
template <typename T>
TileBlock<T const> read_block(TiledRuntimeTensor<T> const &, std::vector<int> const &, RuntimeTensor<T> const &tile) {
    return block_of<T const>(tile.impl(), tile.impl().rank(),
                             [](std::size_t, std::size_t n) { return std::pair<std::size_t, std::size_t>{0, n}; });
}
template <typename T>
TileBlock<T const> read_block(TiledRuntimeTensorView<T> const &x, std::vector<int> const &coord, RuntimeTensor<T> const &tile) {
    return block_of<T const>(tile.impl(), tile.impl().rank(), [&](std::size_t ax, std::size_t) {
        auto const &r = x.space_range(ax, coord[ax]);
        return std::pair<std::size_t, std::size_t>{static_cast<std::size_t>(r.first), static_cast<std::size_t>(r.second)};
    });
}

/// The block the output writes at @p coord, creating and materializing the tile when absent.
template <typename T>
TileBlock<T> write_block(TiledRuntimeTensor<T> &x, std::vector<int> const &coord) {
    auto &tile = x.tile(coord);
    tile.materialize();
    return block_of<T>(tile.impl(), tile.impl().rank(),
                       [](std::size_t, std::size_t n) { return std::pair<std::size_t, std::size_t>{0, n}; });
}
template <typename T>
TileBlock<T> write_block(TiledRuntimeTensorView<T> &x, std::vector<int> const &coord) {
    auto &tile = x.parent().tile(coord);
    tile.materialize();
    return block_of<T>(tile.impl(), tile.impl().rank(), [&](std::size_t ax, std::size_t) {
        auto const &r = x.space_range(ax, coord[ax]);
        return std::pair<std::size_t, std::size_t>{static_cast<std::size_t>(r.first), static_cast<std::size_t>(r.second)};
    });
}

} // namespace tiled_access

/// How an eager tiled einsum runs its pairs of tiles. @c Auto prices both with the machine's cost
/// model, as TiledExpansion does for a captured one; the others force a lowering, for tests.
enum class TiledLowering : std::uint8_t {
    Auto,
    PerTile, ///< One dense contraction per pair of stored tiles.
    Densify, ///< Gather A and B into dense buffers, contract once, add the result into C's tiles.
};

/// Densified buffers above this many bytes in all are never allocated for an eager call: the dense
/// buffers are built per call, and a contraction that large prices per tile anyway.
inline constexpr std::size_t kMaxEagerDenseBytes = std::size_t{64} << 20;

/// Which tiles a tiled operand stores, as an order-independent fingerprint: their count and two
/// independent 64-bit sums of a hash of each coordinate. Two different sets of tiles on the same
/// grid share one only by a 128-bit collision, and nothing chooses tile sets to cause one.
struct TileSetFingerprint {
    std::size_t   count{0};
    std::uint64_t h1{0};
    std::uint64_t h2{0};

    bool operator==(TileSetFingerprint const &) const = default;
};

inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

template <typename X>
TileSetFingerprint fingerprint_of(X const &x) {
    TileSetFingerprint f;
    for (auto const &[coord, tile] : tiled_access::stored(x)) {
        std::uint64_t h = 0x2545F4914F6CDD1DULL;
        for (int const c : coord) {
            h = splitmix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(c)));
        }
        f.h1 += h;
        f.h2 += splitmix64(h + 0x632BE59BD9B4E019ULL);
        ++f.count;
    }
    return f;
}

/**
 * @brief An eager tiled contraction's plan, remembered per thread so a repeated call is a replay.
 *
 * Planning walks every pair of stored tiles to price the contraction, and a densified run zeroes
 * dense buffers and works out which output tiles the pairs reach. All of it depends only on the
 * spec, the partitions and which tiles A and B store, so the plan is keyed on those, the tile sets
 * by fingerprint, and a densified entry keeps its buffers: since the same tiles are stored, each
 * call writes the same places, and the rest of each buffer is still zero from the first.
 */
template <typename T>
struct TiledPlan {
    std::string                   spec;
    std::vector<std::vector<int>> a_sizes, b_sizes, c_sizes;
    TileSetFingerprint            a_tiles, b_tiles;
    bool                          densify{false};
    /// Densified only: the output tiles the pairs reach, and the dense buffers and their layouts.
    std::vector<std::vector<int>> reached;
    std::vector<std::size_t>      da, db, dc;
    std::vector<std::vector<int>> oa, ob, oc;
    std::vector<T>                a_buf, b_buf, c_buf;

    [[nodiscard]] std::size_t bytes() const noexcept { return sizeof(T) * (a_buf.size() + b_buf.size() + c_buf.size()); }
};

/// Plans kept per thread, most recent first.
inline constexpr std::size_t kTiledPlanEntries = 16;
/// Bytes of dense buffers the plans of one thread hold at most; the oldest go first.
inline constexpr std::size_t kTiledPlanBytes = std::size_t{256} << 20;

template <typename T>
std::vector<TiledPlan<T>> &tiled_plans() {
    thread_local std::vector<TiledPlan<T>> plans;
    return plans;
}

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
void tiled_runtime_einsum(ParsedEinsumSpec const &parsed, T c_pf, CType *C, T ab_pf, AType const &A, BType const &B,
                          TiledLowering lowering = TiledLowering::Auto) {
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
        last_tiled_lowering() = "grid_walk";
        return;
    }

    // The dense extent of each operand axis, and where each of its tiles starts within it.
    auto const dense_layout = [](std::vector<std::vector<int>> const &sizes, std::vector<size_t> &dims,
                                 std::vector<std::vector<int>> &offsets) {
        dims.assign(sizes.size(), 0);
        offsets.assign(sizes.size(), {});
        for (size_t ax = 0; ax < sizes.size(); ++ax) {
            for (int const n : sizes[ax]) {
                offsets[ax].push_back(static_cast<int>(dims[ax]));
                dims[ax] += static_cast<size_t>(n);
            }
        }
    };

    // Densified, as TiledExpansion lowers a captured contraction: every stored tile of A and B is
    // copied into place in its dense buffer, zero elsewhere, the contraction runs once over the full
    // extents, and its result is added into exactly the C tiles the per-tile path would reach, which
    // this creates as that path would. C's existing tiles were scaled by c_pf above, so adding is the
    // whole update. The sums run in another order than tile by tile, so the result agrees with it to
    // rounding.
    auto const run_densified = [&](TiledPlan<T> &plan) {
        auto const dense_block = [](auto &impl, std::vector<std::vector<int>> const &offsets, std::vector<std::vector<int>> const &sizes,
                                    std::vector<int> const &coord) {
            using P = std::remove_reference_t<decltype(*impl.data())>;
            return block_of<P>(impl, coord.size(), [&](std::size_t ax, std::size_t) {
                auto const first = static_cast<std::size_t>(offsets[ax][static_cast<std::size_t>(coord[ax])]);
                return std::pair<std::size_t, std::size_t>{
                    first, first + static_cast<std::size_t>(sizes[ax][static_cast<std::size_t>(coord[ax])])};
            });
        };
        einsums::detail::TensorImpl<T> a_impl(plan.a_buf.data(), plan.da);
        einsums::detail::TensorImpl<T> b_impl(plan.b_buf.data(), plan.db);
        einsums::detail::TensorImpl<T> c_impl(plan.c_buf.data(), plan.dc);
        for (auto const &[coord, tile] : tiled_access::stored(A)) {
            block_copy<false, T>(tiled_access::read_block(A, coord, tile), dense_block(a_impl, plan.oa, tiled_access::sizes(A), coord));
        }
        for (auto const &[coord, tile] : tiled_access::stored(B)) {
            block_copy<false, T>(tiled_access::read_block(B, coord, tile), dense_block(b_impl, plan.ob, tiled_access::sizes(B), coord));
        }
        // C's buffer is zeroed every call: a previous result holding a NaN would otherwise survive a
        // kernel that scales C by zero rather than overwriting it.
        std::ranges::fill(plan.c_buf, T{0});
        dispatch::erased_string_einsum<T>(parsed, T{0}, c_impl, ab_pf, a_impl, b_impl, false, false);
        einsums::detail::TensorImpl<T> const &c_result = c_impl;
        for (auto const &coord : plan.reached) {
            block_copy<true, T>(dense_block(c_result, plan.oc, tiled_access::sizes(*C), coord), tiled_access::write_block(*C, coord));
        }
        last_tiled_lowering() = "densified";
    };

    bool const         blocks_fit = aidx.size() <= kMaxBlockRank && bidx.size() <= kMaxBlockRank && cidx.size() <= kMaxBlockRank;
    bool const         cacheable  = lowering == TiledLowering::Auto && blocks_fit;
    TileSetFingerprint a_tiles, b_tiles;
    bool               known_per_tile = false; // a remembered plan said per tile: skip pricing
    if (cacheable) {
        a_tiles     = fingerprint_of(A);
        b_tiles     = fingerprint_of(B);
        auto &plans = tiled_plans<T>();
        auto  hit   = std::ranges::find_if(plans, [&](TiledPlan<T> const &p) {
            return p.a_tiles == a_tiles && p.b_tiles == b_tiles && p.spec == parsed.raw && p.a_sizes == tiled_access::sizes(A) &&
                   p.b_sizes == tiled_access::sizes(B) && p.c_sizes == tiled_access::sizes(*C);
        });
        if (hit != plans.end()) {
            std::rotate(plans.begin(), hit, hit + 1); // most recent first
            if (plans.front().densify) {
                run_densified(plans.front());
                return;
            }
            known_per_tile = true;
        }
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

    auto const decode = [&](size_t sentinel) {
        for (size_t u = 0; u < nu; ++u) {
            ucoord[u] = static_cast<int>(sentinel / stride[u]);
            sentinel %= stride[u];
        }
    };

    if (known_per_tile) {
        for (auto const &pair : pairs) {
            decode(pair.sentinel);
            contribute(ucoord, *pair.a_coord, *pair.a, *pair.b_coord, *pair.b);
        }
        last_tiled_lowering() = "per_tile";
        return;
    }

    TiledPlan<T> plan;
    plan.spec    = parsed.raw;
    plan.a_sizes = tiled_access::sizes(A);
    plan.b_sizes = tiled_access::sizes(B);
    plan.c_sizes = tiled_access::sizes(*C);
    plan.a_tiles = a_tiles;
    plan.b_tiles = b_tiles;
    dense_layout(plan.a_sizes, plan.da, plan.oa);
    dense_layout(plan.b_sizes, plan.db, plan.ob);
    dense_layout(plan.c_sizes, plan.dc, plan.oc);
    auto const   volume = [](std::vector<size_t> const &d) { return std::accumulate(d.begin(), d.end(), size_t{1}, std::multiplies<>{}); };
    size_t const bytes  = sizeof(T) * (volume(plan.da) + volume(plan.db) + volume(plan.dc));

    if (!pairs.empty() && blocks_fit) {
        if (lowering == TiledLowering::Densify) {
            plan.densify = true;
        } else if (lowering == TiledLowering::Auto && bytes <= kMaxEagerDenseBytes) {
            std::vector<bool> in_c(nu, false), in_a(nu, false);
            for (int const u : c_tab) {
                in_c[u] = true;
            }
            for (int const u : a_tab) {
                in_a[u] = true;
            }
            TiledContractionShape const shape{.part = &part, .in_c = in_c, .in_a = in_a, .element_size = sizeof(T)};
            CostModel const            &model    = eager_tiled_cost_model();
            double                      by_tiles = 0.0;
            for (auto const &pair : pairs) {
                decode(pair.sentinel);
                by_tiles += tile_pair_cost_us(model, shape, ucoord);
            }
            plan.densify = densified_cost_us(model, shape) < by_tiles;
        }
    }

    if (plan.densify) {
        std::set<std::vector<int>> reached;
        for (auto const &pair : pairs) {
            decode(pair.sentinel);
            for (size_t ax = 0; ax < cidx.size(); ++ax) {
                ccoord[ax] = ucoord[c_tab[ax]];
            }
            reached.insert(ccoord);
        }
        plan.reached.assign(reached.begin(), reached.end());
        plan.a_buf.assign(volume(plan.da), T{0});
        plan.b_buf.assign(volume(plan.db), T{0});
        plan.c_buf.assign(volume(plan.dc), T{0});
    }

    if (cacheable) {
        // Remember the plan, most recent first, within the entry and byte budgets.
        auto &plans = tiled_plans<T>();
        std::erase_if(plans, [&](TiledPlan<T> const &p) {
            return p.a_tiles == plan.a_tiles && p.b_tiles == plan.b_tiles && p.spec == plan.spec && p.a_sizes == plan.a_sizes &&
                   p.b_sizes == plan.b_sizes && p.c_sizes == plan.c_sizes;
        });
        plans.insert(plans.begin(), std::move(plan));
        std::size_t held = 0;
        std::size_t keep = 0;
        while (keep < plans.size() && keep < kTiledPlanEntries && held + plans[keep].bytes() <= kTiledPlanBytes) {
            held += plans[keep].bytes();
            ++keep;
        }
        if (keep == 0) {
            keep = 1; // the plan this call runs, even when it alone is over the budget
        }
        plans.resize(keep);
        TiledPlan<T> &current = plans.front();
        if (current.densify) {
            run_densified(current);
            return;
        }
    } else if (plan.densify) {
        run_densified(plan);
        return;
    }

    for (auto const &pair : pairs) {
        decode(pair.sentinel);
        contribute(ucoord, *pair.a_coord, *pair.a, *pair.b_coord, *pair.b);
    }
    last_tiled_lowering() = "per_tile";
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
