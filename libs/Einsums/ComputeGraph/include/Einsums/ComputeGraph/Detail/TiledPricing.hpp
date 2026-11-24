//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

struct CostModel;

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief The index roles a tiled contraction is priced by, one entry per unique index letter.
 *
 * A letter in C came from A or from B, and is an M or an N extent of each tile pair's GEMM; one
 * absent from C is contracted, a K extent. @c part holds each letter's tile partition.
 */
struct TiledContractionShape {
    std::vector<std::vector<int>> const *part{nullptr};
    std::vector<bool>                    in_c;
    std::vector<bool>                    in_a;
    std::size_t                          element_size{0};
};

/**
 * @brief Estimated microseconds for one tile pair's contraction, the tiles named by @p ucoord.
 *
 * Summed over every contributing pair, this is what running the contraction tile by tile costs;
 * each estimate carries a call's fixed overhead, which is what makes many small tiles lose to one
 * dense contraction. Shared by TiledExpansion and the eager tiled einsum, so both price alike.
 */
[[nodiscard]] EINSUMS_EXPORT double tile_pair_cost_us(CostModel const &model, TiledContractionShape const &shape,
                                                      std::vector<int> const &ucoord);

/**
 * @brief Estimated microseconds for the densified contraction: one contraction over the full extents,
 *        plus gathering A and B into dense buffers and accumulating the result back into C's tiles.
 *
 * The gather and scatter are memory traffic and are priced at bandwidth: A is M x K, B is K x N, C
 * is M x N and is touched twice.
 */
[[nodiscard]] EINSUMS_EXPORT double densified_cost_us(CostModel const &model, TiledContractionShape const &shape);

/// @brief How the most recent eager tiled einsum on this thread ran: "per_tile", "densified", or
///        "grid_walk" (a spec whose output letters the inputs do not cover). Test introspection only,
///        like @c dispatch::last_dispatch_route.
[[nodiscard]] EINSUMS_EXPORT char const *&last_tiled_lowering();

/// @brief The cost model an eager tiled einsum prices its lowering with: the machine's default, built
///        on first use and shared by every thread after.
[[nodiscard]] EINSUMS_EXPORT CostModel const &eager_tiled_cost_model();

EINSUMS_NAMESPACE_END(compute_graph::detail)
