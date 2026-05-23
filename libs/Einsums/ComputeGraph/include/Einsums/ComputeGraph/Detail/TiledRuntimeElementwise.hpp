//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <cstddef>
#include <stdexcept>

namespace einsums::compute_graph::detail {

/**
 * @brief Apply a dense per-tile operation to every populated tile.
 *
 * The building block for elementwise tiled ops (scale / element_transform /
 * axpy): walk the populated tiles of a `TiledRuntimeTensor`, materialize each,
 * and hand the underlying dense `RuntimeTensor` tile to @p op. Absent tiles are
 * rigorously zero and are skipped, which is correct for any op that maps zero to
 * zero (true for scale and y+=a*x; element_transform on a sparse tile set
 * transforms only stored tiles — see its wrapper).
 *
 * Serial over tiles; each dense op is itself BLAS/threaded.
 */
template <typename T, typename Op>
void tiled_for_each_tile(TiledRuntimeTensor<T> *A, Op &&op) {
    for (auto &kv : A->tiles()) {
        kv.second.materialize();
        op(&kv.second);
    }
}

/// Tiled `A *= factor` (per-tile dense scale).
template <typename T>
void tiled_scale(T factor, TiledRuntimeTensor<T> *A) {
    tiled_for_each_tile(A, [factor](RuntimeTensor<T> *tile) { linear_algebra::scale(factor, tile); });
}

/// Tiled in-place unary map `A[i] = op(A[i])` over every stored tile. Walks the
/// tile's contiguous storage directly (a RuntimeTensor tile has runtime rank, so
/// the RankTensorConcept-constrained tensor_algebra::element_transform doesn't
/// apply) — same approach as the element_transform_python wrapper.
template <typename T, typename UnaryOp>
void tiled_element_transform(TiledRuntimeTensor<T> *A, UnaryOp unary_op) {
    tiled_for_each_tile(A, [&unary_op](RuntimeTensor<T> *tile) {
        T           *d = tile->data();
        size_t const n = tile->size();
        for (size_t i = 0; i < n; ++i) {
            d[i] = unary_op(d[i]);
        }
    });
}

/**
 * @brief Tiled `Y += alpha * X` (axpy), tile-by-tile.
 *
 * X and Y must share an identical tile grid. A tile present in X but not in Y is
 * created in Y on demand (it starts zeroed, so `Y += alpha*X` is correct). A
 * tile absent in X contributes nothing.
 */
template <typename T>
void tiled_axpy(T alpha, TiledRuntimeTensor<T> const &X, TiledRuntimeTensor<T> *Y) {
    if (X.tile_sizes() != Y->tile_sizes()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::axpy (tiled): X and Y must share the same tile grid");
    }
    for (auto const &kv : X.tiles()) {
        auto const &coord  = kv.first;
        auto const &x_tile = kv.second;
        auto       &y_tile = Y->tile(coord); // infer-and-create (zeroed)
        y_tile.materialize();
        linear_algebra::axpy(alpha, x_tile, &y_tile);
    }
}

} // namespace einsums::compute_graph::detail
