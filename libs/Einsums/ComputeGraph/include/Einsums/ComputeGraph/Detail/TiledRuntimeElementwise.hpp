//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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

// ── Scalar reductions ────────────────────────────────────────────────────────

/// Tiled dot: sum over shared tiles of the dense per-tile dot (non-conjugated,
/// matching linear_algebra::dot). X and Y must share a tile grid; a tile present
/// in only one operand contributes nothing.
template <typename T>
T tiled_dot(TiledRuntimeTensor<T> const &A, TiledRuntimeTensor<T> const &B) {
    if (A.tile_sizes() != B.tile_sizes()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dot (tiled): operands must share the same tile grid");
    }
    T acc{0};
    for (auto const &kv : A.tiles()) {
        if (B.has_tile(kv.first)) {
            acc += linear_algebra::dot(kv.second, B.tile(kv.first));
        }
    }
    return acc;
}

/// Tiled trace: sum the diagonals of the diagonal tiles. Requires a rank-2
/// tensor whose two axes share the same tile partition (so each (i,i) tile is
/// square and the global diagonal is exactly the union of their diagonals).
template <typename T>
T tiled_trace(TiledRuntimeTensor<T> const &A) {
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::trace (tiled): input must be rank-2; got rank {}.", A.rank());
    }
    if (A.tile_sizes()[0] != A.tile_sizes()[1]) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace (tiled): the two axes must share the same tile partition (square grid)");
    }
    T         acc{0};
    int const n = static_cast<int>(A.tile_sizes()[0].size());
    for (int i = 0; i < n; ++i) {
        std::vector<int> const coord{i, i};
        if (A.has_tile(coord)) {
            auto const  &t = A.tile(coord);
            size_t const m = std::min(t.dim(0), t.dim(1)); // square, but be safe
            for (size_t k = 0; k < m; ++k) {
                acc += t(std::vector<size_t>{k, k});
            }
        }
    }
    return acc;
}

/// Tiled norm. FROBENIUS = sqrt(sum of per-tile Frobenius norms squared);
/// MAXABS = max over tiles. The induced ONE/INFTY/TWO norms need cross-tile
/// row/column aggregation and are not supported.
template <typename T>
RemoveComplexT<T> tiled_norm(linear_algebra::Norm norm_type, TiledRuntimeTensor<T> const &A) {
    using R = RemoveComplexT<T>;
    if (norm_type == linear_algebra::Norm::FROBENIUS) {
        R sumsq{0};
        for (auto const &kv : A.tiles()) {
            R const n = linear_algebra::norm(linear_algebra::Norm::FROBENIUS, kv.second);
            sumsq += n * n;
        }
        return std::sqrt(sumsq);
    }
    if (norm_type == linear_algebra::Norm::MAXABS) {
        R m{0};
        for (auto const &kv : A.tiles()) {
            m = std::max(m, linear_algebra::norm(linear_algebra::Norm::MAXABS, kv.second));
        }
        return m;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                            "cg::norm (tiled): only FROBENIUS and MAXABS are supported (ONE/INFTY/TWO need cross-tile aggregation)");
}

} // namespace einsums::compute_graph::detail
