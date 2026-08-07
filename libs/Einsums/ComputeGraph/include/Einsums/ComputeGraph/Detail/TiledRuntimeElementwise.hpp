//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Apply a dense per-tile operation to every populated tile.
 *
 * The building block for elementwise tiled ops (scale / element_transform /
 * axpy): walk the populated tiles of a `TiledRuntimeTensor`, materialize each,
 * and hand the underlying dense `RuntimeTensor` tile to @p op. Absent tiles are
 * rigorously zero and are skipped, which is correct for any op that maps zero to
 * zero (true for scale and y+=a*x; element_transform on a sparse tile set
 * transforms only stored tiles, see its wrapper).
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
/// apply), the same approach as the element_transform_python wrapper.
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

/// Tiled in-place complex conjugate `A := conj(A)`, applied per stored tile and a
/// no-op for real T. Absent tiles are zero and stay absent.
template <typename T>
void tiled_conj(TiledRuntimeTensor<T> *A) {
    tiled_for_each_tile(A, [](RuntimeTensor<T> *tile) { einsums::detail::impl_conj(tile->impl()); });
}

/// Tiled complex-to-real elementwise extraction `Out[tile] = f(In[tile])` where f
/// is one of the TensorImpl kernels real/imag/abs. ``In`` and ``Out`` must share
/// a tile grid; only In's stored tiles are visited. Absent tiles are zero, and the
/// real part, imaginary part, or magnitude of zero is zero. ``Out`` tiles are
/// created on demand. Used by the cg conj/real/imag/abs ops for tiled operands.
template <typename TOut, typename TIn, typename Kernel>
void tiled_complex_to_real(TiledRuntimeTensor<TIn> const &In, TiledRuntimeTensor<TOut> *Out, Kernel &&kernel) {
    if (In.tile_sizes() != Out->tile_sizes()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::real/imag/abs (tiled): In and Out must share the same tile grid");
    }
    for (auto const &kv : In.tiles()) {
        auto &out_tile = Out->tile(kv.first); // infer-and-create (zeroed)
        out_tile.materialize();
        kernel(kv.second.impl(), out_tile.impl());
    }
}

template <typename TOut, typename TIn>
void tiled_real(TiledRuntimeTensor<TIn> const &In, TiledRuntimeTensor<TOut> *Out) {
    tiled_complex_to_real(In, Out, [](auto const &i, auto &o) { einsums::detail::impl_real(i, o); });
}
template <typename TOut, typename TIn>
void tiled_imag(TiledRuntimeTensor<TIn> const &In, TiledRuntimeTensor<TOut> *Out) {
    tiled_complex_to_real(In, Out, [](auto const &i, auto &o) { einsums::detail::impl_imag(i, o); });
}
template <typename TOut, typename TIn>
void tiled_abs(TiledRuntimeTensor<TIn> const &In, TiledRuntimeTensor<TOut> *Out) {
    tiled_complex_to_real(In, Out, [](auto const &i, auto &o) { einsums::detail::impl_abs(i, o); });
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

/**
 * @brief Tiled `C = alpha * (A / B) + beta * C`, tile by tile.
 *
 * All three operands must share an identical tile grid. Only the tiles stored in
 * A are divided: an absent A tile is a rigorous zero and `0/B` is zero, so such a
 * tile contributes nothing and C keeps `beta * C` there. Pre-existing C tiles
 * that A never reaches are therefore still scaled by @p beta, exactly as the
 * dense op would leave them.
 *
 * A tile present in A whose B counterpart is ABSENT is an error rather than an
 * infinity: an absent denominator block is a structurally missing divisor, which
 * in practice means the caller built the denominator with the wrong sparsity.
 * The dense op would silently produce infinities.
 */
template <typename T>
void tiled_direct_division(T alpha, TiledRuntimeTensor<T> const &A, TiledRuntimeTensor<T> const &B, T beta, TiledRuntimeTensor<T> *C) {
    if (A.tile_sizes() != B.tile_sizes() || A.tile_sizes() != C->tile_sizes()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::direct_division (tiled): A, B and C must share the same tile grid");
    }
    std::vector<std::vector<int>> written;
    for (auto const &kv : A.tiles()) {
        auto const &coord = kv.first;
        if (!B.has_tile(coord)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::direct_division (tiled): numerator tile is present but the denominator tile is absent, so the "
                                    "divisor is structurally zero");
        }
        auto &c_tile = C->tile(coord); // infer-and-create (zeroed)
        c_tile.materialize();
        linear_algebra::direct_division(alpha, kv.second, B.tile(coord), beta, &c_tile);
        written.push_back(coord);
    }
    // C tiles the numerator never reached still take their beta scaling.
    for (auto &kv : C->tiles()) {
        if (std::ranges::find(written, kv.first) != written.end()) {
            continue;
        }
        kv.second.materialize();
        linear_algebra::scale(beta, &kv.second);
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

/**
 * @brief Tiled ``C = beta*C + alpha*P(A)``, tile by tile.
 *
 * The permutation acts on the tile GRID and within each tile alike: C's grid
 * along axis ``i`` must equal A's grid along the axis C's ``i``-th index names,
 * and each stored A tile lands in exactly one C tile (its coordinates permuted
 * the same way its axes are). ``beta`` applies exactly once to every stored C
 * tile - including tiles the permutation never reaches - and an absent A tile
 * is a rigorous zero, so its target keeps ``beta*C`` there. The same sparsity
 * discipline as @ref tiled_direct_division. Each dense per-tile permute goes
 * through @ref dispatch::string_permute, i.e. HPTT.
 */
template <typename T>
void tiled_permute(ParsedPermuteSpec const &parsed, T beta, TiledRuntimeTensor<T> *C, T alpha, TiledRuntimeTensor<T> const &A) {
    size_t const rank = parsed.c_indices.size();
    if (rank != parsed.a_indices.size() || rank != static_cast<size_t>(A.rank()) || rank != static_cast<size_t>(C->rank())) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::permute (tiled): spec rank does not match the operand ranks");
    }

    // perm[i] = the A axis that C's axis i takes its extent and tile grid from.
    std::vector<size_t> perm(rank);
    for (size_t i = 0; i < rank; ++i) {
        auto it = std::find(parsed.a_indices.begin(), parsed.a_indices.end(), parsed.c_indices[i]);
        if (it == parsed.a_indices.end()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::permute (tiled): output index '{}' not found in the input indices",
                                    parsed.c_indices[i]);
        }
        perm[i] = static_cast<size_t>(it - parsed.a_indices.begin());
    }

    auto const &a_sizes = A.tile_sizes();
    auto const &c_sizes = C->tile_sizes();
    for (size_t i = 0; i < rank; ++i) {
        if (c_sizes[i] != a_sizes[perm[i]]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::permute (tiled): C's tile grid along axis {} must match A's along the permuted axis {}", i,
                                    perm[i]);
        }
    }

    // beta applies exactly once to every stored C tile, up front; the
    // contribution pass below then accumulates (each target is written by at
    // most one A tile, the permutation being a bijection on coordinates).
    for (auto &kv : C->tiles()) {
        kv.second.materialize();
        if (beta == T{0}) {
            kv.second.zero();
        } else if (beta != T{1}) {
            linear_algebra::scale(beta, &kv.second);
        }
    }

    for (auto const &kv : A.tiles()) {
        std::vector<int> target(rank);
        for (size_t i = 0; i < rank; ++i) {
            target[i] = kv.first[perm[i]];
        }
        auto &c_tile = C->tile(target); // infer-and-create (zeroed)
        c_tile.materialize();
        dispatch::string_permute(parsed, T{1}, &c_tile, alpha, kv.second);
    }
}

/// Tiled conjugated dot `sum conj(A) * B`: per-tile ``true_dot`` over shared
/// tiles, matching @ref tiled_dot's grid and sparsity rules. For real T this
/// coincides with tiled_dot, exactly as the dense pair does.
template <typename T>
T tiled_dotc(TiledRuntimeTensor<T> const &A, TiledRuntimeTensor<T> const &B) {
    if (A.tile_sizes() != B.tile_sizes()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dotc (tiled): operands must share the same tile grid");
    }
    T acc{0};
    for (auto const &kv : A.tiles()) {
        if (B.has_tile(kv.first)) {
            acc += linear_algebra::true_dot(kv.second, B.tile(kv.first));
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

// ── Per-block eigendecomposition (block-diagonal only) ───────────────────────

/// Shared driver for tiled syev/heev: validate a block-diagonal square grid and
/// run the dense per-block eigensolver @p block_op on each diagonal tile,
/// writing that block's eigenvalues into W's matching rank-1 tile. A is
/// overwritten with the per-block eigenvectors in place.
template <typename T, typename WT, typename BlockOp>
void tiled_eig_impl(TiledRuntimeTensor<T> *A, TiledRuntimeTensor<WT> *W, BlockOp block_op, char const *name) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "{}: A must be rank-2 and W rank-1; got {}, {}.", name, A->rank(), W->rank());
    }
    if (A->tile_sizes()[0] != A->tile_sizes()[1]) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: the two axes must share the same tile partition (square grid)", name);
    }
    if (A->tile_sizes()[0] != W->tile_sizes()[0]) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: W's tile partition must match A's row partition", name);
    }
    for (auto const &kv : A->tiles()) {
        if (kv.first[0] != kv.first[1]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: requires a block-diagonal tiled matrix (no off-diagonal tiles)", name);
        }
    }
    int const n = static_cast<int>(A->tile_sizes()[0].size());
    for (int i = 0; i < n; ++i) {
        auto &w = W->tile({i}); // rank-1 eigenvalue block
        w.materialize();
        w.zero(); // absent diagonal block -> zero block -> zero eigenvalues
        std::vector<int> const diag{i, i};
        if (A->has_tile(diag)) {
            auto &a = A->tile(diag);
            a.materialize();
            block_op(&a, &w);
        }
    }
}

/// Tiled real-symmetric eigendecomposition: per diagonal block, dense syev.
template <bool ComputeEigenvectors, typename T>
void tiled_syev(TiledRuntimeTensor<T> *A, TiledRuntimeTensor<T> *W) {
    tiled_eig_impl(
        A, W, [](RuntimeTensor<T> *a, RuntimeTensor<T> *w) { linear_algebra::syev<ComputeEigenvectors>(a, w); }, "cg::syev (tiled)");
}

/// Tiled Hermitian eigendecomposition: per diagonal block, dense heev. W holds
/// the (real) eigenvalues.
template <bool ComputeEigenvectors, typename T>
void tiled_heev(TiledRuntimeTensor<T> *A, TiledRuntimeTensor<RemoveComplexT<T>> *W) {
    tiled_eig_impl(
        A, W, [](RuntimeTensor<T> *a, RuntimeTensor<RemoveComplexT<T>> *w) { linear_algebra::heev<ComputeEigenvectors>(a, w); },
        "cg::heev (tiled)");
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
