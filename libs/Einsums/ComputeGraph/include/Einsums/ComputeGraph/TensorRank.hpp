//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// True when the tensor type carries its rank as a static compile-time
// constant (the GeneralTensor / TensorView family). False for runtime-rank
// tensors (RuntimeTensor / future BlockRuntimeTensor / TiledRuntimeTensor),
// which expose ``Rank == einsums::dynamic_rank`` as a sentinel.
template <typename T>
concept HasCompileTimeRank = requires {
    { std::remove_cvref_t<T>::Rank } -> std::convertible_to<std::size_t>;
} && (std::remove_cvref_t<T>::Rank != einsums::dynamic_rank);

// A tensor whose rank is only known at runtime via a member ``.rank()``.
// Used as the dispatch tag for cg::einsum and other ComputeGraph operations
// when they need a code path that doesn't bake rank into the type.
template <typename T>
concept RuntimeRankTensorConcept = BasicTensorConcept<T> && (!HasCompileTimeRank<T>)&&requires(T const &t) {
    { t.rank() } -> std::convertible_to<std::size_t>;
};

namespace detail {

// Return the rank of a tensor, sourced at compile time when the tensor
// type carries a static ``::Rank`` (the GeneralTensor/TensorView family)
// and at runtime via ``.rank()`` otherwise (the RuntimeTensor family).
//
// Used by per-rank dispatch so the same function-template body works
// against both typed tensors (Tensor<T,R>, perf-identical to the previous
// ``if constexpr (AType::Rank == ...)`` form because the comparison still
// folds at compile time) and runtime-rank tensors (RuntimeTensor<T, Alloc>,
// where the comparison becomes a real but negligible branch dwarfed by the
// contraction work).
template <typename T>
constexpr std::size_t tensor_rank(T const &t) noexcept {
    using Clean = std::remove_cvref_t<T>;
    if constexpr (requires { Clean::Rank; }) {
        if constexpr (Clean::Rank == einsums::dynamic_rank) {
            return t.rank(); // dynamic-rank sentinel, defer to runtime
        } else {
            return Clean::Rank;
        }
    } else {
        return t.rank();
    }
}

// Walk an index-list selection, one maximal contiguous run at a time.
//
// gather, scatter and scatter_add are the same traversal: one side is addressed
// through the index lists, the other runs 0..extent on every axis. Written
// directly it is an odometer that rebuilds both offsets from scratch per
// element, which costs a multiply-add per axis per element and, worse, hides
// the shape the callers hit most. Domain restriction routinely selects a whole
// axis with ``range(n)`` - gather has no wildcard, so callers spell it out -
// and when that is the FASTEST axis and both sides step by one, the selection
// along it is a contiguous block that the element loop was copying a
// multiply-add at a time.
//
// @p op is called as ``op(indexed_offset, linear_offset, length)``, with length
// 1 unless the fastest axis collapsed into a run.
//
// @p idx_str are the strides of the side addressed through @p indices, and
// @p lin_str those of the side walked linearly; which is source and which is
// destination is the caller's business, and is what distinguishes gather from
// scatter.
template <typename Fn>
void for_each_selection_run(std::vector<std::vector<std::size_t>> const &indices, std::vector<std::size_t> const &extents,
                            std::vector<std::size_t> const &idx_str, std::vector<std::size_t> const &lin_str, Fn &&op) {
    std::size_t const N = indices.size();
    for (std::size_t k = 0; k < N; ++k) {
        if (extents[k] == 0) {
            return; // an empty selection is a no-op, not an error
        }
    }

    // The fastest axis collapses into one run when its indices ascend by one
    // and neither side skips elements along it.
    bool run0 = idx_str[0] == 1 && lin_str[0] == 1;
    for (std::size_t i = 1; run0 && i < extents[0]; ++i) {
        run0 = indices[0][i] == indices[0][i - 1] + 1;
    }

    // What each axis contributes to the indexed offset, so the walk adds a
    // table entry per axis instead of multiplying an index by a stride.
    std::vector<std::vector<std::size_t>> off(N);
    for (std::size_t k = 0; k < N; ++k) {
        off[k].resize(extents[k]);
        for (std::size_t i = 0; i < extents[k]; ++i) {
            off[k][i] = indices[k][i] * idx_str[k];
        }
    }

    std::size_t outer = 1;
    for (std::size_t k = 1; k < N; ++k) {
        outer *= extents[k];
    }

    std::vector<std::size_t> idx(N, 0);
    for (std::size_t count = 0; count < outer; ++count) {
        std::size_t i_off = 0, l_off = 0;
        for (std::size_t k = 1; k < N; ++k) {
            i_off += off[k][idx[k]];
            l_off += idx[k] * lin_str[k];
        }
        if (run0) {
            op(i_off + off[0][0], l_off, extents[0]);
        } else {
            for (std::size_t i = 0; i < extents[0]; ++i) {
                op(i_off + off[0][i], l_off + i * lin_str[0], std::size_t{1});
            }
        }
        // Axis 1 fastest among the outer axes, matching the element walk this
        // replaces. Correctness-only ordering: the indexed side is a gather, so
        // it is not contiguous across axes anyway.
        for (std::size_t k = 1; k < N; ++k) {
            if (++idx[k] < extents[k]) {
                break;
            }
            idx[k] = 0;
        }
    }
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
