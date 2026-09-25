//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Whether @p impl's strides are monotone in its own declared storage order.
 *
 * A permute_view keeps the storage-order FLAG of its parent but presents reordered strides, so
 * ``is_row_major()``/``is_column_major()`` alone cannot prove the canonical layout a GEMM or a
 * flat permute assumes. Found by the large-rank differential fuzzer: a view with the slice axes
 * swapped passed the flag gate and produced wrong results. Extent-1 axes are never traversed, so
 * their (possibly inflated) strides are ignored.
 */
template <typename T>
[[nodiscard]] bool strides_follow_layout(::einsums::detail::TensorImpl<T> const &impl) {
    bool const        row_major = impl.is_row_major();
    std::size_t const rank      = impl.rank();
    std::size_t       prev      = 0;
    bool              first     = true;
    for (std::size_t n = 0; n < rank; ++n) {
        std::size_t const d = row_major ? rank - 1 - n : n;
        if (impl.dim(d) <= 1) {
            continue;
        }
        std::size_t const st = impl.stride(d);
        if (!first && st < prev) {
            return false;
        }
        prev  = st;
        first = false;
    }
    return true;
}

/**
 * @brief Whether a BLAS call can address @p impl as a matrix at all.
 *
 * A GEMM is handed a base pointer and a leading dimension, so the minor axis has to step by one
 * element. A view that drops a LEADING axis of a three-index tensor leaves a rank-two operand whose
 * minor stride is the parent's next extent: a perfectly good operand for the generic algorithm and
 * not a matrix BLAS can describe. @ref strides_follow_layout does not settle it, because what that
 * checks is strides INCREASING in layout order and ``(6, 36)`` increases.
 *
 * Extent-1 axes are never traversed, so an operand whose every axis holds one element is
 * addressable whatever its strides claim.
 */
template <typename T>
[[nodiscard]] bool minor_stride_is_unit(::einsums::detail::TensorImpl<T> const &impl) {
    bool const        row_major = impl.is_row_major();
    std::size_t const rank      = impl.rank();
    for (std::size_t n = 0; n < rank; ++n) {
        std::size_t const d = row_major ? rank - 1 - n : n;
        if (impl.dim(d) <= 1) {
            continue;
        }
        return impl.stride(d) == 1;
    }
    return true;
}

/// A contiguous buffer laid out in its declared storage order: what a flat copy or a batched GEMM
/// over equal-sized slices can address directly.
template <typename T>
[[nodiscard]] bool canonical_dense(::einsums::detail::TensorImpl<T> const &impl) {
    return impl.is_contiguous() && strides_follow_layout(impl);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
