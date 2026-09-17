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
 * @brief Whether a column-major BLAS call can address @p impl as a matrix.
 *
 * BLAS reads element ``(i, j)`` of an operand at ``ptr + i + j * lda``, and the
 * tensor holds it at ``ptr + i * stride(0) + j * stride(1)``. Those agree over
 * the whole extent exactly when the conditions below hold, with ``lda`` the
 * larger stride, which is what ``get_lda()`` returns.
 *
 * An axis of extent one is never traversed, so it constrains nothing: a ``1 x n``
 * operand is addressable whatever its row stride, and an ``m x 1`` one whatever
 * its column stride. Skipping that case is not a nicety. It is a single row of a
 * larger matrix, which the fuzzers generate constantly and which BLAS handles
 * perfectly well.
 *
 * This reads strides only, never ``is_column_major()``. A permute_view keeps the
 * storage-order FLAG of its parent while presenting reordered strides, so the
 * flag can disagree with the layout; the large-rank differential fuzzer found
 * exactly that against a view with its slice axes swapped.
 *
 * @warning Not the same question @ref derive_gemm_hint asks, and deliberately
 * not sharing its predicates. That one decides whether an operand is canonical
 * enough to be worth routing to BLAS at all, where declining costs only a fall
 * back to the generic algorithm. This one decides whether a BLAS call already
 * chosen will be CORRECT, so it has to be exact in both directions: a false
 * positive is a wrong answer, and a false negative refuses a call that works.
 */
template <typename T>
bool column_major_addressable(::einsums::detail::TensorImpl<T> const &impl) {
    if (impl.rank() != 2) {
        return false;
    }
    size_t const m = impl.dim(0), n = impl.dim(1);
    size_t const s0 = impl.stride(0), s1 = impl.stride(1);
    // m > 1 pins the row stride to one element; n > 1 pins the column stride to
    // be the leading dimension, and get_lda() reports the larger of the two.
    return (m <= 1 || s0 == 1) && (n <= 1 || s1 >= s0);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
