//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ExtremaKernels.hpp
/// @brief The kernels behind elementwise maximum/minimum and the axis-wise max/min reductions.
///
/// Shared by the eager entry points and by ExecutorBuilder, so an eager call and a replayed
/// node run the same code.

#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "StridedWalk.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief NumPy's ``maximum`` or ``minimum`` of two reals.
 *
 * A NaN in either operand is the result, so a NaN propagates through every fold built on this
 * (``numpy.max``, ``numpy.maximum.accumulate``). Equal operands are told apart by sign, so the
 * signed zeros are ordered: the maximum of +0 and -0 is +0 and the minimum is -0, in either
 * operand order. A plain ``a > b ? a : b`` gets both wrong: it drops a NaN in @p a and returns
 * whichever zero came second.
 *
 * @param[in] op Which of the two.
 * @param[in] a  The first operand, or a fold's accumulator.
 * @param[in] b  The second operand, or the next element.
 * @return The extremum.
 */
template <typename T>
[[nodiscard]] inline T extremum(MaxMinOp op, T a, T b) noexcept {
    if (std::isnan(a)) {
        return a;
    }
    if (std::isnan(b)) {
        return b;
    }
    if (a == b) {
        // Only the signed zeros compare equal and differ; the maximum takes the one without
        // the sign bit, the minimum the one with it.
        bool const a_negative = std::signbit(a);
        if (op == MaxMinOp::Max) {
            return a_negative ? b : a;
        }
        return a_negative ? a : b;
    }
    if (op == MaxMinOp::Max) {
        return a > b ? a : b;
    }
    return a < b ? a : b;
}

/// @brief ``C = maximum(A, B)`` or ``C = minimum(A, B)``, element by element.
/// @param[in]  op Which of the two.
/// @param[in]  A  The first operand.
/// @param[in]  B  The second operand, the same shape as @p A.
/// @param[out] C  The destination, the same shape. May be @p A or @p B with the same layout.
///
/// Each element is read before its own write and never again, so a destination that IS one of
/// its operands (same storage, same strides) is safe; the API layer rejects any other overlap.
template <typename T>
void run_elementwise_max_min(MaxMinOp op, ::einsums::detail::TensorImpl<T> const &A, ::einsums::detail::TensorImpl<T> const &B,
                             ::einsums::detail::TensorImpl<T> &C) {
    StridedGeometry const ga = geometry_of(A);
    StridedGeometry const gb = geometry_of(B);
    StridedGeometry const gc = geometry_of(C);
    T const              *a  = A.data();
    T const              *b  = B.data();
    T                    *c  = C.data();
    walk_elements<3>(gc.dims, {&ga.strides, &gb.strides, &gc.strides}, elementwise_fork_floor(),
                     [&](std::array<std::size_t, 3> const &off) { c[off[2]] = extremum(op, a[off[0]], b[off[1]]); });
}

/**
 * @brief Reduce @p A by maximum or minimum over every axis not in @p kept.
 *
 * @param[in]  op   Which of the two.
 * @param[in]  A    The tensor to reduce. Every reduced axis must have a nonzero extent; the API
 *                  layer refuses an empty reduction, which has no value to return.
 * @param[in]  kept The surviving axes of @p A, ascending; they are @p out 's axes, in order.
 * @param[out] out  The destination, overwritten.
 *
 * Each output element folds its reduced elements in one fixed order (the first reduced axis
 * fastest), on one thread, so the result does not depend on the team size. Threads divide the
 * output elements among themselves. Assigning from the first element rather than from a seed
 * keeps an all ``-inf`` maximum at ``-inf``, which a ``lowest()`` seed would not.
 */
template <typename T>
void run_reduce_axes(MaxMinOp op, ::einsums::detail::TensorImpl<T> const &A, std::vector<std::size_t> const &kept,
                     ::einsums::detail::TensorImpl<T> &out) {
    std::size_t const        rank = A.rank();
    std::vector<bool>        is_kept(rank, false);
    std::vector<std::size_t> kept_strides;
    std::vector<std::size_t> reduced_dims;
    std::vector<std::size_t> reduced_strides;
    for (std::size_t const axis : kept) {
        is_kept[axis] = true;
        kept_strides.push_back(A.stride(axis));
    }
    std::size_t reduced_count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (!is_kept[axis]) {
            reduced_dims.push_back(A.dim(axis));
            reduced_strides.push_back(A.stride(axis));
            reduced_count *= A.dim(axis);
        }
    }
    if (reduced_count == 0) {
        return; // refused at the API; never fold nothing into an uninitialized value
    }

    StridedGeometry const go = geometry_of(out);
    T const              *a  = A.data();
    T                    *o  = out.data();
    // Each output element is a whole fold, so the fork floor is in folded elements.
    std::size_t const floor = std::max<std::size_t>(1, elementwise_fork_floor() / reduced_count);
    walk_elements<2>(go.dims, {&kept_strides, &go.strides}, floor, [&](std::array<std::size_t, 2> const &off) {
        T    acc   = a[off[0]];
        bool first = true;
        walk_range<1>(reduced_dims, {&reduced_strides}, {off[0]}, 0, reduced_count, [&](std::array<std::size_t, 1> const &r) {
            if (first) {
                first = false;
                return;
            }
            acc = extremum(op, acc, a[r[0]]);
        });
        o[off[1]] = acc;
    });
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
