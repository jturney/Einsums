//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file MaskKernels.hpp
/// @brief The kernels behind comparisons into 1/0 masks and the select that reads them.
///
/// Shared by the eager entry points and by ExecutorBuilder, so an eager call and a replayed
/// node run the same code.

#include <Einsums/ComputeGraph/PredExpr.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <array>
#include <cstddef>
#include <functional>

#include "StridedWalk.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Call @p body with the standard-library comparison object @p op names.
 *
 * Dispatched once per kernel call rather than per element, so the walk's inner loop holds one
 * comparison the compiler can see. The function objects are the language's own operators, which
 * is what makes the IEEE behavior on NaN (false, except for ``!=``) hold without special cases.
 */
template <typename Body>
decltype(auto) with_comparison(CmpOp op, Body &&body) {
    switch (op) {
    case CmpOp::Gt:
        return body(std::greater<>{});
    case CmpOp::Ge:
        return body(std::greater_equal<>{});
    case CmpOp::Lt:
        return body(std::less<>{});
    case CmpOp::Le:
        return body(std::less_equal<>{});
    case CmpOp::Eq:
        return body(std::equal_to<>{});
    case CmpOp::Ne:
        return body(std::not_equal_to<>{});
    }
    return body(std::not_equal_to<>{}); // unreachable: every enumerator is handled above
}

/// @brief ``out = (A op B) ? 1 : 0``, element by element.
/// @param[in]  op  The comparison.
/// @param[in]  A   The left operand.
/// @param[in]  B   The right operand, the same shape.
/// @param[out] out The mask, the same shape; may be A or B with the same layout.
template <typename T>
void run_compare(CmpOp op, ::einsums::detail::TensorImpl<T> const &A, ::einsums::detail::TensorImpl<T> const &B,
                 ::einsums::detail::TensorImpl<T> &out) {
    StridedGeometry const ga = geometry_of(A);
    StridedGeometry const gb = geometry_of(B);
    StridedGeometry const go = geometry_of(out);
    T const              *a  = A.data();
    T const              *b  = B.data();
    T                    *o  = out.data();
    with_comparison(op, [&](auto cmp) {
        walk_elements<3>(go.dims, {&ga.strides, &gb.strides, &go.strides}, elementwise_fork_floor(),
                         [&](std::array<std::size_t, 3> const &off) { o[off[2]] = cmp(a[off[0]], b[off[1]]) ? T{1} : T{0}; });
    });
}

/// @brief ``out = (A op s) ? 1 : 0``, element by element.
/// @param[in]  op     The comparison.
/// @param[in]  A      The left operand.
/// @param[in]  scalar The right-hand side, converted to @p T once.
/// @param[out] out    The mask, the same shape; may be A with the same layout.
template <typename T>
void run_compare_scalar(CmpOp op, ::einsums::detail::TensorImpl<T> const &A, double scalar, ::einsums::detail::TensorImpl<T> &out) {
    StridedGeometry const ga = geometry_of(A);
    StridedGeometry const go = geometry_of(out);
    T const              *a  = A.data();
    T                    *o  = out.data();
    T const               s  = static_cast<T>(scalar);
    with_comparison(op, [&](auto cmp) {
        walk_elements<2>(go.dims, {&ga.strides, &go.strides}, elementwise_fork_floor(),
                         [&](std::array<std::size_t, 2> const &off) { o[off[1]] = cmp(a[off[0]], s) ? T{1} : T{0}; });
    });
}

/**
 * @brief ``out = (mask != 0) ? A : B``, element by element.
 *
 * A true select: the element not chosen is never part of the result, so an inf or NaN in the
 * branch not taken cannot leak through, as it would through ``mask*A + (1-mask)*B``
 * (``0 * inf`` is NaN). Any nonzero mask value is true, so a mask that went through arithmetic
 * still selects, and so is NaN, which compares unequal to zero, matching NumPy's truthiness.
 *
 * @param[in]  mask The selector.
 * @param[in]  A    Chosen where the mask is nonzero.
 * @param[in]  B    Chosen where it is zero.
 * @param[out] out  The result; may be any of the three with the same layout.
 */
template <typename T>
void run_select(::einsums::detail::TensorImpl<T> const &mask, ::einsums::detail::TensorImpl<T> const &A,
                ::einsums::detail::TensorImpl<T> const &B, ::einsums::detail::TensorImpl<T> &out) {
    StridedGeometry const gm = geometry_of(mask);
    StridedGeometry const ga = geometry_of(A);
    StridedGeometry const gb = geometry_of(B);
    StridedGeometry const go = geometry_of(out);
    T const              *m  = mask.data();
    T const              *a  = A.data();
    T const              *b  = B.data();
    T                    *o  = out.data();
    walk_elements<4>(go.dims, {&gm.strides, &ga.strides, &gb.strides, &go.strides}, elementwise_fork_floor(),
                     [&](std::array<std::size_t, 4> const &off) { o[off[3]] = m[off[0]] != T{0} ? a[off[1]] : b[off[2]]; });
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
