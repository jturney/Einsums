//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file OperandOverlap.hpp
/// @brief Whether two operands provably share elements, and whether they are the same layout.
///
/// One rule for every op that writes while it reads: the string einsum, and the element-wise,
/// select, reduction and scan nodes.

#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// The bytes an operand's elements span, and whether they fill that span without gaps.
struct ByteRegion {
    unsigned char const *lo{nullptr};  ///< The first byte of the first element.
    unsigned char const *hi{nullptr};  ///< One past the last byte of the last element.
    bool                 contiguous{}; ///< Whether every element in the span belongs to the operand.
    bool                 empty{};      ///< Whether the operand has no elements.
};

/// @brief The byte region of @p t.
/// @param[in] t A dense tensor or view.
/// @return Its span in bytes; empty when any extent is zero.
template <typename X>
[[nodiscard]] ByteRegion byte_region_of(X const &t) {
    auto const *lo     = reinterpret_cast<unsigned char const *>(t.data());
    std::size_t last   = 0;
    std::size_t nelems = 1;
    for (std::size_t d = 0; d < tensor_rank(t); d++) {
        if (t.dim(d) == 0) {
            return {.lo = lo, .hi = lo, .contiguous = true, .empty = true};
        }
        last += (t.dim(d) - 1) * t.stride(d);
        nelems *= t.dim(d);
    }
    return {.lo = lo, .hi = lo + (last + 1) * sizeof(*t.data()), .contiguous = last + 1 == nelems, .empty = false};
}

/**
 * @brief Whether @p x and @p y provably share an element.
 *
 * Interval overlap alone is not proof: disjoint column-major slices of one parent interleave in
 * memory without sharing an element. So only two shapes count as overlap: both regions are
 * contiguous and intersect, or both start at the same byte. Regions are compared in bytes, so
 * operands of different element types are handled alike.
 *
 * @param[in] x One operand.
 * @param[in] y The other.
 * @return True when they provably share storage.
 */
template <typename X, typename Y>
[[nodiscard]] bool provably_overlap(X const &x, Y const &y) {
    ByteRegion const a = byte_region_of(x);
    ByteRegion const b = byte_region_of(y);
    if (a.empty || b.empty) {
        return false;
    }
    if (!(a.lo < b.hi && b.lo < a.hi)) {
        return false;
    }
    return (a.contiguous && b.contiguous) || a.lo == b.lo;
}

/**
 * @brief Whether @p x and @p y are the same elements in the same order.
 *
 * The one aliasing an element-wise kernel can tolerate: each element is read immediately before
 * its own write and never again.
 *
 * @param[in] x One operand.
 * @param[in] y The other.
 * @return True when they start at the same byte with the same element size, rank, extents and
 *         strides.
 */
template <typename X, typename Y>
[[nodiscard]] bool same_layout(X const &x, Y const &y) {
    if (static_cast<void const *>(x.data()) != static_cast<void const *>(y.data()) || sizeof(*x.data()) != sizeof(*y.data()) ||
        tensor_rank(x) != tensor_rank(y)) {
        return false;
    }
    for (std::size_t d = 0; d < tensor_rank(x); d++) {
        if (x.dim(d) != y.dim(d) || x.stride(d) != y.stride(d)) {
            return false;
        }
    }
    return true;
}

/// @brief Whether writing @p out while reading @p in is unsafe for an element-wise kernel.
/// @param[in] out The destination.
/// @param[in] in  An operand.
/// @return True when they provably overlap without being the same layout.
template <typename Out, typename In>
[[nodiscard]] bool unsafe_elementwise_alias(Out const &out, In const &in) {
    return provably_overlap(out, in) && !same_layout(out, in);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
