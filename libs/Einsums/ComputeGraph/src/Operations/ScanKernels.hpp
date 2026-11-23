//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file ScanKernels.hpp
/// @brief The kernel behind prefix scans along one axis.
///
/// Shared by the eager entry point and by ExecutorBuilder, so an eager call and a replayed node
/// run the same code.

#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "ExtremaKernels.hpp"
#include "StridedWalk.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Call @p body with the binary operator and identity of @p op for element type @p T.
 *
 * Dispatched once per kernel call, so a line's inner loop holds one operator the compiler can see.
 * The max and min arms exist only for real @p T; the API refuses them for complex before a
 * kernel is ever reached.
 */
template <typename T, typename Body>
void with_scan_operator(ScanOp op, Body &&body) {
    switch (op) {
    case ScanOp::Sum:
        body([](T acc, T x) { return acc + x; }, T{0});
        return;
    case ScanOp::Prod:
        body([](T acc, T x) { return acc * x; }, T{1});
        return;
    case ScanOp::Max:
    case ScanOp::Min:
        if constexpr (!IsComplexV<T>) {
            MaxMinOp const extremum_op = op == ScanOp::Max ? MaxMinOp::Max : MaxMinOp::Min;
            T const        identity    = op == ScanOp::Max ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
            body([extremum_op](T acc, T x) { return extremum(extremum_op, acc, x); }, identity);
        }
        return;
    }
}

/**
 * @brief Scan @p A along the descriptor's axis into @p out.
 *
 * @param[in]  desc The operator, axis and flags.
 * @param[in]  A    The tensor to scan.
 * @param[out] out  The result, the same shape as @p A; may be @p A itself with the same layout.
 *
 * Every index tuple over the other axes is one independent line, walked through each tensor's
 * strides, so a view, slice or transpose scans as an owning tensor does. Each line is folded
 * sequentially, on one thread, in the order the flags give; threads divide the lines among
 * themselves. The result of a sum or product therefore depends on the data alone, never on the
 * team size.
 *
 * An inclusive line starts from its first element rather than from the identity, as NumPy's
 * cumsum does, so a leading -0 stays -0. In either form each element is read before its own
 * write, which is what makes the in-place scan exact.
 */
template <typename T>
void run_scan(ScanDescriptor const &desc, ::einsums::detail::TensorImpl<T> const &A, ::einsums::detail::TensorImpl<T> &out) {
    std::size_t const rank = A.rank();
    std::size_t const n    = A.dim(desc.axis);
    if (n == 0 || A.size() == 0) {
        return;
    }
    std::vector<std::size_t> line_dims;
    std::vector<std::size_t> a_strides;
    std::vector<std::size_t> o_strides;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (axis != desc.axis) {
            line_dims.push_back(A.dim(axis));
            a_strides.push_back(A.stride(axis));
            o_strides.push_back(out.stride(axis));
        }
    }
    std::size_t const sa      = A.stride(desc.axis);
    std::size_t const so      = out.stride(desc.axis);
    T const          *a_data  = A.data();
    T                *o_data  = out.data();
    bool const        reverse = desc.reverse;
    bool const        exclude = desc.exclusive;
    // Each line is a whole fold, so the fork floor is in folded elements.
    std::size_t const floor = std::max<std::size_t>(1, elementwise_fork_floor() / n);

    with_scan_operator<T>(desc.op, [&](auto combine, T identity) {
        walk_elements<2>(line_dims, {&a_strides, &o_strides}, floor, [&](std::array<std::size_t, 2> const &off) {
            T const *a   = a_data + off[0];
            T       *o   = o_data + off[1];
            T        acc = identity;
            for (std::size_t step = 0; step < n; ++step) {
                std::size_t const k = reverse ? n - 1 - step : step;
                T const           x = a[k * sa];
                if (exclude) {
                    o[k * so] = acc;
                    acc       = combine(acc, x);
                } else {
                    acc       = step == 0 ? x : combine(acc, x);
                    o[k * so] = acc;
                }
            }
        });
    });
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
