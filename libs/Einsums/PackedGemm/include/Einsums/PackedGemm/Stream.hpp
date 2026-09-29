//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

/**
 * @brief Extents and element strides of one strided operand, one entry per axis.
 *
 * @versionadded{2.0.0}
 */
struct StreamLayout {
    std::vector<int64_t> dims;    ///< Extent of each axis.
    std::vector<int64_t> strides; ///< Stride of each axis, in elements.
};

/**
 * @brief One contraction fed by a stream over a large tensor @f$S@f$.
 *
 * The term computes
 * @f[
 *     C[\rho(\mathrm{idx})] \leftarrow c_{pf} C[\rho(\mathrm{idx})] + \alpha \sum S[\mathrm{idx}]\, W[\pi(\mathrm{idx})]
 * @f]
 * where every axis of @f$C@f$ and of @f$W@f$ carries the label of some axis of @f$S@f$. The maps are
 * given per axis of @f$S@f$: @ref c_axis names the axis of @f$C@f$ with the same label, or -1 when the
 * label is summed, and @ref w_axis does the same for @f$W@f$.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
struct StreamTerm {
    T               *c{nullptr}; ///< Output data. Terms with the same pointer accumulate into one output.
    StreamLayout     c_layout;   ///< Output geometry.
    T const         *w{nullptr}; ///< The small operand's data.
    StreamLayout     w_layout;   ///< The small operand's geometry.
    std::vector<int> c_axis;     ///< For each axis of S, the axis of C with the same label, or -1.
    std::vector<int> w_axis;     ///< For each axis of S, the axis of W with the same label, or -1.
    T                alpha{1};   ///< Scale on the contraction.
    T                c_pf{0};    ///< Scale on the output's prior contents; only the first term on an output may differ from 1.
};

/**
 * @brief Evaluate every term with one read of @f$S@f$, in storage order.
 *
 * A contraction in which one operand supplies every index and the other operands are small is
 * memory-bandwidth bound: its cost is one read of @f$S@f$. This walks @f$S@f$ once, fastest axis
 * innermost, and adds each element's contribution to every term's output, so @f$N@f$ terms over one
 * tensor cost one read of it instead of @f$N@f$.
 *
 * The team is sized to the work: a stream too short to pay for forking threads runs serially and
 * writes every output in place. A parallel walk accumulates outputs in per-thread private buffers and
 * reduces them at the end, unless @p partition_axes names axes of @f$S@f$ the walk may split between
 * threads instead: the kernel partitions one of them (preferring an axis with a block for every
 * thread, then the largest stride), and each term whose output carries that axis writes its own
 * disjoint slice directly. Every term whose output does not carry the chosen axis is still privatized,
 * so the caller must keep those outputs small enough for a copy per thread to stay in cache.
 *
 * Each output's @c c_pf is applied once, before any accumulation, even when @f$S@f$ is empty. The work
 * is split over the threads the parallel region actually receives, so a call from inside another
 * parallel region, where the region does not fork, runs correctly on one thread.
 *
 * @param s The streamed tensor's data.
 * @param s_layout Its geometry.
 * @param terms The contractions to evaluate. No output may overlap @f$S@f$, a small operand, or another output.
 * @param partition_axes Axes of @f$S@f$ the walk may partition for direct writes; empty to privatize every output.
 *
 * @throws std::invalid_argument If a term's axis maps do not match @f$S@f$'s rank, or a term after the
 * first on an output has a @c c_pf other than 1.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
EINSUMS_EXPORT void stream_contract(T const *s, StreamLayout const &s_layout, std::vector<StreamTerm<T>> const &terms,
                                    std::vector<int> const &partition_axes);

EINSUMS_NAMESPACE_END(packed_gemm)
