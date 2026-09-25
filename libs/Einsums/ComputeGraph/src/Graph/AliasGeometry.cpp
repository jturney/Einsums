//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AliasGeometry.cpp
/// @brief The pointer-derived half of alias geometry.
///
/// Every function here reads addresses and strides, so every one of them needs
/// tensors that have been allocated. The structural derivation, which needs
/// none, is a class and so lives whole in the header beside these declarations,
/// which is also where each of these functions is documented.
///
/// The two box derivations below are tried in order and neither subsumes the
/// other. The matched one decodes a byte offset into per-axis indices and is
/// exact when the child's layout agrees with the parent's; the span one
/// brackets the whole access and is looser, but it answers for a reshaped view
/// that the first refuses. @ref derive_alias_box is the pair of them, plus the
/// normalization that turns a full-cover box back into the whole tensor.

#include "AliasGeometry.hpp"

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph::alias_geometry)

bool handle_byte_span(TensorHandle const &h, char const *&lo, char const *&hi) {
    if (h.is_tiled) {
        return false;
    }
    return detail::strided_byte_span(h.data_ptr, h.dims, h.strides, h.element_size, lo, hi);
}

LayoutFit layout_axis_order(TensorHandle const &h, std::vector<size_t> &order, size_t &total) {
    size_t const rank = h.dims.size();
    if (rank == 0 || h.strides.size() != rank || h.is_tiled) {
        return LayoutFit::None;
    }
    order.clear();
    total = 1;
    for (size_t d = 0; d < rank; ++d) {
        if (h.dims[d] == 0) {
            return LayoutFit::None;
        }
        total *= h.dims[d];
        if (h.dims[d] > 1) {
            order.push_back(d);
        }
    }
    std::ranges::sort(order, [&](size_t a, size_t b) { return h.strides[a] < h.strides[b]; });
    size_t reach  = 1; // one past the last offset the axes so far can reach
    bool   packed = true;
    for (size_t const d : order) {
        if (h.strides[d] < reach) {
            return LayoutFit::None;
        }
        packed = packed && h.strides[d] == reach;
        reach  = h.strides[d] * h.dims[d];
    }
    std::ranges::reverse(order); // the digit peel runs most significant first
    return packed ? LayoutFit::Packed : LayoutFit::Nested;
}

namespace {

/// Recover @p child's region in @p parent's axis space by matching each child
/// axis to the parent axis that carries the same stride. Exact when it answers,
/// and it answers for every view whose axes are parent axes: a sub-block, a
/// dropped axis (which is a single-index interval in the parent), a transposed
/// view, and any mix of the three.
///
/// Returns false whenever the layout is not provably a sub-box, which leaves the
/// caller to try the looser derivation below or, failing that, to treat the
/// access as whole-tensor. Being wrong here would UNDER-serialize, so every
/// ambiguity declines rather than guesses.
bool derive_matched_alias_box(TensorHandle const &parent, TensorHandle const &child,
                              std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    size_t const rank = parent.dims.size();
    if (rank == 0 || parent.element_size == 0 || child.element_size != parent.element_size) {
        return false;
    }
    if (parent.strides.size() != rank || child.strides.size() != child.dims.size() || parent.is_tiled || child.is_tiled) {
        return false;
    }
    // A layout that is not nested cannot be decoded: the offset would name more
    // than one index, and the axis matching would be ambiguous with it. Equal
    // strides on two traversed axes are the common way in and are rejected
    // here, along with the overlapping strides that are the subtle way in.
    std::vector<size_t> order;
    size_t              total = 0;
    if (layout_axis_order(parent, order, total) == LayoutFit::None) {
        return false;
    }

    auto const *p = static_cast<char const *>(parent.data_ptr);
    auto const *c = static_cast<char const *>(child.data_ptr);
    if (c < p) {
        return false;
    }
    size_t off = static_cast<size_t>(c - p);
    if (off % parent.element_size != 0) {
        return false;
    }
    off /= parent.element_size;

    // Peel the offset apart largest stride first, which is unique for a nested
    // layout. An offset that is not on the lattice cannot survive it: it leaves
    // a non-zero remainder, or a digit past its axis's extent that the range
    // check at the end rejects.
    std::vector<std::int64_t> start(rank, 0);
    for (size_t const d : order) {
        start[d] = static_cast<std::int64_t>(off / parent.strides[d]);
        off %= parent.strides[d];
    }
    if (off != 0) {
        return false; // offset does not land on a parent index
    }

    // A child axis that reuses a parent stride keeps that axis; the rest are
    // pinned to a single index by the offset.
    std::vector<std::int64_t> extent(rank, 1);
    std::vector<bool>         matched(child.dims.size(), false);
    for (size_t d = 0; d < rank; ++d) {
        for (size_t e = 0; e < child.dims.size(); ++e) {
            if (!matched[e] && child.strides[e] == parent.strides[d] && child.dims[e] > 1) {
                extent[d]  = static_cast<std::int64_t>(child.dims[e]);
                matched[e] = true;
                break;
            }
        }
    }
    for (size_t e = 0; e < child.dims.size(); ++e) {
        if (!matched[e] && child.dims[e] > 1) {
            return false; // a traversed child axis with no parent counterpart
        }
    }

    box.clear();
    box.reserve(rank);
    for (size_t d = 0; d < rank; ++d) {
        std::int64_t const hi = start[d] + extent[d];
        if (start[d] < 0 || hi > static_cast<std::int64_t>(parent.dims[d])) {
            return false;
        }
        box.emplace_back(start[d], hi);
    }
    return true;
}

/// Recover @p child's region in @p parent's axis space from the contiguous
/// OFFSET RANGE it spans, for the children the axis matching above cannot
/// describe: a reshaped window carries axes that are products of parent axes
/// rather than parent axes, so there is no correspondence to recover.
///
/// The shape comes from the DLPNO port, whose rank-3 dressed factors are
/// reshaped windows of a flat scratch pool. It buys that port no schedule
/// today, and the reason is worth recording: its hand-outs are PREFIXES of
/// their buffer, so two of them overlap however precisely they are described,
/// and the parallelism there comes from the pool being several buffers rather
/// than from any box. What the bound buys is that a pool carved into DISJOINT
/// windows is separable at all, which the axis match cannot do at any width.
///
/// **Soundness.** Strides are unsigned, so every element the child addresses
/// lies at an offset in `[base, base + sum (dim - 1) * stride]`, whatever its
/// axis order and however its own axes overlap each other. A packed parent
/// decodes each offset in that range to exactly one index, so the smallest box
/// containing the decoded ends contains every element the child can touch. That
/// box is looser than the matched one - a range of offsets is not a box, so
/// once the two ends differ in some axis every lower axis has to open to its
/// full extent - and loose is the safe direction: a box that is too WIDE costs
/// hazard edges, only a box that is too NARROW loses one.
bool derive_span_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    if (child.is_tiled || parent.element_size == 0 || child.element_size != parent.element_size) {
        return false;
    }
    if (child.strides.size() != child.dims.size() || parent.data_ptr == nullptr || child.data_ptr == nullptr) {
        return false;
    }

    // Packed, not merely nested: a gapped parent has offsets between its
    // elements, and the bound below walks a RANGE of offsets rather than the
    // lattice points the matched derivation sticks to.
    std::vector<size_t> order;
    size_t              total = 0;
    if (layout_axis_order(parent, order, total) != LayoutFit::Packed) {
        return false;
    }

    auto const *p = static_cast<char const *>(parent.data_ptr);
    auto const *c = static_cast<char const *>(child.data_ptr);
    if (c < p) {
        return false;
    }
    size_t base = static_cast<size_t>(c - p);
    if (base % parent.element_size != 0) {
        return false;
    }
    base /= parent.element_size;

    size_t reach = 0; // offset of the last element the child can address
    for (size_t e = 0; e < child.dims.size(); ++e) {
        if (child.dims[e] == 0) {
            return false; // an empty access has no region to bound
        }
        reach += (child.dims[e] - 1) * child.strides[e];
    }
    if (base >= total || reach > total - 1 - base) {
        return false; // not contained in the parent, so not describable in its axes
    }

    box.clear();
    box.reserve(parent.dims.size());
    for (size_t const d : parent.dims) {
        box.emplace_back(0, static_cast<std::int64_t>(d));
    }
    // Peel both ends most significant first. While the digits agree the box is
    // that single index; the first axis where they differ takes the span
    // between them and every axis below it keeps the full extent it started
    // with, because the offsets between the two ends run through all of them.
    size_t lo = base;
    size_t hi = base + reach;
    for (size_t const d : order) {
        size_t const stride = parent.strides[d];
        size_t const lo_d   = lo / stride;
        size_t const hi_d   = hi / stride;
        lo %= stride;
        hi %= stride;
        if (hi_d >= parent.dims[d]) {
            return false; // cannot happen for a contained child of a packed parent
        }
        box[d] = {static_cast<std::int64_t>(lo_d), static_cast<std::int64_t>(hi_d) + 1};
        if (lo_d != hi_d) {
            break;
        }
    }
    return true;
}

} // namespace

bool derive_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    if (!derive_matched_alias_box(parent, child, box) && !derive_span_alias_box(parent, child, box)) {
        return false;
    }
    bool whole = box.size() == parent.dims.size();
    for (size_t d = 0; whole && d < box.size(); ++d) {
        whole = box[d].first == 0 && box[d].second == static_cast<std::int64_t>(parent.dims[d]);
    }
    if (whole) {
        box.clear();
        return false;
    }
    return true;
}

bool whole_cover(AliasBox const &box, std::vector<size_t> const &dims) {
    if (box.size() != dims.size()) {
        return false;
    }
    for (size_t d = 0; d < box.size(); ++d) {
        if (box[d].first != 0 || !std::cmp_equal(box[d].second, dims[d])) {
            return false;
        }
    }
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::alias_geometry)
