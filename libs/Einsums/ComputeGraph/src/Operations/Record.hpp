//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// The node-recording helpers the src/Operations translation units share. Private to them: each
// records an OpKind::Custom node whose executor reaches its operands through the graph's slots on
// every run, so the node follows rebind() and the memory planner moving a tensor's storage.

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/CommonTensorTypes.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cstddef>
#include <omp.h>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

/// Record a one-in one-out node calling ``apply(dst, src)`` on the operands' live impls, the way the
/// eager entry calls it. @p reads_dst lists the destination as an input too, for an op that reads
/// it before writing.
template <typename TD, typename TS, typename Fn>
void record_unary(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef dst, SlotRef src, Fn apply,
                  bool reads_dst = false) {
    OperandAccessor const d_access(dst.second, packed_gemm::get_scalar_type<TD>());
    OperandAccessor const s_access(src.second, packed_gemm::get_scalar_type<TS>());
    auto                  executor = [d_access, s_access, apply, execute_label]() {
        LabeledSection(execute_label);
        apply(*d_access.impl<TD>(), *s_access.impl<TS>());
    };
    std::vector<TensorId> inputs{src.first};
    if (reads_dst) {
        inputs.push_back(dst.first);
    }
    ctx.record(OpKind::Custom, name, std::move(inputs), {dst.first}, std::move(executor));
}

/// Record a one-in one-out node writing ``reduce(src)`` into the first element of @p r.
template <typename TR, typename TA, typename Fn>
void record_reduction(CaptureContext &ctx, char const *name, char const *execute_label, SlotRef r, SlotRef a, Fn reduce) {
    OperandAccessor const r_access(r.second, packed_gemm::get_scalar_type<TR>());
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<TA>());
    auto                  executor = [r_access, a_access, reduce, execute_label]() {
        LabeledSection(execute_label);
        r_access.impl<TR>()->data()[0] = reduce(*a_access.impl<TA>());
    };
    ctx.record(OpKind::Custom, name, {a.first}, {r.first}, std::move(executor));
}

/// An accessor per slot, all of element type @p T.
template <typename T>
std::vector<OperandAccessor> accessors(std::vector<SlotRef> const &refs) {
    std::vector<OperandAccessor> out;
    out.reserve(refs.size());
    for (SlotRef const &r : refs) {
        out.emplace_back(r.second, packed_gemm::get_scalar_type<T>());
    }
    return out;
}

/// Visit every multi-index of an index space of extents @p dims, calling ``visit(idx)`` once each:
/// axis 0 fastest, or the last axis fastest when @p LastFastest. Nothing is visited when any
/// extent is zero. Each caller turns the index into offsets through its own operands' strides, which
/// is what makes a strided view come out right.
template <bool LastFastest = false, typename Visit>
void for_each_index(std::vector<std::size_t> const &dims, Visit &&visit) {
    std::size_t total = 1;
    for (std::size_t const d : dims) {
        total *= d;
    }
    std::vector<std::size_t> idx(dims.size(), 0);
    for (std::size_t count = 0; count < total; ++count) {
        visit(std::as_const(idx));
        for (std::size_t n = 0; n < dims.size(); ++n) {
            std::size_t const k = LastFastest ? dims.size() - 1 - n : n;
            if (++idx[k] < dims[k]) {
                break;
            }
            idx[k] = 0;
        }
    }
}

/// The byte-free element offset of multi-index @p idx under @p strides.
inline std::size_t offset_of(std::vector<std::size_t> const &idx, std::vector<std::size_t> const &strides) {
    std::size_t off = 0;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        off += idx[k] * strides[k];
    }
    return off;
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
//
// @p parallel opts the outer walk into an OpenMP team, and it is the CALLER's
// assertion that concurrent @p op calls never touch the same element. gather
// may assert it unconditionally - its writes walk the linear side, disjoint by
// construction even when the index lists repeat. scatter and scatter_add may
// not: their writes go through the index lists, and a repeated index is two
// ops racing on one element. The team is skipped inside an existing parallel
// region (an OpenMP-executor graph replay keeps its across-node parallelism
// and each node stays serial inside) and below a size floor, where the fork
// costs more than the copy.
template <typename Fn>
void for_each_selection_run(std::vector<std::vector<std::size_t>> const &indices, std::vector<std::size_t> const &extents,
                            std::vector<std::size_t> const &idx_str, std::vector<std::size_t> const &lin_str, Fn &&op,
                            bool parallel = false) {
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

    auto visit = [&](std::size_t count) {
        // De-linearize with axis 1 fastest, the same tuple order the serial
        // odometer below produces; for the parallel walk only the SET of
        // tuples matters, but sharing the mapping keeps the two paths one
        // shape.
        std::size_t i_off = 0, l_off = 0, rem = count;
        for (std::size_t k = 1; k < N; ++k) {
            std::size_t const ik = rem % extents[k];
            rem /= extents[k];
            i_off += off[k][ik];
            l_off += ik * lin_str[k];
        }
        if (run0) {
            op(i_off + off[0][0], l_off, extents[0]);
        } else {
            for (std::size_t i = 0; i < extents[0]; ++i) {
                op(i_off + off[0][i], l_off + i * lin_str[0], std::size_t{1});
            }
        }
    };

    // The floor keeps tiny selections off the team: below it the fork/join
    // costs more than the move. 32k elements is ~256 KiB of doubles, well past
    // that crossover on anything current.
    constexpr std::size_t parallel_floor = 32768;
    if (parallel && outer > 1 && outer * extents[0] >= parallel_floor && omp_get_max_threads() > 1 && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (std::int64_t count = 0; count < static_cast<std::int64_t>(outer); ++count) {
            visit(static_cast<std::size_t>(count));
        }
        return;
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

EINSUMS_NAMESPACE_END(compute_graph::detail)
