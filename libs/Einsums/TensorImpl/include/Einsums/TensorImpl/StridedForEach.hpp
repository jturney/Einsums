//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(detail)

/// @brief One operand of @ref strided_for_each: its first element and its strides, in elements.
template <typename T, typename Strides>
struct StridedOperand {
    T             *data;
    Strides const *strides;
};

/// @brief The operand @ref strided_for_each walks for a tensor implementation (or anything with @c data and @c strides).
template <typename Tensor>
auto strided(Tensor &t) {
    return StridedOperand<std::remove_reference_t<decltype(*t.data())>, std::remove_cvref_t<decltype(t.strides())>>{
        .data = t.data(), .strides = &t.strides()};
}

/// @brief One operand's part of a run: its first element and its step between elements, in elements.
template <typename T>
struct RunOperand {
    T          *data;
    std::size_t inc;
};

/// @brief Hand @p run each maximal stretch of elements that every operand walks with one fixed step.
///
/// The common walk under every elementwise operation and reduction. @p run receives a stretch's length
/// and one @ref RunOperand per operand, in the order the operands are passed: run(n, RunOperand...).
///
/// The axes are ordered by the first operand's strides, smallest innermost, so the first operand, the
/// written one where there is one, is walked along its own fastest axis whatever the index order. Axes
/// of extent 1 are dropped, and neighbouring axes that are contiguous in EVERY operand are merged, so
/// operands that are each one vector in the same order make a single run, and each run is as long as
/// all the layouts allow. An empty extent makes no runs. A tensor with no axis longer than one makes one
/// run of one element, with every step 1, which every vector routine accepts: BLAS reads a step of 0 as
/// "do nothing" or "repeat one element".
///
/// Runs are visited one after another on the calling thread, in the order of the first operand's
/// memory; @p run may parallelize within a run.
template <typename Dims, typename F, typename... Ops>
void strided_for_each_run(Dims const &dims, F &&run, Ops const &...ops) {
    constexpr std::size_t N = sizeof...(Ops);
    static_assert(N >= 1, "strided_for_each_run needs at least one operand");

    struct Axis {
        std::size_t                extent;
        std::array<std::size_t, N> stride;
    };

    // Ranks this small keep their axes on the stack: elementwise operations on small tensors are
    // called often enough that an allocation per call shows.
    constexpr std::size_t         inline_rank = 8;
    std::array<Axis, inline_rank> inline_axes;
    std::vector<Axis>             heap_axes;
    Axis                         *axes  = inline_axes.data();
    std::size_t                   count = 0;
    if (dims.size() > inline_rank) {
        heap_axes.resize(dims.size());
        axes = heap_axes.data();
    }
    for (std::size_t d = 0; d < dims.size(); ++d) {
        if (dims[d] == 0) {
            return;
        }
        if (dims[d] > 1) {
            axes[count++] = Axis{static_cast<std::size_t>(dims[d]), {static_cast<std::size_t>((*ops.strides)[d])...}};
        }
    }

    auto const call = [&]<std::size_t... K>(std::index_sequence<K...>, std::array<std::size_t, N> const &offset, std::size_t n,
                                            std::array<std::size_t, N> const &inc) {
        run(n, RunOperand<std::remove_pointer_t<decltype(ops.data)>>{ops.data + offset[K], inc[K]}...);
    };
    constexpr auto seq = std::make_index_sequence<N>{};

    std::array<std::size_t, N> offset{};
    if (count == 0) {
        std::array<std::size_t, N> unit;
        unit.fill(1);
        call(seq, offset, 1, unit);
        return;
    }

    std::stable_sort(axes, axes + count, [](Axis const &a, Axis const &b) { return a.stride[0] < b.stride[0]; });

    // Merge an axis into the one inside it when it continues every operand's walk without a gap.
    std::size_t merged = 1;
    for (std::size_t k = 1; k < count; ++k) {
        Axis &inner = axes[merged - 1];
        bool  tiles = true;
        for (std::size_t n = 0; n < N; ++n) {
            tiles = tiles && axes[k].stride[n] == inner.stride[n] * inner.extent;
        }
        if (tiles) {
            inner.extent *= axes[k].extent;
        } else {
            axes[merged++] = axes[k];
        }
    }

    // The innermost axis is the run; the rest are an odometer, fastest first.
    Axis const                           inner = axes[0];
    std::array<std::size_t, inline_rank> inline_index{};
    std::vector<std::size_t>             heap_index;
    std::size_t                         *index = inline_index.data();
    if (merged > inline_rank) {
        heap_index.assign(merged, 0);
        index = heap_index.data();
    }
    while (true) {
        call(seq, offset, inner.extent, inner.stride);
        std::size_t k = 1;
        for (; k < merged; ++k) {
            if (++index[k] < axes[k].extent) {
                for (std::size_t n = 0; n < N; ++n) {
                    offset[n] += axes[k].stride[n];
                }
                break;
            }
            for (std::size_t n = 0; n < N; ++n) {
                offset[n] -= (axes[k].extent - 1) * axes[k].stride[n];
            }
            index[k] = 0;
        }
        if (k == merged) {
            return;
        }
    }
}

/// @brief Apply @p f to the matching elements of operands that share @p dims but not their strides.
///
/// The fallback for elementwise operations whose operands are laid out differently, such as a dense
/// output beside a strided view. @p f receives one element of each operand, in the order the operands
/// are passed, and the first operand is taken to be the one written. The walk is
/// @ref strided_for_each_run's: the written operand along its own fastest axis, in runs as long as all
/// the layouts allow. A run where every operand steps by one element is a plain unit-stride loop the
/// compiler can vectorize; any other run steps each operand by its own stride.
///
/// Every element is visited exactly once and @p f sees the same values a visit in index order would,
/// so a purely elementwise @p f gives the same result. Operands that overlap in memory are the caller's
/// problem, as they are for any order of visit.
template <typename Dims, typename F, typename... Ops>
void strided_for_each(Dims const &dims, F &&f, Ops const &...ops) {
    strided_for_each_run(
        dims,
        [&](std::size_t n, auto const &...r) {
            if (((r.inc == 1) && ...)) {
                for (std::size_t i = 0; i < n; ++i) {
                    f(r.data[i]...);
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    f(r.data[i * r.inc]...);
                }
            }
        },
        ops...);
}

EINSUMS_NAMESPACE_END(detail)
