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

/// @brief Apply @p f to the matching elements of operands that share @p dims but not their strides.
///
/// The fallback for elementwise operations whose operands are laid out differently, such as a dense
/// output beside a strided view. @p f receives one element of each operand, in the order the operands
/// are passed, and the first operand is taken to be the one written.
///
/// The axes are walked in the order of the first operand's strides, smallest innermost, so the written
/// operand is visited along its own fastest axis whatever the index order. Axes of extent 1 are dropped,
/// and neighbouring axes that are contiguous in EVERY operand are merged, so each innermost run is as
/// long as all the layouts allow. A run where every operand steps by one element is a plain unit-stride
/// loop the compiler can vectorize; any other run steps each operand by its own stride.
///
/// Every element is visited exactly once and @p f sees the same values a visit in index order would,
/// so a purely elementwise @p f gives the same result. Operands that overlap in memory are the caller's
/// problem, as they are for any order of visit.
template <typename Dims, typename F, typename... Ops>
void strided_for_each(Dims const &dims, F &&f, Ops const &...ops) {
    constexpr std::size_t N = sizeof...(Ops);
    static_assert(N >= 1, "strided_for_each needs at least one operand");

    struct Axis {
        std::size_t                extent;
        std::array<std::size_t, N> stride;
    };

    std::vector<Axis> axes;
    axes.reserve(dims.size());
    for (std::size_t d = 0; d < dims.size(); ++d) {
        if (dims[d] == 0) {
            return;
        }
        if (dims[d] > 1) {
            axes.push_back(Axis{static_cast<std::size_t>(dims[d]), {static_cast<std::size_t>((*ops.strides)[d])...}});
        }
    }

    auto const pointers = std::make_tuple(ops.data...);

    // One run: @p n elements from @p offset, each operand stepping by its own @p inc.
    auto const run = [&]<std::size_t... K>(std::index_sequence<K...>, std::array<std::size_t, N> const &offset, std::size_t n,
                                           std::array<std::size_t, N> const &inc) {
        auto const base = std::make_tuple((std::get<K>(pointers) + offset[K])...);
        if (((inc[K] == 1) && ...)) {
            for (std::size_t i = 0; i < n; ++i) {
                f(std::get<K>(base)[i]...);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                f(std::get<K>(base)[i * inc[K]]...);
            }
        }
    };
    constexpr auto seq = std::make_index_sequence<N>{};

    std::array<std::size_t, N> offset{};
    if (axes.empty()) {
        run(seq, offset, 1, std::array<std::size_t, N>{});
        return;
    }

    std::stable_sort(axes.begin(), axes.end(), [](Axis const &a, Axis const &b) { return a.stride[0] < b.stride[0]; });

    // Merge an axis into the one inside it when it continues every operand's walk without a gap.
    std::vector<Axis> merged{axes.front()};
    for (std::size_t k = 1; k < axes.size(); ++k) {
        Axis &inner = merged.back();
        bool  tiles = true;
        for (std::size_t n = 0; n < N; ++n) {
            tiles = tiles && axes[k].stride[n] == inner.stride[n] * inner.extent;
        }
        if (tiles) {
            inner.extent *= axes[k].extent;
        } else {
            merged.push_back(axes[k]);
        }
    }

    // The innermost axis is the run; the rest are an odometer, fastest first.
    Axis const               inner = merged.front();
    std::vector<std::size_t> index(merged.size(), 0);
    while (true) {
        run(seq, offset, inner.extent, inner.stride);
        std::size_t k = 1;
        for (; k < merged.size(); ++k) {
            if (++index[k] < merged[k].extent) {
                for (std::size_t n = 0; n < N; ++n) {
                    offset[n] += merged[k].stride[n];
                }
                break;
            }
            for (std::size_t n = 0; n < N; ++n) {
                offset[n] -= (merged[k].extent - 1) * merged[k].stride[n];
            }
            index[k] = 0;
        }
        if (k == merged.size()) {
            return;
        }
    }
}

EINSUMS_NAMESPACE_END(detail)
