//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file StridedWalk.hpp
/// @brief A rank-erased walk over the elements of several same-shaped operands, by stride.
///
/// The kernels behind the max/min, comparison, select and scan nodes all visit a logical index
/// space and touch each operand at its own strides, so a transposed or sliced view is handled
/// exactly like an owning tensor. This is that walk, written once.

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// The extents and strides (in elements) of one operand.
struct StridedGeometry {
    std::vector<std::size_t> dims;    ///< Extent of each axis.
    std::vector<std::size_t> strides; ///< Stride of each axis, in elements.
};

/// @brief The geometry of @p t.
/// @param[in] t The operand.
/// @return Its extents and strides.
template <typename T>
[[nodiscard]] StridedGeometry geometry_of(::einsums::detail::TensorImpl<T> const &t) {
    StridedGeometry g;
    g.dims.resize(t.rank());
    g.strides.resize(t.rank());
    for (std::size_t a = 0; a < t.rank(); ++a) {
        g.dims[a]    = t.dim(a);
        g.strides[a] = t.stride(a);
    }
    return g;
}

/// @brief Whether a walk over @p items independent units of work should fork a team.
/// @param[in] items How many units the walk divides among threads.
/// @param[in] floor The smallest count worth a fork.
/// @return True when there are threads to use, no enclosing team, and enough work.
[[nodiscard]] inline bool walk_wants_threads(std::size_t items, std::size_t floor) {
#ifdef _OPENMP
    return items >= floor && omp_get_max_threads() > 1 && omp_in_parallel() == 0;
#else
    (void)items;
    (void)floor;
    return false;
#endif
}

/**
 * @brief Visit the logical indices ``[begin, end)`` of @p dims, first axis fastest.
 *
 * @param[in] dims    The index space.
 * @param[in] strides One stride list per operand, each as long as @p dims.
 * @param[in] base    Each operand's starting offset, added to every offset passed on.
 * @param[in] begin   The first linear index to visit.
 * @param[in] end     One past the last.
 * @param[in] body    Called as ``body(offsets)`` with one offset per operand.
 *
 * The first axis runs fastest because a column-major tensor, which is what Einsums builds, then
 * walks its storage in order. The multi-index is decomposed once at @p begin and then carried as
 * an odometer, so a chunk costs O(rank) per element only on a carry.
 */
template <std::size_t N, typename Body>
void walk_range(std::vector<std::size_t> const &dims, std::array<std::vector<std::size_t> const *, N> const &strides,
                std::array<std::size_t, N> const &base, std::size_t begin, std::size_t end, Body &&body) {
    if (begin >= end) {
        return;
    }
    std::size_t const          rank = dims.size();
    std::vector<std::size_t>   idx(rank, 0);
    std::array<std::size_t, N> offsets = base;
    std::size_t                rest    = begin;
    for (std::size_t a = 0; a < rank; ++a) {
        idx[a] = rest % dims[a];
        rest /= dims[a];
        for (std::size_t op = 0; op < N; ++op) {
            offsets[op] += idx[a] * (*strides[op])[a];
        }
    }
    for (std::size_t item = begin; item < end; ++item) {
        body(offsets);
        for (std::size_t a = 0; a < rank; ++a) {
            ++idx[a];
            for (std::size_t op = 0; op < N; ++op) {
                offsets[op] += (*strides[op])[a];
            }
            if (idx[a] < dims[a]) {
                break;
            }
            // Unsigned wrap-around is exact here: the subtraction undoes exactly what the
            // increments along this axis added.
            for (std::size_t op = 0; op < N; ++op) {
                offsets[op] -= dims[a] * (*strides[op])[a];
            }
            idx[a] = 0;
        }
    }
}

/**
 * @brief Visit every logical index of @p dims once, in parallel when it is worth a fork.
 *
 * @param[in] dims    The index space. A zero extent makes the walk empty.
 * @param[in] strides One stride list per operand.
 * @param[in] floor   The smallest element count worth a fork.
 * @param[in] body    Called as ``body(offsets)``; must be safe to call concurrently for
 *                    distinct indices, and must not throw (an exception cannot leave an
 *                    OpenMP region).
 *
 * Each thread takes one contiguous chunk of the linear index range, so which thread visits
 * which element depends on the team size; a body whose result depends on visiting order
 * across elements does not belong here.
 */
template <std::size_t N, typename Body>
void walk_elements(std::vector<std::size_t> const &dims, std::array<std::vector<std::size_t> const *, N> const &strides, std::size_t floor,
                   Body &&body) {
    std::size_t total = 1;
    for (std::size_t const d : dims) {
        total *= d;
    }
    if (total == 0) {
        return;
    }
    std::array<std::size_t, N> const base{};
#ifdef _OPENMP
    if (walk_wants_threads(total, floor)) {
        EINSUMS_OMP_PARALLEL {
            auto const        team  = static_cast<std::size_t>(omp_get_num_threads());
            auto const        me    = static_cast<std::size_t>(omp_get_thread_num());
            std::size_t const chunk = (total + team - 1) / team;
            std::size_t const begin = std::min(total, me * chunk);
            std::size_t const end   = std::min(total, begin + chunk);
            walk_range<N>(dims, strides, base, begin, end, body);
        }
        return;
    }
#endif
    walk_range<N>(dims, strides, base, 0, total, body);
}

/// @brief The element-count floor the element-wise kernels fork at.
/// @return The library's bandwidth-bound threshold.
[[nodiscard]] inline std::size_t elementwise_fork_floor() {
    return ::einsums::hardware::omp_min_parallel_elements();
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
