//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <cstddef>

namespace einsums {

/**
 * @brief Cost, in nanoseconds, of entering and leaving an OpenMP parallel region.
 *
 * Measured once on first use: an empty region at the default thread count, warm
 * team, best of several trials. Zero without OpenMP or on a single thread.
 *
 * This is not a small number. On a 10-thread machine an EMPTY region costs around
 * 20 microseconds, and it grows with the thread count, so a loop has to have real
 * work in it before parallelizing is anything but a loss.
 */
EINSUMS_EXPORT double omp_region_cost_ns();

/**
 * @brief Elements below which an elementwise loop should not be parallelized.
 *
 * Derived from @ref omp_region_cost_ns rather than hardcoded, because the region
 * cost varies by an order of magnitude across thread counts and OpenMP runtimes.
 * Elementwise kernels are bandwidth-bound, so the conversion assumes a
 * deliberately conservative one element per nanosecond: the break-even is where
 * the loop's own time matches the cost of entering the region.
 *
 * The motivating case is a tiled tensor: an amplitude update over a 64-element
 * tile was forking a team to divide 64 numbers, and a graph replay does thousands
 * of those per iteration.
 */
EINSUMS_EXPORT std::size_t omp_min_parallel_elements();

} // namespace einsums

/**
 * @def EINSUMS_OMP_PARALLEL_FOR_SIMD_IF
 *
 * Parallelize and vectorize the following loop only when @p cond holds.
 *
 * Use this instead of @ref EINSUMS_OMP_PARALLEL_FOR_SIMD on any loop whose trip
 * count can be small, passing something like
 * ``elems >= einsums::omp_min_parallel_elements()``. An unconditional region on a
 * short loop costs far more than the loop.
 *
 * @versionadded{2.0.0}
 */
#define EINSUMS_OMP_PARALLEL_FOR_SIMD_IF(cond) EINSUMS_OMP_SIMD_PRAGMA(parallel for if (cond))

/**
 * @def EINSUMS_OMP_PARALLEL_FOR_IF
 *
 * Parallelize the following loop only when @p cond holds. See
 * @ref EINSUMS_OMP_PARALLEL_FOR_SIMD_IF.
 *
 * @versionadded{2.0.0}
 */
#define EINSUMS_OMP_PARALLEL_FOR_IF(cond) EINSUMS_OMP_PRAGMA(parallel for if (cond))
