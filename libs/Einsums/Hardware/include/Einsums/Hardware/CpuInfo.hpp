//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(hardware)

/// Data-cache sizes in bytes. Zero-initialised fields mean "not detected"; the
/// accessors below always return usable values, falling back to conservative
/// defaults.
struct CacheSizes {
    std::int64_t l1{32 * 1024};
    std::int64_t l2{256 * 1024};
    std::int64_t l3{8 * 1024 * 1024};
};

/**
 * @brief Hardware facts detected once, shared by every module that needs them.
 *
 * This exists because the same facts used to be detected in more than one place
 * and the copies disagreed. PackedGemm sized its cache blocking from
 * ``hw.l2cachesize`` while ComputeGraph's cost model read
 * ``hw.perflevel0.l2cachesize``; on Apple Silicon the first reports the
 * EFFICIENCY cluster, so the blocking was derived from a quarter of the L2 the
 * cores running the work actually have. Detection belongs in one place.
 *
 * What lives here is hardware FACTS and the cheap thresholds derived directly
 * from them. Algorithm policy does not: PackedGemm's micro-kernel register
 * blocking and its BLIS cache blocking are tuning decisions that happen to read
 * these numbers, and they stay with the algorithm.
 */
struct CpuInfo {
    /// SIMD vector length in doubles: SSE/NEON = 2, AVX = 4, AVX-512 = 8.
    int simd_width_f64{2};

    CacheSizes cache;

    /// Cost in nanoseconds of entering and leaving an OpenMP parallel region at
    /// the default thread count. Zero without OpenMP or on a single thread.
    double omp_region_cost_ns{0.0};
};

/// Detected once on first use.
EINSUMS_EXPORT CpuInfo const &cpu_info();

/**
 * @brief Cost in nanoseconds of entering and leaving an OpenMP parallel region.
 *
 * Measured, not assumed: an empty region at the default thread count, warm team,
 * best of several trials. This is not a small number -- on a 10-thread machine an
 * EMPTY region costs around 20 microseconds, and it grows with the thread count,
 * so a loop needs real work in it before parallelizing is anything but a loss.
 *
 * Exposed to Python because cost models are not all in C++: the DLPNO example
 * chooses how finely to batch its per-pair work, and that decision turns on
 * exactly this number.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT double omp_region_cost_ns();

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
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::size_t omp_min_parallel_elements();

/**
 * @brief Work, in flops, below which parallelizing a contraction is a net loss.
 *
 * Same derivation as @ref omp_min_parallel_elements, converted at the ~1 GFLOP/s
 * that SMALL contractions actually achieve rather than at peak. Using peak would
 * put the break-even far too high and exclude shapes that genuinely want threads.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::int64_t omp_min_parallel_flops();

EINSUMS_NAMESPACE_END(hardware)

/**
 * @def EINSUMS_OMP_PARALLEL_FOR_SIMD_IF
 *
 * Parallelize and vectorize the following loop only when @p cond holds.
 *
 * Use this instead of @ref EINSUMS_OMP_PARALLEL_FOR_SIMD on any loop whose trip
 * count can be small, passing something like
 * ``elems >= einsums::hardware::omp_min_parallel_elements()``. An unconditional
 * region on a short loop costs far more than the loop.
 *
 * @versionadded{2.0.0}
 */
// Routed through the clause-aware macro: EINSUMS_OMP_SIMD_PRAGMA appends
// ` simd` to the END of what it is given, which puts it after the `if` clause
// and yields a pragma icx rejects outright.
#define EINSUMS_OMP_PARALLEL_FOR_SIMD_IF(cond) EINSUMS_OMP_SIMD_CLAUSE_PRAGMA(parallel for, if (cond))

/**
 * @def EINSUMS_OMP_PARALLEL_FOR_IF
 *
 * Parallelize the following loop only when @p cond holds. See
 * @ref EINSUMS_OMP_PARALLEL_FOR_SIMD_IF.
 *
 * @versionadded{2.0.0}
 */
#define EINSUMS_OMP_PARALLEL_FOR_IF(cond) EINSUMS_OMP_PRAGMA(parallel for if (cond))
