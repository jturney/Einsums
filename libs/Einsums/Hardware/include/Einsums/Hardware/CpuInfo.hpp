//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

EINSUMS_NAMESPACE_BEGIN(hardware)

/// Data-cache sizes in bytes. Zero-initialised fields mean "not detected"; the
/// accessors below always return usable values, falling back to conservative
/// defaults.
struct CacheSizes {
    std::int64_t l1{std::int64_t{32} << 10};
    std::int64_t l2{std::int64_t{256} << 10};
    std::int64_t l3{std::int64_t{8} << 20};
};

/**
 * @brief Hardware facts detected once, shared by every module that needs them.
 *
 * Detected in one place so modules cannot disagree. Holds facts and thresholds derived directly
 * from them; tuning policy such as PackedGemm's blocking stays with the algorithm.
 */
struct CpuInfo {
    /// SIMD vector length in doubles of the rung the running process dispatches to: SSE/NEON = 2,
    /// AVX = 4, AVX-512 = 8. From `stripes::selected_arch()`, not compile flags; the width the
    /// library was compiled for is `compiled_simd_width_f64`.
    int simd_width_f64{2};

    /// SIMD vector length in floats of the selected rung: twice `simd_width_f64`.
    int simd_width_f32{4};

    /// SIMD vector length in doubles implied by the compile flags of the library
    /// itself (the width of `stripes::Vec<double>` in a non-rung translation unit).
    int compiled_simd_width_f64{2};

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
 * An empty region at the current team size, warm team, best of several trials: around 20 us on
 * a 10-thread machine. Read from a calibration file (written only by @c calibrate_hardware) when
 * there is one, otherwise measured, which drifts by tens of percent per process.
 * @c --einsums:hardware:omp-region-cost-ns pins it.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT double omp_region_cost_ns();

/**
 * @brief Elements below which an elementwise loop should not be parallelized.
 *
 * The region cost at a conservative one element per nanosecond.
 * @c --einsums:hardware:omp-min-parallel-elements pins it.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::size_t omp_min_parallel_elements();

/**
 * @brief Work, in flops, below which parallelizing a contraction is a net loss.
 *
 * The region cost times a multiplier calibrated end to end on tiled CCSD residuals.
 * @c --einsums:hardware:omp-min-parallel-flops pins it.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::int64_t omp_min_parallel_flops();

/**
 * @brief Where @ref omp_region_cost_ns looks for a calibration file.
 *
 * @c --einsums:hardware:calibration if set, otherwise a host-keyed file in the platform cache
 * directory. Empty when there is nowhere to look; the cost is then measured.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string default_calibration_path();

/**
 * @brief Whether @ref omp_region_cost_ns came from a calibration, not an in-process measurement.
 *        Also true single-threaded (the cost is zero) and when pinned.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT bool region_cost_is_calibrated();

/**
 * @brief Measure the OpenMP region cost at every team size and record it.
 *
 * Called by the @c calibrate_hardware tool; the library never writes the file itself. Sweeps
 * team sizes 1..@c omp_get_max_threads(), a few milliseconds each.
 *
 * @param path Destination, or empty for @ref default_calibration_path.
 * @param error Set to a human-readable reason when this returns false.
 * @return Whether the file was written.
 */
EINSUMS_EXPORT bool write_calibration(std::string const &path, std::string *error = nullptr);

/**
 * @brief The OpenMP runtime's current ceiling on a parallel region's team size.
 *
 * ``omp_get_max_threads()``: what is in effect now, not what ``OMP_NUM_THREADS`` requested.
 * 1 without OpenMP.
 *
 * @return The team size a parallel region would get right now.
 *
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT int get_max_threads();

/**
 * @brief Set the OpenMP runtime's ceiling on a parallel region's team size.
 *
 * ``omp_set_num_threads()``, which also governs OpenMP-threaded BLAS; for a per-thread BLAS
 * setting see @ref einsums::blas::set_num_threads_this_thread. A no-op without OpenMP.
 *
 * @param[in] nthreads Thread count ceiling to request.
 *
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT void set_num_threads(int nthreads);

EINSUMS_NAMESPACE_END(hardware)
