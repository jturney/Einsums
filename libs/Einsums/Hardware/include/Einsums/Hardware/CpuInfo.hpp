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
 * Read from a calibration file when one exists, and measured here and now when
 * it does not. Per process the measurement drifts by tens of percent with
 * whatever else is starting up, which is harmless for a threshold and not
 * harmless for a chooser ranking discrete options against the rate.
 *
 * Nothing in the library writes that file. @ref write_calibration does, and
 * @c calibrate_hardware calls it, so the value is one somebody chose to record
 * rather than whichever measurement a process happened to take first. A cached
 * measurement taken while the machine was busy is indistinguishable from a
 * slower machine; a file you generated can be regenerated, diffed and deleted.
 *
 * @c EINSUMS_OMP_REGION_COST_NS pins a value outright.
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

/**
 * @brief Where @ref omp_region_cost_ns looks for a calibration file.
 *
 * @c EINSUMS_HARDWARE_CALIBRATION if set, otherwise a host-keyed default under
 * the platform cache directory. Empty when there is nowhere to look, which is
 * not an error: the measurement simply happens per process.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string default_calibration_path();

/**
 * @brief Whether @ref omp_region_cost_ns came from a calibration rather than a
 *        measurement taken just now.
 *
 * The calibrated path is the one worth being on. A measurement taken in-process
 * is taken under whatever the machine and the OpenMP state happen to be at that
 * moment, and a caller that ranks discrete options against the rate wants
 * neither of those in the answer. This is how such a caller can say which it
 * got, rather than leaving a reader of its numbers to guess.
 *
 * True for a single-threaded run, where the cost is exactly zero either way, and
 * for an explicit @c EINSUMS_OMP_REGION_COST_NS pin.
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT bool region_cost_is_calibrated();

/**
 * @brief Measure the OpenMP region cost at every team size and record it.
 *
 * The write half of the calibration file, owned here because this module owns
 * the facts in it and sits below the one that owns a JSON parser. Called by the
 * @c calibrate_hardware tool, which is where a person asks for this; the library
 * itself never writes, so a value is only ever one somebody chose to record.
 *
 * Sweeps team sizes 1..@c omp_get_max_threads() because the cost is a function
 * of the team, and a later run at four threads must not be handed the ten-thread
 * number. A few milliseconds per entry.
 *
 * @param path Destination, or empty for @ref default_calibration_path.
 * @param error Set to a human-readable reason when this returns false.
 * @return Whether the file was written.
 */
EINSUMS_EXPORT bool write_calibration(std::string const &path, std::string *error = nullptr);

/**
 * @brief The OpenMP runtime's current ceiling on a parallel region's team size.
 *
 * A thin wrapper over ``omp_get_max_threads()``, reading the same process-wide
 * runtime every OpenMP-threaded BLAS and every @c #pragma @c omp @c parallel
 * region in the process reads. It is not ``OMP_NUM_THREADS``: the environment
 * records what was requested at startup, this reports what is in effect now.
 *
 * Returns 1 in a build without OpenMP, which is also the conservative answer
 * for anything gating on thread count.
 *
 * @return The team size a parallel region would get right now.
 *
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT int get_max_threads();

/**
 * @brief Set the OpenMP runtime's ceiling on a parallel region's team size.
 *
 * A thin wrapper over ``omp_set_num_threads()``. This is process-wide, the
 * same ICV every OpenMP-threaded BLAS and every @c #pragma @c omp @c parallel
 * region in the process reads, not a per-thread setting - see
 * @ref einsums::blas::set_num_threads_this_thread for the narrower per-thread
 * BLAS knob.
 *
 * A no-op in a build without OpenMP.
 *
 * @param[in] nthreads Thread count ceiling to request.
 *
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT void set_num_threads(int nthreads);

EINSUMS_NAMESPACE_END(hardware)
