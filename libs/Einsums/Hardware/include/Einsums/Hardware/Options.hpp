//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <cstdint>
#include <limits>

/*
 * The hardware detector's options, declared where the detector reads them.
 *
 * Each one replaces a detected or measured fact, so a benchmark can compare
 * two settings from ONE binary (the benchmarks involved swing by tens of
 * percent across rebuilds) and a test can pin the machine model it asserts
 * against on a runner whose own hardware would give another.
 *
 * The detector memoizes the facts on first use, so an option only takes
 * effect when it is parsed before then: on the command line or in the
 * environment, not through a config::set after the first contraction.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// L1 data cache size in bytes, in place of the detected one. Zero, the default, detects it.
inline constinit cl::ConfigOption<std::int64_t> HardwareL1CacheSize =
    cl::config_opt<std::int64_t>("einsums:hardware:l1-cache-size", "L1 data cache size in bytes (0 = detect)", "Hardware", 0, "BYTES",
                                 cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 40));

/// L2 cache size in bytes, in place of the detected one. Zero, the default, detects it.
inline constinit cl::ConfigOption<std::int64_t> HardwareL2CacheSize =
    cl::config_opt<std::int64_t>("einsums:hardware:l2-cache-size", "L2 cache size in bytes (0 = detect)", "Hardware", 0, "BYTES",
                                 cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 40));

/// L3 cache size in bytes, in place of the detected one. Zero, the default, detects it.
inline constinit cl::ConfigOption<std::int64_t> HardwareL3CacheSize =
    cl::config_opt<std::int64_t>("einsums:hardware:l3-cache-size", "L3 cache size in bytes (0 = detect)", "Hardware", 0, "BYTES",
                                 cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 40));

/// Directory where measured constants are remembered between runs.
///
/// Empty, the default, uses the platform's cache location: `%LOCALAPPDATA%`,
/// `$XDG_CACHE_HOME` or `~/.cache`, each with an `einsums` subdirectory.
inline constinit cl::ConfigOption<std::string> CacheDir = cl::config_opt<std::string>(
    "einsums:cache-dir", "Directory for measured constants kept between runs (empty = the platform cache location)", "Hardware", "",
    "PATH");

/// The calibration file the OpenMP region cost is read from.
///
/// Empty, the default, means a host-keyed file under @ref CacheDir. Either way
/// the file is optional: a missing or unreadable one means measure. It mirrors
/// `einsums:hardware:profile`, which does the same for the ComputeGraph cost
/// model, and like that file it is only ever written by `calibrate_hardware`.
inline constinit cl::ConfigOption<std::string> HardwareCalibration = cl::config_opt<std::string>(
    "einsums:hardware:calibration", "Calibration file for the OpenMP region cost (empty = a host-keyed file under the cache directory)",
    "Hardware", "", "PATH");

/// Pin the cost of entering an OpenMP parallel region, in nanoseconds.
///
/// Negative, the default, takes the calibrated value when there is one and
/// measures otherwise. A pin beats both, so a benchmark can hold the rate
/// fixed across machines without touching any file.
inline constinit cl::ConfigOption<double> HardwareOmpRegionCostNs = cl::config_opt<double>(
    "einsums:hardware:omp-region-cost-ns", "Pin the cost of entering an OpenMP parallel region, in ns (-1 = calibrate or measure)",
    "Hardware", -1.0, "NS", cl::RangeBetween(-1.0, 1e9));

/// Pin the element count at which elementwise kernels open a parallel region.
///
/// Negative, the default, derives it from the region cost. Zero parallelizes
/// every loop.
inline constinit cl::ConfigOption<std::int64_t> HardwareOmpMinParallelElements = cl::config_opt<std::int64_t>(
    "einsums:hardware:omp-min-parallel-elements", "Elements at which elementwise kernels go parallel (-1 = derive from the region cost)",
    "Hardware", -1, "N", cl::RangeBetween<std::int64_t>(-1, (std::numeric_limits<std::int64_t>::max)()));

/// Pin the flop count at which a contraction opens a parallel region.
///
/// Negative, the default, derives it from the region cost. Zero parallelizes
/// every contraction.
inline constinit cl::ConfigOption<std::int64_t> HardwareOmpMinParallelFlops = cl::config_opt<std::int64_t>(
    "einsums:hardware:omp-min-parallel-flops", "Flops at which a contraction goes parallel (-1 = derive from the region cost)", "Hardware",
    -1, "N", cl::RangeBetween<std::int64_t>(-1, (std::numeric_limits<std::int64_t>::max)()));

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the hardware detector's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_Hardware_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_Hardware = register_Einsums_Hardware_options();
}

EINSUMS_NAMESPACE_END()
