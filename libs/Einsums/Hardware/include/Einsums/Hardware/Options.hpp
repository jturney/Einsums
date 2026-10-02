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
 * The hardware detector's options. Each overrides a detected or measured fact, for same-binary
 * A/B runs and for tests that pin a machine model. Facts are memoized on first use, so set these
 * on the command line or in the environment.
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

/// The optional calibration file for the OpenMP region cost; empty means a host-keyed file under
/// @ref CacheDir. Written only by `calibrate_hardware`.
inline constinit cl::ConfigOption<std::string> HardwareCalibration = cl::config_opt<std::string>(
    "einsums:hardware:calibration", "Calibration file for the OpenMP region cost (empty = a host-keyed file under the cache directory)",
    "Hardware", "", "PATH");

/// Pin the cost of entering an OpenMP parallel region, in nanoseconds. Negative calibrates or measures.
inline constinit cl::ConfigOption<double> HardwareOmpRegionCostNs = cl::config_opt<double>(
    "einsums:hardware:omp-region-cost-ns", "Pin the cost of entering an OpenMP parallel region, in ns (-1 = calibrate or measure)",
    "Hardware", -1.0, "NS", cl::RangeBetween(-1.0, 1e9));

/// Pin the element count at which elementwise kernels go parallel. Negative derives it; zero always.
inline constinit cl::ConfigOption<std::int64_t> HardwareOmpMinParallelElements = cl::config_opt<std::int64_t>(
    "einsums:hardware:omp-min-parallel-elements", "Elements at which elementwise kernels go parallel (-1 = derive from the region cost)",
    "Hardware", -1, "N", cl::RangeBetween<std::int64_t>(-1, (std::numeric_limits<std::int64_t>::max)()));

/// Pin the flop count at which a contraction goes parallel. Negative derives it; zero always.
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
