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
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Measure this machine and write its hardware profile, or report on the profile in effect.
 *
 * The one implementation behind the @c einsums-tune executable, @c einsums @c tune and
 * @c python @c -m @c einsums.tune. Without an action it measures GEMM efficiency, memory and
 * cache bandwidth, permute throughput, call overhead, per-kernel thread efficiency and the OpenMP
 * region cost, and writes them as one profile, by default this machine's in the cache directory.
 * The actions report instead: @c --show the profile in effect and why one is not used, @c --check
 * whether a current one applies, @c --export its path, and @c --import adopts a file measured
 * elsewhere. @c --help lists every argument.
 *
 * The runtime must be running; its own @c --einsums:* options among @p args are skipped.
 *
 * @param[in] args The arguments, without the program name.
 * @return The exit status: 0 on success, 1 when an action reports failure, 2 for bad arguments.
 * @versionadded{2.0.0}
 */
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT int tune(std::vector<std::string> const &args);

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief One size of a crossover sweep: the median time of each side of a decision there.
 * @versionadded{2.0.0}
 */
struct CrossoverPoint {
    double size{0.0};       ///< What the decision's threshold is compared against: elements, flops.
    double small_side{0.0}; ///< Seconds on the side the threshold gives small sizes.
    double large_side{0.0}; ///< Seconds on the side the threshold gives large sizes.
};

/**
 * @brief How far a crossover is known.
 * @versionadded{2.0.0}
 */
enum class CrossoverBound : std::uint8_t {
    Exact,   ///< Between two swept sizes.
    AtMost,  ///< The large side already wins at the smallest size swept.
    AtLeast, ///< The large side still loses at the largest size swept.
};

/**
 * @brief Where a decision's threshold belongs, from a sweep.
 * @versionadded{2.0.0}
 */
struct Crossover {
    double         value{0.0};                   ///< The threshold.
    CrossoverBound bound{CrossoverBound::Exact}; ///< Whether the sweep bracketed it.
};

/**
 * @brief The size from which the large side wins and keeps winning.
 *
 * A point within @p tie of even counts for neither side, so one noisy point near the crossover
 * does not move it. Between the last size the large side loses and the first from which it never
 * does, the threshold is their geometric mean; a sweep that never brackets it returns the end it
 * ran into, marked as a bound.
 *
 * @param[in] points The sweep, by increasing size; not empty.
 * @param[in] tie The relative difference below which two times are even.
 * @return The crossover.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT Crossover find_crossover(std::vector<CrossoverPoint> const &points, double tie = 0.03);

/**
 * @brief The extents the GEMM grid samples on each axis: @p count of them spaced evenly in log size
 *        from @p min_extent to @p max_extent, each a multiple of 4 and none a power of two from 16 up.
 *
 * A power-of-two extent makes every leading dimension a power-of-two stride, the one case where the
 * packers' strided reads collide in the cache: PackedGemm ran a 512 x 512 x 128 GEMM at 245 GFLOPS
 * on an M4 and a 508 x 508 x 132 one at 344, while the vendor GEMM barely noticed. A grid sampled
 * on powers of two prices every packed contraction at its worst case, so each such extent is moved
 * down to the multiple of 4 below it.
 *
 * @param[in] min_extent The smallest extent.
 * @param[in] max_extent The largest extent.
 * @param[in] count How many extents, before duplicates after rounding are dropped.
 * @return The extents, ascending.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::vector<std::size_t> grid_extents(std::size_t min_extent, std::size_t max_extent, std::size_t count);

EINSUMS_NAMESPACE_END(compute_graph::detail)
