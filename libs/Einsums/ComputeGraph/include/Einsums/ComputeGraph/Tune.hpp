//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

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
