//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <string>

/*
 * The dispatch ladder's option. Stripes chooses the rung (Stripes/RuntimeFeatures.hpp); Einsums
 * declares the option and passes it on (apply_SIMD_options).
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Cap the SIMD dispatch rung below the best one this machine supports.
///
/// Empty, the default, leaves the choice to Stripes: its STRIPES_ARCH
/// environment variable if set, and otherwise the highest rung the CPU and
/// operating system can run. A rung the machine cannot run is replaced, with a
/// warning, by the next supported one below it, and a rung of another
/// architecture is ignored, so this can only ever lower the rung. See
/// stripes::resolve_arch().
///
/// The rung is chosen on the first dispatch and kept, so this must be parsed
/// before then: on the command line or in the environment.
inline constinit cl::ConfigOption<std::string> SimdArch = cl::config_opt<std::string>(
    "einsums:simd:arch", "Cap the SIMD dispatch rung: baseline, v2, v3, v4 or sme (empty = the best supported)", "SIMD", "", "RUNG");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the dispatch ladder's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_SIMD_options();

/**
 * @brief Hand Stripes, which chooses the dispatch rung, the --einsums:simd:arch setting and Einsums'
 *        logging for its messages. The runtime calls it once the options are parsed and logging is
 *        up, which is before anything dispatches.
 */
EINSUMS_EXPORT void apply_SIMD_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_SIMD = register_Einsums_SIMD_options();
}

EINSUMS_NAMESPACE_END()
