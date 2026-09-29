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
 * The runtime dispatch ladder's options, declared where the ladder reads them.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Cap the SIMD dispatch rung below the best one this machine supports.
///
/// Empty, the default, dispatches to the highest rung the CPU and operating
/// system can run. A rung the machine cannot run is replaced, with a warning,
/// by the next supported one below it, and a rung of another architecture is
/// ignored, so this can only ever lower the rung. See simd::resolve_arch().
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

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_SIMD = register_Einsums_SIMD_options();
}

EINSUMS_NAMESPACE_END()
