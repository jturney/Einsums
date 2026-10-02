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

/// Cap the SIMD dispatch rung. Empty leaves the choice to Stripes (STRIPES_ARCH, else the best the
/// machine runs). It can only lower the rung: an unsupported rung falls to the next one below, with
/// a warning (see stripes::resolve_arch()). The rung is fixed at the first dispatch, so set this on
/// the command line or in the environment.
inline constinit cl::ConfigOption<std::string> SimdArch = cl::config_opt<std::string>(
    "einsums:simd:arch", "Cap the SIMD dispatch rung: baseline, v2, v3, v4 or sme (empty = the best supported)", "SIMD", "", "RUNG");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the dispatch ladder's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_SIMD_options();

/**
 * @brief Pass --einsums:simd:arch and a log handler to Stripes. Called once options and logging are
 *        up, before anything dispatches.
 */
EINSUMS_EXPORT void apply_SIMD_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_SIMD = register_Einsums_SIMD_options();
}

EINSUMS_NAMESPACE_END()
