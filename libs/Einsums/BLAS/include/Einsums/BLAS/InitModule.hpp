//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 */

namespace einsums {

/**
 * @brief Registers the initialization and finalization functions with the runtime manager.
 *
 * @versionadded{1.1.0}
 */
EINSUMS_EXPORT int setup_Einsums_BLAS(); // NOLINT(readability-identifier-naming)

/**
 * @brief Initialize the BLAS runtime.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize_Einsums_BLAS(); // NOLINT(readability-identifier-naming)

/**
 * @brief Finalize the BLAS runtime.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void finalize_Einsums_BLAS(); // NOLINT(readability-identifier-naming)

namespace detail {
#ifndef DOXYGEN
static int initialize_module_Einsums_BLAS = setup_Einsums_BLAS(); // NOLINT(bugprone-throwing-static-initialization)
#endif

} // namespace detail

} // namespace einsums