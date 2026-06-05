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

EINSUMS_EXPORT int init_Einsums_Utilities(); // NOLINT(readability-identifier-naming)

/**
 * @brief Initializes the random number generator.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void initialize_Einsums_Utilities(); // NOLINT(readability-identifier-naming)

namespace detail {

static int initialize_module_Einsums_Utilities = init_Einsums_Utilities(); // NOLINT(bugprone-throwing-static-initialization)

}

} // namespace einsums