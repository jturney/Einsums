//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>


EINSUMS_NAMESPACE_BEGIN()

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

EINSUMS_NAMESPACE_END()