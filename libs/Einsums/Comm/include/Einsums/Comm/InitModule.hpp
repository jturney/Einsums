//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

/*
 * The static initializer below makes every including TU reference setup_Einsums_Comm(), so the
 * linker keeps it and the registration runs at load. Without it the pre-startup hook that sets
 * comm::is_initialized() never fires.
 */

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Registers the initialization and finalization functions with the runtime manager.
 */
EINSUMS_EXPORT int setup_Einsums_Comm(); // NOLINT(readability-identifier-naming)

/**
 * @brief Initialize the Comm runtime (MPI_Init under MPI, no-op under the mock backend).
 */
EINSUMS_EXPORT void initialize_Einsums_Comm(); // NOLINT(readability-identifier-naming)

/**
 * @brief Finalize the Comm runtime (MPI_Finalize under MPI, no-op under the mock backend).
 */
EINSUMS_EXPORT void finalize_Einsums_Comm(); // NOLINT(readability-identifier-naming)

namespace detail {
static int initialize_module_Einsums_Comm = setup_Einsums_Comm(); // NOLINT(bugprone-throwing-static-initialization)

} // namespace detail

EINSUMS_NAMESPACE_END()
