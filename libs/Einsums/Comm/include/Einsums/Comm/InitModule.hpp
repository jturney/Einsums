//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

/*
 * Header-driven module initialization. The static initializer in detail:: below
 * forces every TU that includes this header to take a reference to
 * setup_Einsums_Comm(), which both (a) prevents `--gc-sections` from dropping
 * the symbol at link time and (b) auto-runs the registration at executable
 * load. This is the same pattern Tensor, BLAS, BufferAllocator, and others use. Comm's
 * register_pre_startup_function(initialize) is what actually flips
 * comm::is_initialized() to true once the runtime starts; without this header
 * being referenced from a public, library-internal TU, the pre-startup hook
 * never fires.
 */

namespace einsums {

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

} // namespace einsums
