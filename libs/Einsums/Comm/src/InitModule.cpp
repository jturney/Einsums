//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/InitModule.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>

/*
 * Mirrors the BLAS / Tensor / etc. InitModule pattern. setup_Einsums_Comm()
 * registers the pre-startup / shutdown hooks the first time it's called; the
 * companion InitModule.hpp's header-driven static initializer guarantees it
 * IS called from every TU that includes the header (which is what keeps the
 * linker from gc'ing it and what makes the registration actually happen at
 * load time). MPI_Init must run before any other MPI call, so we use
 * register_pre_startup_function rather than the regular startup one.
 */
namespace einsums {

int setup_Einsums_Comm() { // NOLINT(readability-identifier-naming)
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_pre_startup_function(einsums::initialize_Einsums_Comm);
        einsums::register_shutdown_function(einsums::finalize_Einsums_Comm);
        is_initialized = true;
    }

    return 0;
}

void initialize_Einsums_Comm() { // NOLINT(readability-identifier-naming)
    EINSUMS_LOG_INFO("Comm: initializing communication layer");
    einsums::comm::initialize();
}

void finalize_Einsums_Comm() { // NOLINT(readability-identifier-naming)
    EINSUMS_LOG_INFO("Comm: finalizing communication layer");
    einsums::comm::finalize();
}

} // namespace einsums
