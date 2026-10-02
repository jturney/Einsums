//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/InitModule.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>

/*
 * Registers the hooks on first call (see InitModule.hpp). A pre-startup hook, because MPI_Init must
 * precede every other MPI call.
 */
EINSUMS_NAMESPACE_BEGIN()

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

EINSUMS_NAMESPACE_END()
