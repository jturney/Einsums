//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>

namespace {

// Module initialization: register startup/shutdown hooks.
// MPI_Init must happen before any other MPI call, so we use pre-startup.
// MPI_Finalize must happen after all MPI calls, so we use shutdown.

[[maybe_unused]] auto init_Einsums_Comm = []() {
    einsums::register_pre_startup_function([]() {
        EINSUMS_LOG_INFO("Comm: initializing communication layer");
        einsums::comm::initialize();
    });
    einsums::register_shutdown_function([]() {
        EINSUMS_LOG_INFO("Comm: finalizing communication layer");
        einsums::comm::finalize();
    });
    return 0;
}();

} // namespace
