//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Errors.hpp>

#include <cstdint>

namespace einsums {

/**
 * @struct invalid_runtime_state
 *
 * Indicates that the code is handling data that is uninitialized.
 */
struct EINSUMS_EXPORT invalid_runtime_state : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @enum RuntimeState
 *
 * @brief Holds the possible states for the runtime.
 */
enum class RuntimeState : std::int8_t {
    Invalid        = -1, /**< The state is invalid. */
    Initialized    = 0,  /**< The runtime has been initialized. */
    PreStartup     = 1,  /**< The runtime is running the pre-startup functions. */
    Startup        = 2,  /**< The runtime is running the startup functions. */
    PreMain        = 3,  /**< The runtime is preparing to run the main function. */
    Starting       = 4,  /**< The runtime is starting the main function. */
    Running        = 5,  /**< The main function is running. */
    Suspended      = 6,
    PreSleep       = 7,
    Sleeping       = 8,
    PreShutdown    = 9,       /**< The pre-shutdown functions are running. */
    Shutdown       = 10,      /**< The shutdown functions are running. */
    Stopping       = 11,      /**< The runtime is stopping. */
    Terminating    = 12,      /**< The runtime is terminating. */
    Stopped        = 13,      /**< The runtime has stopped. */
    LastValidState = Stopped, /**< Indicates the last valid state. Anything past this is considered invalid. */
};

} // namespace einsums