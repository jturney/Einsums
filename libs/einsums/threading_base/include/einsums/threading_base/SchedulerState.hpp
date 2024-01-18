//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace einsums {

enum class RuntimeState : std::int8_t {
    Invalid           = -1,
    Initialized       = 0,
    FirstValidRuntime = Initialized,
    PreStartup        = 1,
    Startup           = 2,
    PreMain           = 3,
    Starting          = 4,
    Running           = 5,
    Suspended         = 6,
    PreSleep          = 7,
    Sleeping          = 8,
    PreShutdown       = 9,
    Shutdown          = 10,
    Stopping          = 11,
    Terminating       = 12,
    Stopped           = 13,
    LastValidRuntime  = Stopped
};

}