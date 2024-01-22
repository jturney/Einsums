//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace einsums {

enum class runtime_state : std::int8_t {
    invalid             = -1,
    initialized         = 0,
    first_valid_runtime = initialized,
    pre_startup         = 1,
    startup             = 2,
    pre_main            = 3,
    starting            = 4,
    running             = 5,
    suspended           = 6,
    pre_sleep           = 7,
    sleeping            = 8,
    pre_shutdown        = 9,
    shutdown            = 10,
    stopping            = 11,
    terminating         = 12,
    stopped             = 13,
    last_valid_runtime  = stopped
};

}