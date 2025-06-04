//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace einsums::threads {

/// This enumeration describes the possible modes of a scheduler.
enum class SchedulerMode : std::uint32_t {
    /// As the name suggests, this option can be used to disable all other options.
    NothingSpecial = 0x000,
    
};

}