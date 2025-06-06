//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <../../ThreadsBase/include/Einsums/ThreadsBase/SchedulerMode.hpp>

namespace einsums::threads {

SchedulerMode operator&(SchedulerMode sched1, SchedulerMode sched2) {
    return static_cast<SchedulerMode>(static_cast<std::uint32_t>(sched1) & static_cast<std::uint32_t>(sched2));
}

SchedulerMode operator|(SchedulerMode sched1, SchedulerMode sched2) {
    return static_cast<SchedulerMode>(static_cast<std::uint32_t>(sched1) | static_cast<std::uint32_t>(sched2));
}

SchedulerMode operator~(SchedulerMode sched) {
    return static_cast<SchedulerMode>(~static_cast<std::uint32_t>(sched));
}

} // namespace einsums::threads