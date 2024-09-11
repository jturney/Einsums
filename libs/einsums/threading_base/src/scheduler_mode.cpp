//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/threading_base/scheduler_mode.hpp>

namespace einsums::threads {

scheduler_mode operator&(scheduler_mode sched1, scheduler_mode sched2) {
    return static_cast<scheduler_mode>(static_cast<std::uint32_t>(sched1) & static_cast<std::uint32_t>(sched2));
}

scheduler_mode operator|(scheduler_mode sched1, scheduler_mode sched2) {
    return static_cast<scheduler_mode>(static_cast<std::uint32_t>(sched1) | static_cast<std::uint32_t>(sched2));
}

scheduler_mode operator~(scheduler_mode sched) {
    return static_cast<scheduler_mode>(~static_cast<std::uint32_t>(sched));
}

} // namespace einsums::threads