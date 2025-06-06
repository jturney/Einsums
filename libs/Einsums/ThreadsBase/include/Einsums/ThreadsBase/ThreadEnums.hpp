//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

namespace einsums::threads::detail {

enum class ThreadSchedulerState : std::int8_t {
    unknown = 0,
    active  = 1, /*!< thread is currently active (running, has resources) */
    pending = 2, /*!< thread is pending (ready to run, but no hardware resources available) */
    suspended =
        3, /*!< thread has been suspended (waiting for synchronization event, but still known and under control of the thread-manager) */
    terminated = 4, /*!< thread has been stopped and may be garbage collected */
    staged = 5, /*!< this is not a real thread state, but allows reference to staged task descriptions, will eventually be converted into
                   thread objects */
    pending_do_not_schedule = 6, /*!< this is not a real thread state, but allows us to create a thread in a pending state without
                                    scheduling it (internal flag) */
    pending_boost =
        7 /*!< this is not a real thread state, but allows us to suspend a thread in a pending state without high priority rescheduling */
};

EINSUMS_EXPORT char const *get_thread_state_name(ThreadSchedulerState state);

enum class ThreadRestartState : std::int8_t {

};
} // namespace einsums::threads::detail
