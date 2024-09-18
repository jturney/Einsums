//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/coroutine.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/timing.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/set_thread_state.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>

#include <atomic>

namespace einsums::threads::detail {

/// Set a timer to set the state of the given \a thread to the given
/// new value after it expired (at the given time)
EINSUMS_EXPORT thread_id_ref_type set_thread_state_timed(scheduler_base *scheduler, einsums::chrono::steady_time_point const &abs_time,
                                                         thread_id_type const &thrd, thread_schedule_state newstate,
                                                         thread_restart_state newstate_ex, execution::thread_priority priority,
                                                         execution::thread_schedule_hint schedulehint, std::atomic<bool> *started,
                                                         bool retry_on_active, error_code &ec);

inline thread_id_ref_type set_thread_state_timed(scheduler_base *scheduler, einsums::chrono::steady_time_point const &abs_time,
                                                 thread_id_type const &id, std::atomic<bool> *started, bool retry_on_active,
                                                 error_code &ec) {
    return set_thread_state_timed(scheduler, abs_time, id, thread_schedule_state::pending, thread_restart_state::timeout,
                                  execution::thread_priority::normal, execution::thread_schedule_hint(), started, retry_on_active, ec);
}

// Set a timer to set the state of the given \a thread to the given
// new value after it expired (after the given duration)
inline thread_id_ref_type set_thread_state_timed(scheduler_base *scheduler, einsums::chrono::steady_duration const &rel_time,
                                                 thread_id_type const &thrd, thread_schedule_state newstate,
                                                 thread_restart_state newstate_ex, execution::thread_priority priority,
                                                 execution::thread_schedule_hint schedulehint, std::atomic<bool> *started,
                                                 bool retry_on_active, error_code &ec) {
    return set_thread_state_timed(scheduler, rel_time.from_now(), thrd, newstate, newstate_ex, priority, schedulehint, started,
                                  retry_on_active, ec);
}

inline thread_id_ref_type set_thread_state_timed(scheduler_base *scheduler, einsums::chrono::steady_duration const &rel_time,
                                                 thread_id_type const &thrd, std::atomic<bool> *started, bool retry_on_active,
                                                 error_code &ec) {
    return set_thread_state_timed(scheduler, rel_time.from_now(), thrd, thread_schedule_state::pending, thread_restart_state::timeout,
                                  execution::thread_priority::normal, execution::thread_schedule_hint(), started, retry_on_active, ec);
}
} // namespace einsums::threads::detail
