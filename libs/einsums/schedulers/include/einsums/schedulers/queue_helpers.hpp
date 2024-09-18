//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/logging.hpp>
#include <einsums/schedulers/deadlock_detection.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_queue_init_parameters.hpp>
#include <einsums/type_support/unused.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::threads::detail {
///////////////////////////////////////////////////////////////////////////
// debug helper function, logs all suspended threads
// this returns true if all threads in the map are currently suspended
template <typename Map>
bool dump_suspended_threads(std::size_t num_thread, Map &tm, std::int64_t &idle_loop_count, bool running) EINSUMS_COLD;

template <typename Map>
bool dump_suspended_threads(std::size_t num_thread, Map &tm, std::int64_t &idle_loop_count, bool running) {
#if !defined(EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION)
    EINSUMS_UNUSED(num_thread);
    EINSUMS_UNUSED(tm);
    EINSUMS_UNUSED(idle_loop_count);
    EINSUMS_UNUSED(running); //-V601
    return false;
#else
    if (!get_deadlock_detection_enabled())
        return false;

    // attempt to output possibly deadlocked threads occasionally only
    if (EINSUMS_LIKELY((idle_loop_count % EINSUMS_IDLE_LOOP_COUNT_MAX) != 0))
        return false;

    bool result            = false;
    bool collect_suspended = true;

    bool                         logged_headline = false;
    typename Map::const_iterator end             = tm.end();
    for (typename Map::const_iterator it = tm.begin(); it != end; ++it) {
        threads::detail::thread_data const    *thrd         = get_thread_id_data(*it);
        threads::detail::thread_schedule_state state        = thrd->get_state().state();
        threads::detail::thread_schedule_state marked_state = thrd->get_marked_state();

        if (state != marked_state) {
            // log each thread only once
            if (!logged_headline) {
                if (running) {
                    EINSUMS_LOG(warn, "Listing suspended threads while queue ({}) is empty:", num_thread);
                } else {
                    EINSUMS_LOG(warn, "  [TM] Listing suspended threads while queue ({}) is empty:\n", num_thread);
                }
                logged_headline = true;
            }

            if (running) {
                EINSUMS_LOG(warn, "queue({}): {}({}.{:02x}) P{}: {}: {}", num_thread, get_thread_state_name(state), *it,
                            thrd->get_thread_phase(), thrd->get_parent_thread_id(), thrd->get_description(), thrd->get_lco_description());
            } else {
                EINSUMS_LOG(warn, "queue({}): {}({}.{:02x}) P{}: {}: {}", num_thread, get_thread_state_name(state), *it,
                            thrd->get_thread_phase(), thrd->get_parent_thread_id(), thrd->get_description(), thrd->get_lco_description());
            }
            thrd->set_marked_state(state);

            // result should be true if we found only suspended threads
            if (collect_suspended) {
                switch (state) {
                case threads::detail::thread_schedule_state::suspended:
                    result = true; // at least one is suspended
                    break;

                case threads::detail::thread_schedule_state::pending:
                case threads::detail::thread_schedule_state::active:
                    result            = false; // one is active, no deadlock (yet)
                    collect_suspended = false;
                    break;

                default:
                    // If the thread is terminated we don't care too much
                    // anymore.
                    break;
                }
            }
        }
    }
    return result;
#endif
}
} // namespace einsums::threads::detail
