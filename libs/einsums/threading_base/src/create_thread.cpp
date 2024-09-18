//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/logging.hpp>
#include <einsums/modules/coroutines.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/threading_base/create_thread.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_init_data.hpp>

#include <cstddef>

namespace einsums::threads::detail {
void create_thread(scheduler_base *scheduler, thread_init_data &data, thread_id_ref_type &id, error_code &ec) {
    // verify parameters
    switch (data.initial_state) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case thread_schedule_state::pending:
        [[fallthrough]];
    case thread_schedule_state::pending_do_not_schedule:
        [[fallthrough]];
    case thread_schedule_state::pending_boost:
        [[fallthrough]];
    case thread_schedule_state::suspended:
        break;

    default: {
        EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter, "invalid initial state: {}", data.initial_state);
        return;
    }
    }

#ifdef EINSUMS_HAVE_THREAD_DESCRIPTION
    if (!data.description) {
        EINSUMS_THROWS_IF(ec, einsums::error::bad_parameter, "description is nullptr");
        return;
    }
#endif

    thread_self *self = get_self_ptr();

#ifdef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
    if (nullptr == data.parent_id) {
        if (self) {
            data.parent_id    = get_thread_id_data(threads::detail::get_self_id());
            data.parent_phase = self->get_thread_phase();
        }
    }
#endif

    if (nullptr == data.scheduler_base)
        data.scheduler_base = scheduler;

    // Pass recursive high priority from parent to child (but only if there is none is
    // explicitly specified).
    if (self) {
        if (data.priority == execution::thread_priority::default_ &&
            execution::thread_priority::high_recursive == get_thread_id_data(get_self_id())->get_priority()) {
            data.priority = execution::thread_priority::high_recursive;
        }
    }

    if (data.priority == execution::thread_priority::default_)
        data.priority = execution::thread_priority::normal;

    // create the new thread
    scheduler->create_thread(data, &id, ec);

    EINSUMS_LOG(info,
                fmt::runtime("create_thread: pool({}), scheduler({}), thread({}), initial_state({}), run_now({}), "
                             "description({})"),
                *scheduler->get_parent_pool(), *scheduler, id, get_thread_state_name(data.initial_state), data.run_now,
                data.get_description());

    // NOTE: Don't care if the hint is a NUMA hint, just want to wake up a
    // thread.
    scheduler->do_some_work(data.schedule_hint.hint);
}
} // namespace einsums::threads::detail
