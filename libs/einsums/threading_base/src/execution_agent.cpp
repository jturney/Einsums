//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/errors/throw_exception.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/threading_base/detail/reset_lco_description.hpp>
#include <einsums/threading_base/execution_agent.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/set_thread_state.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/threading_base/thread_num_tss.hpp>

#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
#    include <einsums/debugging/backtrace.hpp>
#    include <einsums/threading_base/detail/reset_backtrace.hpp>
#endif

#include <fmt/format.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace einsums::threads::detail {
execution_agent::execution_agent(coroutines::detail::coroutine_impl *coroutine) noexcept : self_(coroutine) {
}

std::string execution_agent::description() const {
    thread_id_type id = self_.get_thread_id();
    if (EINSUMS_UNLIKELY(!id)) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id, "null thread id encountered (is this executed on a einsums-thread?)");
    }

    return fmt::format("{}: {}", id, get_thread_id_data(id)->get_description());
}

void execution_agent::yield(const char *desc) {
    do_yield(desc, thread_schedule_state::pending);
}

void execution_agent::yield_k(std::size_t k, const char *desc) {
    if (k < 16) {
        EINSUMS_SMT_PAUSE;
    } else if (k < 32 || k & 1) {
        do_yield(desc, thread_schedule_state::pending_boost);
    } else {
        do_yield(desc, thread_schedule_state::pending);
    }
}

void execution_agent::spin_k(std::size_t k, const char *) {
    for (std::size_t i = 0; i < k; ++i) {
        EINSUMS_SMT_PAUSE;
    }
}

void execution_agent::resume(const char *desc) {
    do_resume(desc, thread_restart_state::signaled);
}

void execution_agent::abort(const char *desc) {
    do_resume(desc, thread_restart_state::abort);
}

void execution_agent::suspend(const char *desc) {
    do_yield(desc, thread_schedule_state::suspended);
}

void execution_agent::sleep_for(einsums::chrono::steady_duration const &sleep_duration, const char *desc) {
    sleep_until(sleep_duration.from_now(), desc);
}

void execution_agent::sleep_until(einsums::chrono::steady_time_point const &sleep_time, const char *desc) {
    // Just yield until time has passed by...
    auto now = std::chrono::steady_clock::now();

    // Note: we yield at least once to allow for other threads to
    // make progress in any case. We also use yield instead of yield_k
    // for the same reason.
    std::size_t k = 0;
    do {
        if (k < 32 || k & 1) {
            do_yield(desc, thread_schedule_state::pending_boost);
        } else {
            do_yield(desc, thread_schedule_state::pending);
        }
        ++k;
        now = std::chrono::steady_clock::now();
    } while (now < sleep_time.value());
}

#if defined(EINSUMS_HAVE_VERIFY_LOCKS)
struct on_exit_reset_held_lock_data {
    on_exit_reset_held_lock_data() : data_(einsums::util::get_held_locks_data()) {}

    ~on_exit_reset_held_lock_data() { einsums::util::set_held_locks_data(std::move(data_)); }

    std::unique_ptr<einsums::util::held_locks_data> data_;
};
#else
struct on_exit_reset_held_lock_data {};
#endif

thread_restart_state execution_agent::do_yield(const char *desc, thread_schedule_state state) {
    thread_id_ref_type id = self_.get_thread_id(); // keep alive
    if (EINSUMS_UNLIKELY(!id)) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id, "execution_agent::do_yield",
                                "null thread id encountered (is this executed on a einsums-thread?)");
    }

    // handle interruption, if needed
    thread_data *thrd_data = get_thread_id_data(id);
    EINSUMS_ASSERT(thrd_data);
    thrd_data->interruption_point();

    // Misbehaved threads may try to yield while handling an exception. This
    // is dangerous if the thread can migrate to other worker threads since
    // the count for std::uncaught_exceptions may become inconsistent
    // (including negative). If at any point in the future there is a
    // legitimate use case for yielding with uncaught exceptions this
    // assertion can be revisited, but until then we prefer to be strict
    // about it.
    EINSUMS_ASSERT(std::uncaught_exceptions() == 0);

    thrd_data->set_last_worker_thread_num(einsums::get_local_worker_thread_num());

    thread_restart_state statex = thread_restart_state::unknown;

    {
#ifdef EINSUMS_HAVE_THREAD_DESCRIPTION
        reset_lco_description desc(id.noref(), ::einsums::detail::thread_description(desc));
#endif
#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
        reset_backtrace bt(id);
#endif
        [[maybe_unused]] on_exit_reset_held_lock_data held_locks;

        EINSUMS_ASSERT(thrd_data->get_state().state() == thread_schedule_state::active);
        EINSUMS_ASSERT(state != thread_schedule_state::active);
        statex = self_.yield(thread_result_type(state, invalid_thread_id));
        EINSUMS_ASSERT(get_thread_id_data(id)->get_state().state() == thread_schedule_state::active);
    }

    // handle interruption, if needed
    thrd_data->interruption_point();

    // handle interrupt and abort
    if (statex == thread_restart_state::abort) {
        EINSUMS_THROW_EXCEPTION(einsums::error::yield_aborted, "{} thread({}) aborted (yield returned wait_abort)", desc, description());
    }

    return statex;
}

void execution_agent::do_resume(const char * /* desc */, thread_restart_state statex) {
    auto thrd = self_.get_thread_id();
    set_thread_state(std::move(thrd), thread_schedule_state::pending, statex, execution::thread_priority::normal,
                     execution::thread_schedule_hint{static_cast<std::int16_t>(get_thread_id_data(thrd)->get_last_worker_thread_num())},
                     true);
}
} // namespace einsums::threads::detail
