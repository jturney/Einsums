//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/itt_notify.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/synchronization/condition_variable.hpp>
#include <einsums/synchronization/mutex.hpp>
#include <einsums/threading_base/thread_data.hpp>
#include <einsums/timing/steady_clock.hpp>
#include <einsums/type_support/unused.hpp>

#include <mutex>
#include <utility>

namespace einsums {
///////////////////////////////////////////////////////////////////////////
mutex::mutex(char const *const description) : owner_id_(threads::detail::invalid_thread_id) {
    itt_sync_create(this, "lcos::local::mutex", description);
    itt_sync_rename(this, "lcos::local::mutex");
}

mutex::~mutex() {
    itt_sync_destroy(this);
}

void mutex::lock(char const *description, error_code &ec) {
    EINSUMS_ASSERT(threads::detail::get_self_ptr() != nullptr);

    itt_sync_prepare(this);
    std::unique_lock<mutex_type> l(mtx_);

    threads::detail::thread_id_type self_id = einsums::threads::detail::get_self_id();
    if (owner_id_ == self_id) {
        itt_sync_cancel(this);
        l.unlock();
        EINSUMS_THROWS_IF(ec, einsums::error::deadlock, "{}: {}", description, "The calling thread already owns the mutex");
        return;
    }

    while (owner_id_ != threads::detail::invalid_thread_id) {
        cond_.wait(l, ec);
        if (ec) {
            itt_sync_cancel(this);
            return;
        }
    }

    util::register_lock(this);
    itt_sync_acquired(this);
    owner_id_ = self_id;
}

bool mutex::try_lock(char const * /* description */, error_code & /* ec */) {
    EINSUMS_ASSERT(threads::detail::get_self_ptr() != nullptr);

    itt_sync_prepare(this);
    std::unique_lock<mutex_type> l(mtx_);

    if (owner_id_ != threads::detail::invalid_thread_id) {
        itt_sync_cancel(this);
        return false;
    }

    threads::detail::thread_id_type self_id = einsums::threads::detail::get_self_id();
    util::register_lock(this);
    itt_sync_acquired(this);
    owner_id_ = self_id;
    return true;
}

void mutex::unlock(error_code &ec) {
    EINSUMS_ASSERT(threads::detail::get_self_ptr() != nullptr);

    itt_sync_releasing(this);
    // Unregister lock early as the lock guard below may suspend.
    util::unregister_lock(this);
    std::unique_lock<mutex_type> l(mtx_);

    threads::detail::thread_id_type self_id = einsums::threads::detail::get_self_id();
    if (EINSUMS_UNLIKELY(owner_id_ != self_id)) {
        l.unlock();
        EINSUMS_THROWS_IF(ec, einsums::error::lock_error, "mutex::unlock", "The calling thread does not own the mutex");
        return;
    }

    itt_sync_released(this);
    owner_id_ = threads::detail::invalid_thread_id;

    {
        [[maybe_unused]] util::ignore_while_checking il(&l);

        cond_.notify_one(std::move(l), execution::thread_priority::boost, ec);
    }
}

///////////////////////////////////////////////////////////////////////////
timed_mutex::timed_mutex(char const *const description) : mutex(description) {
}

timed_mutex::~timed_mutex() {
}

bool timed_mutex::try_lock_until(einsums::chrono::steady_time_point const &abs_time, char const * /* description */, error_code &ec) {
    EINSUMS_ASSERT(threads::detail::get_self_ptr() != nullptr);

    itt_sync_prepare(this);
    std::unique_lock<mutex_type> l(mtx_);

    threads::detail::thread_id_type self_id = einsums::threads::detail::get_self_id();
    if (owner_id_ != threads::detail::invalid_thread_id) {
        einsums::threads::detail::thread_restart_state const reason = cond_.wait_until(l, abs_time, ec);
        if (ec) {
            itt_sync_cancel(this);
            return false;
        }

        if (reason == einsums::threads::detail::thread_restart_state::timeout) //-V110
        {
            itt_sync_cancel(this);
            return false;
        }

        if (owner_id_ != threads::detail::invalid_thread_id) //-V110
        {
            itt_sync_cancel(this);
            return false;
        }
    }

    util::register_lock(this);
    itt_sync_acquired(this);
    owner_id_ = self_id;
    return true;
}
} // namespace einsums
