//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/coroutines/coroutine_fwd.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/threading_base.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>
#include <einsums/timing/steady_clock.hpp>

namespace einsums {
    ///////////////////////////////////////////////////////////////////////////
    class mutex
    {
    public:
        EINSUMS_NON_COPYABLE(mutex);

    protected:
        using mutex_type = einsums::concurrency::detail::spinlock;

    public:
        EINSUMS_EXPORT mutex(char const* const description = "");

        EINSUMS_EXPORT ~mutex();

        EINSUMS_EXPORT void lock(char const* description, error_code& ec = throws);

        void lock(error_code& ec = throws) { return lock("mutex::lock", ec); }

        EINSUMS_EXPORT bool try_lock(char const* description, error_code& ec = throws);

        bool try_lock(error_code& ec = throws) { return try_lock("mutex::try_lock", ec); }

        EINSUMS_EXPORT void unlock(error_code& ec = throws);

    protected:
        mutable mutex_type mtx_;
        threads::detail::thread_id_type owner_id_;
        einsums::detail::condition_variable cond_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class timed_mutex : private mutex
    {
    public:
        EINSUMS_NON_COPYABLE(timed_mutex);

    public:
        EINSUMS_EXPORT timed_mutex(char const* const description = "");

        EINSUMS_EXPORT ~timed_mutex();

        using mutex::lock;
        using mutex::try_lock;
        using mutex::unlock;

        EINSUMS_EXPORT bool try_lock_until(einsums::chrono::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        bool try_lock_until(
            einsums::chrono::steady_time_point const& abs_time, error_code& ec = throws)
        {
            return try_lock_until(abs_time, "mutex::try_lock_until", ec);
        }

        bool try_lock_for(einsums::chrono::steady_duration const& rel_time, char const* description,
            error_code& ec = throws)
        {
            return try_lock_until(rel_time.from_now(), description, ec);
        }

        bool try_lock_for(einsums::chrono::steady_duration const& rel_time, error_code& ec = throws)
        {
            return try_lock_for(rel_time, "mutex::try_lock_for", ec);
        }
    };
}    // namespace einsums
