//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#include <einsums/assert.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>
#include <einsums/synchronization/detail/counting_semaphore.hpp>
#include <einsums/thread_support/assert_owns_lock.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

    counting_semaphore::counting_semaphore(std::ptrdiff_t value)
      : value_(value)
    {
    }

    counting_semaphore::~counting_semaphore() = default;

    void counting_semaphore::wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        while (value_ < count) { cond_.wait(l, "counting_semaphore::wait"); }
        value_ -= count;
    }

    bool counting_semaphore::wait_until(std::unique_lock<mutex_type>& l,
        einsums::chrono::steady_time_point const& abs_time, std::ptrdiff_t count)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        while (value_ < count)
        {
            // return false if unblocked by timeout expiring
            if (cond_.wait_until(l, abs_time, "counting_semaphore::wait_until") !=
                einsums::threads::detail::thread_restart_state::unknown)
            {
                return false;
            }
        }
        value_ -= count;
        return true;
    }

    bool counting_semaphore::try_wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        if (!(value_ < count))
        {
            // enter wait_locked only if there are sufficient credits
            // available
            wait(l, count);
            return true;
        }
        return false;
    }

    bool counting_semaphore::try_acquire(std::unique_lock<mutex_type>& l)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        if (value_ >= 1)
        {
            --value_;
            return true;
        }
        return false;
    }

    void counting_semaphore::signal(std::unique_lock<mutex_type> l, std::ptrdiff_t count)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        mutex_type* mtx = l.mutex();

        // release no more threads than we get resources
        value_ += count;
        for (std::int64_t i = 0; value_ >= 0 && i < count; ++i)
        {
            // notify_one() returns false if no more threads are
            // waiting
            if (!cond_.notify_one(std::move(l))) break;

            l = std::unique_lock<mutex_type>(*mtx);
        }
    }

    std::ptrdiff_t counting_semaphore::signal_all(std::unique_lock<mutex_type> l)
    {
        EINSUMS_ASSERT_OWNS_LOCK(l);

        std::ptrdiff_t count = static_cast<std::ptrdiff_t>(cond_.size(l));
        signal(std::move(l), count);
        return count;
    }
}    // namespace einsums::detail
