//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
# pragma warning(push)
# pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

    class counting_semaphore
    {
    private:
        using mutex_type = einsums::concurrency::detail::spinlock;

    public:
        EINSUMS_EXPORT counting_semaphore(std::ptrdiff_t value = 0);
        EINSUMS_EXPORT ~counting_semaphore();

        EINSUMS_EXPORT void wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count);

        EINSUMS_EXPORT bool wait_until(std::unique_lock<mutex_type>& l,
            einsums::chrono::steady_time_point const& abs_time, std::ptrdiff_t count);

        EINSUMS_EXPORT bool try_wait(std::unique_lock<mutex_type>& l, std::ptrdiff_t count = 1);

        EINSUMS_EXPORT bool try_acquire(std::unique_lock<mutex_type>& l);

        EINSUMS_EXPORT void signal(std::unique_lock<mutex_type> l, std::ptrdiff_t count);

        EINSUMS_EXPORT std::ptrdiff_t signal_all(std::unique_lock<mutex_type> l);

    private:
        std::ptrdiff_t value_;
        einsums::detail::condition_variable cond_;
    };
}    // namespace einsums::detail

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
# pragma warning(pop)
#endif
