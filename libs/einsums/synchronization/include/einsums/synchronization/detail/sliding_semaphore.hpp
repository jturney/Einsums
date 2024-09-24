//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>

#include <cstdint>
#include <mutex>
#include <utility>

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
# pragma warning(push)
# pragma warning(disable : 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

    class sliding_semaphore
    {
    private:
        using mutex_type = einsums::concurrency::detail::spinlock;

    public:
        EINSUMS_EXPORT sliding_semaphore(std::int64_t max_difference, std::int64_t lower_limit);
        EINSUMS_EXPORT ~sliding_semaphore();

        EINSUMS_EXPORT void set_max_difference(
            std::unique_lock<mutex_type>& l, std::int64_t max_difference, std::int64_t lower_limit);

        EINSUMS_EXPORT void wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        EINSUMS_EXPORT bool try_wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit);

        EINSUMS_EXPORT void signal(std::unique_lock<mutex_type> l, std::int64_t lower_limit);

        EINSUMS_EXPORT std::int64_t signal_all(std::unique_lock<mutex_type> l);

    private:
        std::int64_t max_difference_;
        std::int64_t lower_limit_;
        einsums::detail::condition_variable cond_;
    };
}    // namespace einsums::detail

#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
# pragma warning(pop)
#endif
