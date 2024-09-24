//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/invoke.hpp>
#include <einsums/synchronization/event.hpp>

#include <atomic>
#include <utility>

namespace einsums {
struct once_flag {
  public:
    EINSUMS_NON_COPYABLE(once_flag);

  public:
    once_flag() noexcept : status_(0) {}

  private:
    std::atomic<long>            status_;
    einsums::experimental::event event_;

    template <typename F, typename... Args>
    friend void call_once(once_flag &flag, F &&f, Args &&...args);
};

#define EINSUMS_ONCE_INIT ::einsums::once_flag()

///////////////////////////////////////////////////////////////////////////
template <typename F, typename... Args>
void call_once(once_flag &flag, F &&f, Args &&...args) {
    // Try for a quick win: if the procedure has already been called
    // just skip through:
    long const function_complete_flag_value = 0xc157'30e2;
    long const running_value                = 0x7f07'25e3;

    while (flag.status_.load(std::memory_order_acquire) != function_complete_flag_value) {
        long status = 0;
        if (flag.status_.compare_exchange_strong(status, running_value)) {
            try {
                // reset event to ensure its usability in case the
                // wrapped function was throwing an exception before
                flag.event_.reset();

                EINSUMS_INVOKE(std::forward<F>(f), std::forward<Args>(args)...);

                // set status to done, release waiting threads
                flag.status_.store(function_complete_flag_value);
                flag.event_.set();
                break;
            } catch (...) {
                // reset status to initial, release waiting threads
                flag.status_.store(0);
                flag.event_.set();

                throw;
            }
        }

        // we're done if function was called
        if (status == function_complete_flag_value)
            break;

        // wait for the function finish executing
        flag.event_.wait();
    }
}
} // namespace einsums
