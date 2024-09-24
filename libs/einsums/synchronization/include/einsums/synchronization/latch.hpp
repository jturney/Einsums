//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

/// \file einsums/synchronization/latch.hpp

#pragma once

#include <einsums/assert.hpp>
#include <einsums/concurrency/cache_line_data.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>
#include <einsums/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace einsums {
/// Latches are a thread coordination mechanism that allow one or more
/// threads to block until an operation is completed. An individual latch
/// is a singleuse object; once the operation has been completed, the latch
/// cannot be reused.
class latch {
  public:
    EINSUMS_NON_COPYABLE(latch);

  protected:
    using mutex_type = einsums::concurrency::detail::spinlock;

  public:
    /// Initialize the latch
    ///
    /// Requires: count >= 0.
    /// Synchronization: None
    /// Postconditions: counter_ == count.
    ///
    explicit latch(std::ptrdiff_t count) : mtx_(), cond_(), counter_(count), notified_(count == 0) {}

    /// Requires: No threads are blocked at the synchronization point.
    ///
    /// \note May be called even if some threads have not yet returned
    ///       from wait() or count_down_and_wait(), provided that counter_
    ///       is 0.
    /// \note The destructor might not return until all threads have exited
    ///       wait() or count_down_and_wait().
    /// \note It is the caller's responsibility to ensure that no other
    ///       thread enters wait() after one thread has called the
    ///       destructor. This may require additional coordination.
#if defined(EINSUMS_DEBUG)
    ~latch() { EINSUMS_ASSERT(counter_ == 0); }
#else
    ~latch() = default;
#endif

    /// Returns:        The maximum value of counter that the implementation
    ///                 supports.
    static constexpr std::ptrdiff_t(max)() noexcept { return (std::numeric_limits<std::ptrdiff_t>::max)(); }

    /// Decrements counter_ by n. Does not block.
    ///
    /// Requires: counter_ >= n and n >= 0.
    ///
    /// Synchronization: Synchronizes with all calls that block on this
    /// latch and with all try_wait calls on this latch that return true .
    ///
    /// \throws Nothing.
    ///
    void count_down(std::ptrdiff_t update) {
        EINSUMS_ASSERT(update >= 0);

        std::ptrdiff_t new_count = (counter_ -= update);
        EINSUMS_ASSERT(new_count >= 0);

        if (new_count == 0) {
            std::unique_lock l(mtx_.data_);
            notified_ = true;

            // Note: we use notify_one repeatedly instead of notify_all as we
            // know that our implementation of condition_variable::notify_one
            // relinquishes the lock before resuming the waiting thread
            // which avoids suspension of this thread when it tries to
            // re-lock the mutex while exiting from condition_variable::wait
            while (cond_.data_.notify_one(std::move(l), execution::thread_priority::boost)) {
                l = std::unique_lock(mtx_.data_);
            }
        }
    }

    /// Returns:        With very low probability false. Otherwise
    ///                 counter == 0.
    bool try_wait() const noexcept { return counter_.load(std::memory_order_acquire) == 0; }

    /// If counter_ is 0, returns immediately. Otherwise, blocks the
    /// calling thread at the synchronization point until counter_
    /// reaches 0.
    ///
    /// \throws Nothing.
    ///
    void wait() const {
        std::unique_lock l(mtx_.data_);
        if (counter_.load(std::memory_order_relaxed) > 0 || !notified_) {
            cond_.data_.wait(l, "einsums::latch::wait");

            EINSUMS_ASSERT(counter_.load(std::memory_order_relaxed) == 0);
            EINSUMS_ASSERT(notified_);
        }
    }

    /// Effects: Equivalent to:
    ///             count_down(update);
    ///             wait();
    void arrive_and_wait(std::ptrdiff_t update = 1) {
        EINSUMS_ASSERT(update >= 0);

        std::unique_lock l(mtx_.data_);

        std::ptrdiff_t old_count = counter_.fetch_sub(update, std::memory_order_relaxed);
        EINSUMS_ASSERT(old_count >= update);

        if (old_count > update) {
            cond_.data_.wait(l, "einsums::latch::arrive_and_wait");

            EINSUMS_ASSERT(counter_.load(std::memory_order_relaxed) == 0);
            EINSUMS_ASSERT(notified_);
        } else {
            notified_ = true;

            // Note: we use notify_one repeatedly instead of notify_all as we
            // know that our implementation of condition_variable::notify_one
            // relinquishes the lock before resuming the waiting thread
            // which avoids suspension of this thread when it tries to
            // re-lock the mutex while exiting from condition_variable::wait
            while (cond_.data_.notify_one(std::move(l), execution::thread_priority::boost)) {
                l = std::unique_lock(mtx_.data_);
            }
        }
    }

  protected:
    mutable einsums::concurrency::detail::cache_line_data<mutex_type>                          mtx_;
    mutable einsums::concurrency::detail::cache_line_data<einsums::detail::condition_variable> cond_;
    std::atomic<std::ptrdiff_t>                                                                counter_;
    bool                                                                                       notified_;
};

} // namespace einsums
