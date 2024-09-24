//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/memory.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>
#include <einsums/synchronization/mutex.hpp>
#include <einsums/synchronization/stop_token.hpp>
#include <einsums/thread_support/assert_owns_lock.hpp>
#include <einsums/thread_support/unlock_guard.hpp>
#include <einsums/timing/steady_clock.hpp>
#include <einsums/type_support/unused.hpp>

#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace einsums {
enum class cv_status { no_timeout, timeout, error };

class condition_variable {
  private:
    using mutex_type = detail::condition_variable_data::mutex_type;
    using data_type  = einsums::memory::intrusive_ptr<detail::condition_variable_data>;

  public:
    condition_variable() : data_(data_type(new detail::condition_variable_data, false)) {}

    // Preconditions: There is no thread blocked on *this. [Note: That is,
    //      all threads have been notified; they could subsequently block
    //      on the lock specified in the wait.This relaxes the usual rules,
    //      which would have required all wait calls to happen before
    //      destruction.Only the notification to unblock the wait needs to
    //      happen before destruction.The user should take care to ensure
    //      that no threads wait on *this once the destructor has been
    //      started, especially when the waiting threads are calling the
    //      wait functions in a loop or using the overloads of wait,
    //      wait_for, or wait_until that take a predicate. end note]
    //
    // IOW, ~condition_variable() can execute before a signaled thread
    // returns from a wait. If this happens with condition_variable, that
    // waiting thread will attempt to lock the destructed mutex.
    // To fix this, there must be shared ownership of the data members
    // between the condition_variable_any object and the member functions
    // wait (wait_for, etc.).
    ~condition_variable() = default;

    void notify_one(error_code &ec = throws) const {
        std::unique_lock<mutex_type> l(data_->mtx_);
        data_->cond_.notify_one(std::move(l), ec);
    }

    void notify_all(error_code &ec = throws) const {
        std::unique_lock<mutex_type> l(data_->mtx_);
        data_->cond_.notify_all(std::move(l), ec);
    }

    template <typename Mutex>
    void wait(std::unique_lock<Mutex> &lock, error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        auto data = data_; // keep data alive

        [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

        std::unique_lock<mutex_type>                             l(data->mtx_);
        ::einsums::detail::unlock_guard<std::unique_lock<Mutex>> unlock(lock);

        // The following ensures that the inner lock will be unlocked
        // before the outer to avoid deadlock (fixes issue #3608)
        std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

        data->cond_.wait(l, ec);
    }

    template <typename Mutex, typename Predicate>
    void wait(std::unique_lock<Mutex> &lock, Predicate pred, error_code & /*ec*/ = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        while (!pred()) {
            wait(lock);
        }
    }

    template <typename Mutex>
    cv_status wait_until(std::unique_lock<Mutex> &lock, einsums::chrono::steady_time_point const &abs_time, error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        auto data = data_; // keep data alive

        [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

        std::unique_lock<mutex_type>                             l(data->mtx_);
        ::einsums::detail::unlock_guard<std::unique_lock<Mutex>> unlock(lock);

        // The following ensures that the inner lock will be unlocked
        // before the outer to avoid deadlock (fixes issue #3608)
        std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

        einsums::threads::detail::thread_restart_state const reason = data->cond_.wait_until(l, abs_time, ec);

        if (ec)
            return cv_status::error;

        // if the timer has hit, the waiting period timed out
        return (reason == einsums::threads::detail::thread_restart_state::timeout) ? //-V110
                   cv_status::timeout
                                                                                   : cv_status::no_timeout;
    }

    template <typename Mutex, typename Predicate>
    bool wait_until(std::unique_lock<Mutex> &lock, einsums::chrono::steady_time_point const &abs_time, Predicate pred,
                    error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        while (!pred()) {
            if (wait_until(lock, abs_time, ec) == cv_status::timeout)
                return pred();
        }
        return true;
    }

    template <typename Mutex>
    cv_status wait_for(std::unique_lock<Mutex> &lock, einsums::chrono::steady_duration const &rel_time, error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), ec);
    }

    template <typename Mutex, typename Predicate>
    bool wait_for(std::unique_lock<Mutex> &lock, einsums::chrono::steady_duration const &rel_time, Predicate pred,
                  error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), EINSUMS_MOVE(pred), ec);
    }

  private:
    einsums::concurrency::detail::cache_aligned_data_derived<data_type> data_;
};

///////////////////////////////////////////////////////////////////////////
class condition_variable_any {
  private:
    using mutex_type = detail::condition_variable_data::mutex_type;
    using data_type  = einsums::memory::intrusive_ptr<detail::condition_variable_data>;

  public:
    condition_variable_any() : data_(data_type(new detail::condition_variable_data, false)) {}

    // Preconditions: There is no thread blocked on *this. [Note: That is,
    //      all threads have been notified; they could subsequently block
    //      on the lock specified in the wait.This relaxes the usual rules,
    //      which would have required all wait calls to happen before
    //      destruction.Only the notification to unblock the wait needs to
    //      happen before destruction.The user should take care to ensure
    //      that no threads wait on *this once the destructor has been
    //      started, especially when the waiting threads are calling the
    //      wait functions in a loop or using the overloads of wait,
    //      wait_for, or wait_until that take a predicate. end note]
    //
    // IOW, ~condition_variable_any() can execute before a signaled thread
    // returns from a wait. If this happens with condition_variable, that
    // waiting thread will attempt to lock the destructed mutex.
    // To fix this, there must be shared ownership of the data members
    // between the condition_variable_any object and the member functions
    // wait (wait_for, etc.).
    ~condition_variable_any() = default;

    void notify_one(error_code &ec = throws) const {
        std::unique_lock<mutex_type> l(data_->mtx_);
        data_->cond_.notify_one(std::move(l), ec);
    }

    void notify_all(error_code &ec = throws) const {
        std::unique_lock<mutex_type> l(data_->mtx_);
        data_->cond_.notify_all(std::move(l), ec);
    }

    template <typename Lock>
    void wait(Lock &lock, error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        auto data = data_; // keep data alive

        [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

        std::unique_lock<mutex_type>          l(data->mtx_);
        ::einsums::detail::unlock_guard<Lock> unlock(lock);

        // The following ensures that the inner lock will be unlocked
        // before the outer to avoid deadlock (fixes issue #3608)
        std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

        data->cond_.wait(l, ec);
    }

    template <typename Lock, typename Predicate>
    void wait(Lock &lock, Predicate pred, error_code & /* ec */ = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        while (!pred()) {
            wait(lock);
        }
    }

    template <typename Lock>
    cv_status wait_until(Lock &lock, einsums::chrono::steady_time_point const &abs_time, error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        auto data = data_; // keep data alive

        [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

        std::unique_lock<mutex_type>          l(data->mtx_);
        ::einsums::detail::unlock_guard<Lock> unlock(lock);

        // The following ensures that the inner lock will be unlocked
        // before the outer to avoid deadlock (fixes issue #3608)
        std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

        einsums::threads::detail::thread_restart_state const reason = data->cond_.wait_until(l, abs_time, ec);

        if (ec)
            return cv_status::error;

        // if the timer has hit, the waiting period timed out
        return (reason == einsums::threads::detail::thread_restart_state::timeout) ? //-V110
                   cv_status::timeout
                                                                                   : cv_status::no_timeout;
    }

    template <typename Lock, typename Predicate>
    bool wait_until(Lock &lock, einsums::chrono::steady_time_point const &abs_time, Predicate pred, error_code &ec = throws) {
        EINSUMS_ASSERT_OWNS_LOCK(lock);

        while (!pred()) {
            if (wait_until(lock, abs_time, ec) == cv_status::timeout)
                return pred();
        }
        return true;
    }

    template <typename Lock>
    cv_status wait_for(Lock &lock, einsums::chrono::steady_duration const &rel_time, error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), ec);
    }

    template <typename Lock, typename Predicate>
    bool wait_for(Lock &lock, einsums::chrono::steady_duration const &rel_time, Predicate pred, error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), EINSUMS_MOVE(pred), ec);
    }

    // 32.6.4.2, interruptible waits
    template <typename Lock, typename Predicate>
    bool wait(Lock &lock, stop_token stoken, Predicate pred, error_code &ec = throws) {
        if (stoken.stop_requested()) {
            return pred();
        }

        auto data = data_; // keep data alive

        auto f = [&data, &ec] {
            std::unique_lock<mutex_type> l(data->mtx_);
            data->cond_.notify_all(std::move(l), ec);
        };
        stop_callback<decltype(f)> cb(stoken, EINSUMS_MOVE(f));

        while (!pred()) {
            [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

            std::unique_lock<mutex_type> l(data->mtx_);
            if (stoken.stop_requested()) {
                // pred() has already evaluated to false since we last
                // acquired lock
                return false;
            }

            ::einsums::detail::unlock_guard<Lock> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

            data->cond_.wait(l, ec);
        }

        return true;
    }

    template <typename Lock, typename Predicate>
    bool wait_until(Lock &lock, stop_token stoken, einsums::chrono::steady_time_point const &abs_time, Predicate pred,
                    error_code &ec = throws) {
        if (stoken.stop_requested()) {
            return pred();
        }

        auto data = data_; // keep data alive

        auto f = [&data, &ec] {
            std::unique_lock<mutex_type> l(data->mtx_);
            data->cond_.notify_all(std::move(l), ec);
        };
        stop_callback<decltype(f)> cb(stoken, EINSUMS_MOVE(f));

        while (!pred()) {
            bool should_stop;
            {
                [[maybe_unused]] util::ignore_all_while_checking ignore_lock;

                std::unique_lock<mutex_type> l(data->mtx_);
                if (stoken.stop_requested()) {
                    // pred() has already evaluated to false since we last
                    // acquired lock.
                    return false;
                }

                ::einsums::detail::unlock_guard<Lock> unlock(lock);

                // The following ensures that the inner lock will be unlocked
                // before the outer to avoid deadlock (fixes issue #3608)
                std::lock_guard<std::unique_lock<mutex_type>> unlock_next(l, std::adopt_lock);

                einsums::threads::detail::thread_restart_state const reason = data->cond_.wait_until(l, abs_time, ec);

                if (ec)
                    return false;

                should_stop = (reason == einsums::threads::detail::thread_restart_state::timeout) || stoken.stop_requested();
            }

            if (should_stop) {
                return pred();
            }
        }
        return true;
    }

    template <typename Lock, typename Predicate>
    bool wait_for(Lock &lock, stop_token stoken, einsums::chrono::steady_duration const &rel_time, Predicate pred,
                  error_code &ec = throws) {
        return wait_until(lock, stoken, rel_time.from_now(), EINSUMS_MOVE(pred), ec);
    }

  private:
    einsums::concurrency::detail::cache_aligned_data_derived<data_type> data_;
};
} // namespace einsums
