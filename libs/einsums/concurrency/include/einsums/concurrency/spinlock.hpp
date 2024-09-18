//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/execution_base/this_thread.hpp>
#include <einsums/itt_notify.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/thread_support/spinlock.hpp>

#include <atomic>

namespace einsums::concurrency::detail {
struct spinlock {
  public:
    EINSUMS_NON_COPYABLE(spinlock);

  private:
    std::atomic<bool> v_;

  public:
    spinlock(char const *const desc = "einsums::concurrency::detail::spinlock") : v_(false) { itt_sync_create(this, desc, ""); }

    ~spinlock() { itt_sync_destroy(this); }

    void lock() {
        itt_sync_prepare(this);

        // Checking for the value in is_locked() ensures that
        // acquire_lock is only called when is_locked computes
        // to false. This way we spin only on a load operation
        // which minimizes false sharing that comes with an
        // exchange operation.
        // Consider the following cases:
        // 1. Only one thread wants access critical section:
        //      is_locked() -> false; computes acquire_lock()
        //      acquire_lock() -> false (new value set to true)
        //      Thread acquires the lock and moves to critical
        //      section.
        // 2. Two threads simultaneously access critical section:
        //      Thread 1: is_locked() || acquire_lock() -> false
        //      Thread 1 acquires the lock and moves to critical
        //      section.
        //      Thread 2: is_locked() -> true; execution enters
        //      inside while without computing acquire_lock().
        //      Thread 2 yields while is_locked() computes to
        //      false. Then it retries doing is_locked() -> false
        //      followed by an acquire_lock() operation.
        //      The above order can be changed arbitrarily but
        //      the nature of execution will still remain the
        //      same.
        do {
            util::yield_while([this] { return is_locked(); }, "einsums::concurrency::detail::spinlock::lock", false);
        } while (!acquire_lock());

        itt_sync_acquired(this);
        util::register_lock(this);
    }

    bool try_lock() {
        itt_sync_prepare(this);
        bool r = acquire_lock(); //-V707

        if (r) {
            itt_sync_acquired(this);
            util::register_lock(this);
            return true;
        }

        itt_sync_cancel(this);
        return false;
    }

    void unlock() {
        itt_sync_releasing(this);
        relinquish_lock();

        itt_sync_released(this);
        util::unregister_lock(this);
    }

  private:
    // returns whether the mutex has been acquired
    EINSUMS_FORCEINLINE bool acquire_lock() { return !v_.exchange(true, std::memory_order_acquire); }

    // relinquish lock
    EINSUMS_FORCEINLINE void relinquish_lock() { v_.store(false, std::memory_order_release); }

    EINSUMS_FORCEINLINE bool is_locked() const { return v_.load(std::memory_order_relaxed); }
};
} // namespace einsums::concurrency::detail
