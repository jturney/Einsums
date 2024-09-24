//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>
#include <einsums/assert.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/execution_base/agent_ref.hpp>
#include <einsums/execution_base/this_thread.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {
    /// An exclusive-ownership recursive mutex which implements Boost.Thread's
    /// TimedLockable concept.
    template <typename Mutex = einsums::concurrency::detail::spinlock>
    struct recursive_mutex_impl
    {
    public:
        EINSUMS_NON_COPYABLE(recursive_mutex_impl);

    private:
        std::atomic<std::uint64_t> recursion_count;
        std::atomic<einsums::execution::detail::agent_ref> locking_context;
        Mutex mtx;

    public:
        recursive_mutex_impl(char const* const desc = "recursive_mutex_impl")
          : recursion_count(0)
          , mtx(desc)
        {
        }

        /// Attempts to acquire ownership of the \a recursive_mutex.
        /// Never blocks.
        ///
        /// \returns \a true if ownership was acquired; otherwise, \a false.
        ///
        /// \throws Never throws.
        bool try_lock()
        {
            auto ctx = einsums::execution::this_thread::detail::agent();
            EINSUMS_ASSERT(ctx);

            return try_recursive_lock(ctx) || try_basic_lock(ctx);
        }

        /// Acquires ownership of the \a recursive_mutex. Suspends the
        /// current einsums-thread if ownership cannot be obtained immediately.
        ///
        /// \throws Throws \a einsums#bad_parameter if an error occurs while
        ///         suspending. Throws \a einsums#yield_aborted if the mutex is
        ///         destroyed while suspended. Throws \a einsums#null_thread_id if
        ///         called outside of a einsums-thread.
        void lock()
        {
            auto ctx = einsums::execution::this_thread::detail::agent();
            EINSUMS_ASSERT(ctx);

            if (!try_recursive_lock(ctx))
            {
                mtx.lock();
                locking_context.exchange(ctx);
                einsums::util::ignore_lock(&mtx);
                einsums::util::register_lock(this);
                recursion_count.store(1);
            }
        }

        /// Release ownership of the \a recursive_mutex.
        ///
        /// \throws Throws \a einsums#bad_parameter if an error occurs while
        ///         releasing the mutex. Throws \a einsums#null_thread_id if called
        ///         outside of a einsums-thread.
        void unlock()
        {
            if (0 == --recursion_count)
            {
                locking_context.exchange(einsums::execution::detail::agent_ref());
                einsums::util::unregister_lock(this);
                einsums::util::reset_ignored(&mtx);
                mtx.unlock();
            }
        }

    private:
        bool try_recursive_lock(einsums::execution::detail::agent_ref current_context)
        {
            if (locking_context.load(std::memory_order_acquire) == current_context)
            {
                if (++recursion_count == 1) einsums::util::register_lock(this);
                return true;
            }
            return false;
        }

        bool try_basic_lock(einsums::execution::detail::agent_ref current_context)
        {
            if (mtx.try_lock())
            {
                locking_context.exchange(current_context);
                einsums::util::ignore_lock(&mtx);
                einsums::util::register_lock(this);
                recursion_count.store(1);
                return true;
            }
            return false;
        }
    };
}    // namespace einsums::detail
