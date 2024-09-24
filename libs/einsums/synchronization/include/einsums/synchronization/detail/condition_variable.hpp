//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/concurrency/cache_line_data.hpp>
#include <einsums/concurrency/spinlock.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/execution_base/agent_ref.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/thread_support/atomic_count.hpp>
#include <einsums/timing/steady_clock.hpp>

#include <boost/intrusive/slist.hpp>
#include <cstddef>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {
class condition_variable {
  public:
    EINSUMS_NON_COPYABLE(condition_variable);

  private:
    using mutex_type = einsums::concurrency::detail::spinlock;

  private:
    // define data structures needed for intrusive slist container used for
    // the queues
    struct queue_entry {
        using hook_type = boost::intrusive::slist_member_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>>;

        queue_entry(einsums::execution::detail::agent_ref ctx, void *q) : ctx_(ctx), q_(q) {}

        einsums::execution::detail::agent_ref ctx_;
        void                                 *q_;
        hook_type                             slist_hook_;
    };

    using slist_option_type = boost::intrusive::member_hook<queue_entry, queue_entry::hook_type, &queue_entry::slist_hook_>;

    using queue_type = boost::intrusive::slist<queue_entry, slist_option_type, boost::intrusive::cache_last<true>,
                                               boost::intrusive::constant_time_size<true>>;

    struct reset_queue_entry {
        reset_queue_entry(queue_entry &e, queue_type &q) : e_(e), last_(q.last()) {}

        ~reset_queue_entry() {
            if (e_.ctx_) {
                queue_type *q = static_cast<queue_type *>(e_.q_);
                q->erase(last_); // remove entry from queue
            }
        }

        queue_entry               &e_;
        queue_type::const_iterator last_;
    };

  public:
    EINSUMS_EXPORT condition_variable();

    EINSUMS_EXPORT ~condition_variable();

    EINSUMS_EXPORT bool empty(std::unique_lock<mutex_type> const &lock) const;

    EINSUMS_EXPORT std::size_t size(std::unique_lock<mutex_type> const &lock) const;

    // Return false if no more threads are waiting (returns true if queue
    // is non-empty).
    EINSUMS_EXPORT bool notify_one(std::unique_lock<mutex_type> lock, execution::thread_priority priority, error_code &ec = throws);

    EINSUMS_EXPORT void notify_all(std::unique_lock<mutex_type> lock, execution::thread_priority priority, error_code &ec = throws);

    bool notify_one(std::unique_lock<mutex_type> lock, error_code &ec = throws) {
        return notify_one(std::move(lock), execution::thread_priority::default_, ec);
    }

    void notify_all(std::unique_lock<mutex_type> lock, error_code &ec = throws) {
        return notify_all(std::move(lock), execution::thread_priority::default_, ec);
    }

    EINSUMS_EXPORT void abort_all(std::unique_lock<mutex_type> lock);

    EINSUMS_EXPORT einsums::threads::detail::thread_restart_state wait(std::unique_lock<mutex_type> &lock, char const *description,
                                                                       error_code &ec = throws);

    einsums::threads::detail::thread_restart_state wait(std::unique_lock<mutex_type> &lock, error_code &ec = throws) {
        return wait(lock, "condition_variable::wait", ec);
    }

    EINSUMS_EXPORT einsums::threads::detail::thread_restart_state wait_until(std::unique_lock<mutex_type>             &lock,
                                                                             einsums::chrono::steady_time_point const &abs_time,
                                                                             char const *description, error_code &ec = throws);

    einsums::threads::detail::thread_restart_state wait_until(std::unique_lock<mutex_type>             &lock,
                                                              einsums::chrono::steady_time_point const &abs_time, error_code &ec = throws) {
        return wait_until(lock, abs_time, "condition_variable::wait_until", ec);
    }

    einsums::threads::detail::thread_restart_state wait_for(std::unique_lock<mutex_type>           &lock,
                                                            einsums::chrono::steady_duration const &rel_time, char const *description,
                                                            error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), description, ec);
    }

    einsums::threads::detail::thread_restart_state wait_for(std::unique_lock<mutex_type>           &lock,
                                                            einsums::chrono::steady_duration const &rel_time, error_code &ec = throws) {
        return wait_until(lock, rel_time.from_now(), "condition_variable::wait_for", ec);
    }

  private:
    template <typename Mutex>
    void abort_all(std::unique_lock<Mutex> lock);

    // re-add the remaining items to the original queue
    EINSUMS_EXPORT void prepend_entries(std::unique_lock<mutex_type> &lock, queue_type &queue);

  private:
    queue_type queue_;
};

///////////////////////////////////////////////////////////////////////////
struct condition_variable_data;

EINSUMS_EXPORT void intrusive_ptr_add_ref(condition_variable_data *p);
EINSUMS_EXPORT void intrusive_ptr_release(condition_variable_data *p);

struct condition_variable_data {
    using mutex_type = einsums::concurrency::detail::spinlock;

    condition_variable_data() : count_(1) {}

    einsums::concurrency::detail::cache_aligned_data_derived<mutex_type>                 mtx_;
    einsums::concurrency::detail::cache_aligned_data_derived<detail::condition_variable> cond_;

  private:
    friend EINSUMS_EXPORT void intrusive_ptr_add_ref(condition_variable_data *);
    friend EINSUMS_EXPORT void intrusive_ptr_release(condition_variable_data *);

    einsums::detail::atomic_count count_;
};

} // namespace einsums::detail
