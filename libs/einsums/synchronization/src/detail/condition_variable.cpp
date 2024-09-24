//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/execution_base/this_thread.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/memory.hpp>
#include <einsums/synchronization/detail/condition_variable.hpp>
#include <einsums/synchronization/no_mutex.hpp>
#include <einsums/thread_support/unlock_guard.hpp>
#include <einsums/threading_base/thread_helpers.hpp>
#include <einsums/timing/steady_clock.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <exception>
#include <mutex>
#include <utility>

namespace einsums::detail {

///////////////////////////////////////////////////////////////////////////
condition_variable::condition_variable() {
}

condition_variable::~condition_variable() {
    if (!queue_.empty()) {
        EINSUMS_LOG(err, "~condition_variable: queue is not empty, aborting threads");

        einsums::no_mutex                   no_mtx;
        std::unique_lock<einsums::no_mutex> lock(no_mtx);
        abort_all<einsums::no_mutex>(std::move(lock));
    }
}

bool condition_variable::empty([[maybe_unused]] std::unique_lock<mutex_type> const &lock) const {
    EINSUMS_ASSERT(lock.owns_lock());

    return queue_.empty();
}

std::size_t condition_variable::size([[maybe_unused]] std::unique_lock<mutex_type> const &lock) const {
    EINSUMS_ASSERT(lock.owns_lock());

    return queue_.size();
}

// Return false if no more threads are waiting (returns true if queue
// is non-empty).
bool condition_variable::notify_one([[maybe_unused]] std::unique_lock<mutex_type> lock, execution::thread_priority /* priority */,
                                    error_code                                   &ec) {
    EINSUMS_ASSERT(lock.owns_lock());

    if (!queue_.empty()) {
        auto ctx = queue_.front().ctx_;

        // remove item from queue before error handling
        queue_.front().ctx_.reset();
        queue_.pop_front();

        if (EINSUMS_UNLIKELY(!ctx)) {
            lock.unlock();

            EINSUMS_THROWS_IF(ec, einsums::error::null_thread_id, "condition_variable::notify_one", "null thread id encountered");
            return false;
        }

        bool not_empty = !queue_.empty();
        ctx.resume();
        return not_empty;
    }

    if (&ec != &throws)
        ec = make_success_code();

    return false;
}

void condition_variable::notify_all([[maybe_unused]] std::unique_lock<mutex_type> lock, execution::thread_priority /* priority */,
                                    error_code                                   &ec) {
    EINSUMS_ASSERT(lock.owns_lock());

    // swap the list
    queue_type queue;
    queue.swap(queue_);

    // update reference to queue for all queue entries
    for (queue_entry &qe : queue)
        qe.q_ = &queue;

    while (!queue.empty()) {
        EINSUMS_ASSERT(queue.front().ctx_);
        queue_entry &qe  = queue.front();
        auto         ctx = qe.ctx_;
        qe.ctx_.reset();
        queue.pop_front();
        ctx.resume();
    }

    if (&ec != &throws)
        ec = make_success_code();
}

void condition_variable::abort_all(std::unique_lock<mutex_type> lock) {
    EINSUMS_ASSERT(lock.owns_lock());

    abort_all<mutex_type>(std::move(lock));
}

einsums::threads::detail::thread_restart_state condition_variable::wait(std::unique_lock<mutex_type> &lock, char const * /* description */,
                                                                        error_code & /* ec */) {
    EINSUMS_ASSERT(lock.owns_lock());

    // enqueue the request and block this thread
    auto        this_ctx = einsums::execution::this_thread::detail::agent();
    queue_entry f(this_ctx, &queue_);
    queue_.push_back(f);

    reset_queue_entry r(f, queue_);
    {
        // suspend this thread
        ::einsums::detail::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
        this_ctx.suspend();
    }

    return f.ctx_ ? einsums::threads::detail::thread_restart_state::timeout : einsums::threads::detail::thread_restart_state::signaled;
}

einsums::threads::detail::thread_restart_state condition_variable::wait_until(std::unique_lock<mutex_type>             &lock,
                                                                              einsums::chrono::steady_time_point const &abs_time,
                                                                              char const * /* description */, error_code & /* ec */) {
    EINSUMS_ASSERT(lock.owns_lock());

    // enqueue the request and block this thread
    auto        this_ctx = einsums::execution::this_thread::detail::agent();
    queue_entry f(this_ctx, &queue_);
    queue_.push_back(f);

    reset_queue_entry r(f, queue_);
    {
        // suspend this thread
        ::einsums::detail::unlock_guard<std::unique_lock<mutex_type>> ul(lock);
        this_ctx.sleep_until(abs_time.value());
    }

    return f.ctx_ ? einsums::threads::detail::thread_restart_state::timeout : einsums::threads::detail::thread_restart_state::signaled;
}

template <typename Mutex>
void condition_variable::abort_all(std::unique_lock<Mutex> lock) {
    // new threads might have been added while we were notifying
    while (!queue_.empty()) {
        // swap the list
        queue_type queue;
        queue.swap(queue_);

        // update reference to queue for all queue entries
        for (queue_entry &qe : queue)
            qe.q_ = &queue;

        while (!queue.empty()) {
            auto ctx = queue.front().ctx_;

            // remove item from queue before error handling
            queue.front().ctx_.reset();
            queue.pop_front();

            if (EINSUMS_UNLIKELY(!ctx)) {
                EINSUMS_LOG(err, "condition_variable::abort_all: null thread id encountered");
                continue;
            }

            EINSUMS_LOG(err, "condition_variable::abort_all: pending thread: {}", ctx);

            // unlock while notifying thread as this can suspend
            ::einsums::detail::unlock_guard<std::unique_lock<Mutex>> unlock(lock);

            // forcefully abort thread, do not throw
            ctx.abort();
        }
    }
}

// re-add the remaining items to the original queue
void condition_variable::prepend_entries([[maybe_unused]] std::unique_lock<mutex_type> &lock, queue_type &queue) {
    EINSUMS_ASSERT(lock.owns_lock());

    // splice is constant time only if it == end
    queue.splice(queue.end(), queue_);
    queue_.swap(queue);
}

///////////////////////////////////////////////////////////////////////////
void intrusive_ptr_add_ref(condition_variable_data *p) {
    ++p->count_;
}

void intrusive_ptr_release(condition_variable_data *p) {
    if (0 == --p->count_) {
        delete p;
    }
}

} // namespace einsums::detail
