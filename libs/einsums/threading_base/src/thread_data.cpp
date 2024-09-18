//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/coroutines/detail/coroutine_accessor.hpp>
#include <einsums/functional/function.hpp>
#include <einsums/lock_registration/detail/register_locks.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/thread_support/unlock_guard.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_data.hpp>
#if defined(EINSUMS_HAVE_APEX)
#    include <einsums/threading_base/external_timer.hpp>
#endif

#include <fmt/format.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
namespace einsums::threads::detail {
thread_data::thread_data(thread_init_data &init_data, void *queue, std::ptrdiff_t stacksize, bool is_stackless, thread_id_addref addref)
    : thread_data_reference_counting(addref), current_state_(thread_state(init_data.initial_state, thread_restart_state::signaled))
#ifdef EINSUMS_HAVE_THREAD_DESCRIPTION
      ,
      description_(init_data.description), lco_description_()
#endif
#ifdef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
      ,
      parent_thread_id_(init_data.parent_id), parent_thread_phase_(init_data.parent_phase)
#endif
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
      ,
      marked_state_(thread_schedule_state::unknown)
#endif
#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
      ,
      backtrace_(nullptr)
#endif
      ,
      priority_(init_data.priority), requested_interrupt_(false), enabled_interrupt_(true), ran_exit_funcs_(false),
      is_stackless_(is_stackless), scheduler_base_(init_data.scheduler_base), last_worker_thread_num_(std::size_t(-1)),
      stacksize_(stacksize), stacksize_enum_(init_data.stacksize), queue_(queue) {
    EINSUMS_LOG(debug, "thread::thread({}), description({})", fmt::ptr(this), get_description());

    EINSUMS_ASSERT(stacksize_enum_ != execution::thread_stacksize::current);

#ifdef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
    // store the thread id of the parent thread, mainly for debugging
    // purposes
    if (parent_thread_id_ == nullptr) {
        thread_self *self = get_self_ptr();
        if (self) {
            parent_thread_id_    = get_self_id();
            parent_thread_phase_ = self->get_thread_phase();
        }
    }
#endif
#if defined(EINSUMS_HAVE_APEX)
    set_timer_data(init_data.timer_data);
#endif
}

thread_data::~thread_data() {
    EINSUMS_LOG(debug, "thread_data::~thread_data({})", fmt::ptr(this));
    free_thread_exit_callbacks();
}

void thread_data::destroy_thread() {
    EINSUMS_LOG(debug, "thread_data::destroy_thread({}), description({}), phase({})", fmt::ptr(this), this->get_description(),
                this->get_thread_phase());

    get_scheduler_base()->destroy_thread(this);
}

void thread_data::run_thread_exit_callbacks() {
    std::unique_lock<einsums::detail::spinlock> l(spinlock_pool::spinlock_for(this));

    while (!exit_funcs_.empty()) {
        {
            einsums::detail::unlock_guard<std::unique_lock<einsums::detail::spinlock>> ul(l);
            if (!exit_funcs_.front().empty())
                exit_funcs_.front()();
        }
        exit_funcs_.pop_front();
    }
    ran_exit_funcs_ = true;
}

bool thread_data::add_thread_exit_callback(util::detail::function<void()> const &f) {
    std::lock_guard<einsums::detail::spinlock> l(spinlock_pool::spinlock_for(this));

    if (ran_exit_funcs_ || get_state().state() == thread_schedule_state::terminated) {
        return false;
    }

    exit_funcs_.push_front(f);

    return true;
}

void thread_data::free_thread_exit_callbacks() {
    std::lock_guard<einsums::detail::spinlock> l(spinlock_pool::spinlock_for(this));

    // Exit functions should have been executed.
    EINSUMS_ASSERT(exit_funcs_.empty() || ran_exit_funcs_);

    exit_funcs_.clear();
}

bool thread_data::interruption_point(bool throw_on_interrupt) {
    // We do not protect enabled_interrupt_ and requested_interrupt_
    // from concurrent access here (which creates a benign data race) in
    // order to avoid infinite recursion. This function is called by
    // this_thread::suspend which causes problems if the lock would call
    // suspend itself.
    if (enabled_interrupt_ && requested_interrupt_) {
        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        // now interrupt this thread
        if (throw_on_interrupt) {
            requested_interrupt_ = false; // avoid recursive exceptions
            throw einsums::thread_interrupted();
        }

        return true;
    }
    return false;
}

void thread_data::rebind_base(thread_init_data &init_data) {
    EINSUMS_LOG(debug, "thread_data::rebind_base({}), description({}), phase({}), rebind", fmt::ptr(this), get_description(),
                get_thread_phase());

    free_thread_exit_callbacks();

    current_state_.store(thread_state(init_data.initial_state, thread_restart_state::signaled));

#ifdef EINSUMS_HAVE_THREAD_DESCRIPTION
    description_     = init_data.description;
    lco_description_ = ::einsums::detail::thread_description();
#endif
#ifdef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
    parent_thread_id_    = init_data.parent_id;
    parent_thread_phase_ = init_data.parent_phase;
#endif
#ifdef EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION
    set_marked_state(thread_schedule_state::unknown);
#endif
#ifdef EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
    backtrace_ = nullptr;
#endif
    priority_            = init_data.priority;
    requested_interrupt_ = false;
    enabled_interrupt_   = true;
    ran_exit_funcs_      = false;
    exit_funcs_.clear();
    scheduler_base_ = init_data.scheduler_base;
    last_worker_thread_num_.store(std::size_t(-1), std::memory_order_relaxed);

    // We explicitly set the logical stack size again as it can be different
    // from what the previous use required. However, the physical stack size
    // must be the same as before.
    stacksize_enum_ = init_data.stacksize;
    EINSUMS_ASSERT(stacksize_ == get_stack_size());
    EINSUMS_ASSERT(stacksize_ != 0);

    EINSUMS_LOG(debug, "thread::thread({}), description({}), rebind", fmt::ptr(this), get_description());

#ifdef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
    // store the thread id of the parent thread, mainly for debugging
    // purposes
    if (parent_thread_id_ == nullptr) {
        thread_self *self = get_self_ptr();
        if (self) {
            parent_thread_id_    = get_self_id();
            parent_thread_phase_ = self->get_thread_phase();
        }
    }
#endif
#if defined(EINSUMS_HAVE_APEX)
    set_timer_data(init_data.timer_data);
#endif
}

///////////////////////////////////////////////////////////////////////////
thread_self &get_self() {
    thread_self *p = get_self_ptr();
    if (EINSUMS_UNLIKELY(p == nullptr)) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id, "get_self",
                                "null thread id encountered (is this executed on a einsums-thread?)");
    }
    return *p;
}

thread_self *get_self_ptr() {
    return thread_self::get_self();
}

namespace detail {
void set_self_ptr(thread_self *self) {
    thread_self::set_self(self);
}
} // namespace detail

thread_self::impl_type *get_ctx_ptr() {
    using einsums::threads::coroutines::detail::coroutine_accessor;
    return coroutine_accessor::get_impl(get_self());
}

thread_self *get_self_ptr_checked(error_code &ec) {
    thread_self *p = thread_self::get_self();

    if (EINSUMS_UNLIKELY(p == nullptr)) {
        EINSUMS_THROWS_IF(ec, einsums::error::null_thread_id, "get_self_ptr_checked",
                          "null thread id encountered (is this executed on a einsums-thread?)");
        return nullptr;
    }

    if (&ec != &throws)
        ec = make_success_code();

    return p;
}

thread_id_type get_self_id() {
    thread_self *self = get_self_ptr();
    if (EINSUMS_LIKELY(nullptr != self))
        return self->get_thread_id();

    return invalid_thread_id;
}

thread_data *get_self_id_data() {
    thread_self *self = get_self_ptr();
    if (EINSUMS_LIKELY(nullptr != self))
        return get_thread_id_data(self->get_thread_id());

    return nullptr;
}

std::ptrdiff_t get_self_stacksize() {
    thread_data *thrd_data = get_self_id_data();
    return thrd_data ? thrd_data->get_stack_size() : 0;
}

execution::thread_stacksize get_self_stacksize_enum() {
    thread_data                *thrd_data = get_self_id_data();
    execution::thread_stacksize stacksize = thrd_data ? thrd_data->get_stack_size_enum() : execution::thread_stacksize::default_;
    EINSUMS_ASSERT(stacksize != execution::thread_stacksize::current);
    return stacksize;
}

#ifndef EINSUMS_HAVE_THREAD_PARENT_REFERENCE
thread_id_type get_parent_id() {
    return invalid_thread_id;
}

std::size_t get_parent_phase() {
    return 0;
}
#else
thread_id_type get_parent_id() {
    thread_data *thrd_data = get_self_id_data();
    if (EINSUMS_LIKELY(nullptr != thrd_data)) {
        return thrd_data->get_parent_thread_id();
    }
    return invalid_thread_id;
}

std::size_t get_parent_phase() {
    thread_data *thrd_data = get_self_id_data();
    if (EINSUMS_LIKELY(nullptr != thrd_data)) {
        return thrd_data->get_parent_thread_phase();
    }
    return 0;
}
#endif

#if defined(EINSUMS_HAVE_APEX)
std::shared_ptr<einsums::detail::external_timer::task_wrapper> get_self_timer_data() {
    thread_data *thrd_data = get_self_id_data();
    if (EINSUMS_LIKELY(nullptr != thrd_data)) {
        return thrd_data->get_timer_data();
    }
    return nullptr;
}
void set_self_timer_data(std::shared_ptr<einsums::detail::external_timer::task_wrapper> data) {
    thread_data *thrd_data = get_self_id_data();
    if (EINSUMS_LIKELY(nullptr != thrd_data)) {
        thrd_data->set_timer_data(data);
    }
    return;
}
#endif
} // namespace einsums::threads::detail
