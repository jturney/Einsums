//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace einsums::threads::detail {

struct thread_init_data {

    thread_init_data()
        : func()
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
          ,
          description()
#endif
#if defined(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
          ,
          parent_id(nullptr), parent_phase(0)
#endif
          ,
          priority(execution::thread_priority::normal), schedule_hint(), stacksize(execution::thread_stacksize::default_),
          initial_state(thread_schedule_state::pending), run_now(false), scheduler_base(nullptr) {
    }

    thread_init_data &operator=(thread_init_data &&rhs) noexcept {
        func           = std::move(rhs.func);
        priority       = rhs.priority;
        schedule_hint  = rhs.schedule_hint;
        stacksize      = rhs.stacksize;
        initial_state  = rhs.initial_state;
        run_now        = rhs.run_now;
        scheduler_base = rhs.scheduler_base;
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
        description = std::move(rhs.description);
#endif
#if defined(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
        parent_id    = rhs.parent_id;
        parent_phase = rhs.parent_phase;
#endif
        return *this;
    }

    thread_init_data(thread_init_data &&rhs) noexcept
        : func(std::move(rhs.func))
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
          ,
          description(std::move(rhs.description))
#endif
#if defined(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
          ,
          parent_id(rhs.parent_id), parent_phase(rhs.parent_phase)
#endif
          ,
          priority(rhs.priority), schedule_hint(rhs.schedule_hint), stacksize(rhs.stacksize), initial_state(rhs.initial_state),
          run_now(rhs.run_now), scheduler_base(rhs.scheduler_base) {
    }

    template <typename F>
    thread_init_data(F &&f, ::einsums::detail::thread_description const &desc,
                     execution::thread_priority      priority_  = execution::thread_priority::normal,
                     execution::thread_schedule_hint os_thread  = execution::thread_schedule_hint(),
                     execution::thread_stacksize     stacksize_ = execution::thread_stacksize::default_,
                     thread_schedule_state initial_state_ = thread_schedule_state::pending, bool run_now_ = false,
                     ::einsums::threads::detail::scheduler_base *scheduler_base_ = nullptr)
        : func(std::forward<F>(f))
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
          ,
          description(desc)
#endif
#if defined(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
          ,
          parent_id(nullptr), parent_phase(0)
#endif
          ,
          priority(priority_), schedule_hint(os_thread), stacksize(stacksize_), initial_state(initial_state_), run_now(run_now_),
          scheduler_base(scheduler_base_) {
#ifndef EINSUMS_HAVE_THREAD_DESCRIPTION
        EINSUMS_UNUSED(desc);
#endif
        if (initial_state == thread_schedule_state::staged) {
            EINSUMS_THROW_EXCEPTION(einsums::error::bad_parameter, "threads shouldn't have 'staged' as their initial state");
        }
    }

#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    ::einsums::detail::thread_description get_description() const { return description; }
#else
    ::einsums::detail::thread_description get_description() const { return ::einsums::detail::thread_description("<unknown>"); }
#endif

    thread_function_type func;
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
    ::einsums::detail::thread_description description;
#endif
#if defined(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
    thread_id_type parent_id;
    std::size_t    parent_phase;
#endif
    execution::thread_priority      priority;
    execution::thread_schedule_hint schedule_hint;
    execution::thread_stacksize     stacksize;
    thread_schedule_state           initial_state;
    bool                            run_now;

    ::einsums::threads::detail::scheduler_base *scheduler_base;
};

} // namespace einsums::threads::detail