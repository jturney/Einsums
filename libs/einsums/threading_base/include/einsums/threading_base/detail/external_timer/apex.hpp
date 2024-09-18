//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/threading_base/thread_description.hpp>
#include <einsums/threading_base/threading_base_fwd.hpp>

#include <apex_api.hpp>
#include <cstdint>
#include <memory>

namespace einsums::detail::external_timer {
using apex::finalize;
using apex::init;
using apex::new_task;
using apex::recv;
using apex::register_thread;
using apex::send;
using apex::start;
using apex::stop;
using apex::update_task;
using apex::yield;

EINSUMS_EXPORT std::shared_ptr<task_wrapper> new_task(einsums::detail::thread_description const &description,
                                                      threads::detail::thread_id_type            parent_task);

EINSUMS_EXPORT std::shared_ptr<task_wrapper> update_task(std::shared_ptr<task_wrapper>              wrapper,
                                                         einsums::detail::thread_description const &description);

// This is a scoped object around task scheduling to measure the time spent
// executing einsums threads
struct [[nodiscard]] scoped_timer {
    EINSUMS_EXPORT explicit scoped_timer(std::shared_ptr<task_wrapper> data_ptr);
    scoped_timer(scoped_timer &&)                 = delete;
    scoped_timer(scoped_timer const &)            = delete;
    scoped_timer &operator=(scoped_timer &&)      = delete;
    scoped_timer &operator=(scoped_timer const &) = delete;
    EINSUMS_EXPORT ~scoped_timer();

    EINSUMS_EXPORT void stop();
    EINSUMS_EXPORT void yield();

  private:
    bool                          stopped;
    std::shared_ptr<task_wrapper> _data;
};
} // namespace einsums::detail::external_timer
