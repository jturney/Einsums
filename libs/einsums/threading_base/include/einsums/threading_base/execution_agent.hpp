//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/coroutines/detail/coroutine_impl.hpp>
#include <einsums/coroutines/detail/coroutine_stackful_self.hpp>
#include <einsums/coroutines/thread_enums.hpp>
#include <einsums/coroutines/thread_id_type.hpp>
#include <einsums/execution_base/agent_base.hpp>
#include <einsums/execution_base/context_base.hpp>
#include <einsums/execution_base/resource_base.hpp>
#include <einsums/timing/steady_clock.hpp>

#include <cstddef>
#include <string>

namespace einsums::threads::detail {
struct EINSUMS_EXPORT execution_context : einsums::execution::detail::context_base {
    einsums::execution::detail::resource_base const &resource() const override { return resource_; }
    einsums::execution::detail::resource_base        resource_;
};

struct EINSUMS_EXPORT execution_agent : einsums::execution::detail::agent_base {
    explicit execution_agent(coroutines::detail::coroutine_impl *coroutine) noexcept;

    std::string description() const override;

    execution_context const &context() const override { return context_; }

    void yield(char const *desc) override;
    void yield_k(std::size_t k, char const *desc) override;
    void spin_k(std::size_t k, char const *desc) override;
    void suspend(char const *desc) override;
    void resume(char const *desc) override;
    void abort(char const *desc) override;
    void sleep_for(einsums::chrono::steady_duration const &sleep_duration, char const *desc) override;
    void sleep_until(einsums::chrono::steady_time_point const &sleep_time, char const *desc) override;

  private:
    coroutines::detail::coroutine_stackful_self self_;

    thread_restart_state do_yield(char const *desc, thread_schedule_state state);

    void do_resume(char const *desc, thread_restart_state statex);

    execution_context context_;
};
} // namespace einsums::threads::detail
