//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/execution_base/context_base.hpp>
#include <einsums/timing/steady_clock.hpp>

#include <cstddef>
#include <string>

namespace einsums::execution::detail {

struct agent_base {
    virtual ~agent_base() = default;

    virtual std::string description() const = 0;

    virtual context_base const &context() const = 0;

    virtual void yield(char const *desc)                                                             = 0;
    virtual void yield_k(std::size_t k, char const *desc)                                            = 0;
    virtual void spin_k(std::size_t k, char const *desc)                                             = 0;
    virtual void suspend(char const *desc)                                                           = 0;
    virtual void resume(char const *desc)                                                            = 0;
    virtual void abort(char const *desc)                                                             = 0;
    virtual void sleep_for(einsums::chrono::steady_duration const &sleep_duration, char const *desc) = 0;
    virtual void sleep_until(einsums::chrono::steady_time_point const &sleep_time, char const *desc) = 0;
};

} // namespace einsums::execution::detail