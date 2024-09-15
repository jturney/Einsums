//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/timing/steady_clock.hpp>

#include <fmt/format.h>

#include <chrono>
#include <cstddef>
#include <string>

namespace einsums::execution::detail {
struct agent_base;

class EINSUMS_EXPORT agent_ref {
  public:
    constexpr agent_ref() noexcept : impl_(nullptr) {}
    constexpr agent_ref(agent_base *impl) noexcept : impl_(impl) {}

    constexpr agent_ref(agent_ref const &) noexcept            = default;
    constexpr agent_ref &operator=(agent_ref const &) noexcept = default;

    constexpr agent_ref(agent_ref &&) noexcept            = default;
    constexpr agent_ref &operator=(agent_ref &&) noexcept = default;

    constexpr explicit operator bool() const noexcept { return impl_ != nullptr; }

    void reset(agent_base *impl = nullptr) { impl_ = impl; }

    void yield(char const *desc = "einsums::execution::detail::agent_ref::yield");
    void yield_k(std::size_t k, char const *desc = "einsums::execution::detail::agent_ref::yield_k");
    void spin_k(std::size_t k, char const *desc = "einsums::execution::detail::agent_ref::spin_k");
    void suspend(char const *desc = "einsums::execution::detail::agent_ref::suspend");
    void resume(char const *desc = "einsums::execution::detail::agent_ref::resume");
    void abort(char const *desc = "einsums::execution::detail::agent_ref::abort");

    template <typename Rep, typename Period>
    void sleep_for(std::chrono::duration<Rep, Period> const &sleep_duration,
                   char const                               *desc = "einsums::execution::detail::agent_ref::sleep_for") {
        sleep_for(einsums::chrono::steady_duration{sleep_duration}, desc);
    }

    template <typename Clock, typename Duration>
    void sleep_until(std::chrono::time_point<Clock, Duration> const &sleep_time,
                     char const                                     *desc = "einsums::execution::detail::agent_ref::sleep_until") {
        sleep_until(einsums::chrono::steady_time_point{sleep_time}, desc);
    }

    agent_base &ref() { return *impl_; }

    // TODO:
    // affinity
    // thread_num
    // executor

  private:
    agent_base *impl_;

    void sleep_for(einsums::chrono::steady_duration const &sleep_duration, char const *desc);
    void sleep_until(einsums::chrono::steady_time_point const &sleep_time, char const *desc);

    friend constexpr bool operator==(agent_ref const &lhs, agent_ref const &rhs) { return lhs.impl_ == rhs.impl_; }

    friend constexpr bool operator!=(agent_ref const &lhs, agent_ref const &rhs) { return lhs.impl_ != rhs.impl_; }

    friend std::string format(agent_ref const &);
};

EINSUMS_EXPORT std::string format(agent_ref const &);
} // namespace einsums::execution::detail

template <>
struct fmt::formatter<einsums::execution::detail::agent_ref> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::execution::detail::agent_ref const &a, FormatContext &ctx) const {
        return fmt::formatter<std::string>::format(einsums::execution::detail::format(a), ctx);
    }
};
