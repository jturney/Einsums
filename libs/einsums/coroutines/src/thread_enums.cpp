//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/coroutines/thread_enums.hpp>

#include <cstddef>
#include <ostream>

namespace einsums::threads::detail {
namespace strings {
// clang-format off
        char const* const thread_state_names[] = {
            "unknown",
            "active",
            "pending",
            "suspended",
            "depleted",
            "terminated",
            "staged",
            "pending_do_not_schedule",
            "pending_boost"
        };
// clang-format on

} // namespace strings

auto get_thread_state_name(thread_schedule_state state) -> char const * {
    if (state > thread_schedule_state::pending_boost)
        return "unknown";
    return strings::thread_state_names[static_cast<std::size_t>(state)];
}

auto get_thread_state_name(thread_state state) -> char const * {
    return get_thread_state_name(state.state());
}

auto operator<<(std::ostream &os, thread_schedule_state const t) -> std::ostream & {
    os << get_thread_state_name(t) << " (" << static_cast<std::size_t>(t) << ")";
    return os;
}

///////////////////////////////////////////////////////////////////////
namespace strings {
// clang-format off
        char const* const thread_state_ex_names[] = {
            "wait_unknown",
            "wait_signaled",
            "wait_timeout",
            "wait_terminate",
            "wait_abort"
        };
// clang-format on
} // namespace strings

auto get_thread_state_ex_name(thread_restart_state state_ex) -> char const * {
    if (state_ex > thread_restart_state::abort)
        return "wait_unknown";
    return strings::thread_state_ex_names[static_cast<std::size_t>(state_ex)];
}

auto operator<<(std::ostream &os, thread_restart_state const t) -> std::ostream & {
    os << get_thread_state_ex_name(t) << " (" << static_cast<std::size_t>(t) << ")";
    return os;
}
} // namespace einsums::threads::detail

namespace einsums::execution {
namespace detail {
namespace strings {
// clang-format off
        char const* const thread_priority_names[] = {
            "default",
            "low",
            "normal",
            "high (recursive)",
            "boost",
            "high (non-recursive)",
        };
// clang-format on
} // namespace strings

auto get_thread_priority_name(thread_priority priority) -> char const * {
    if (priority < thread_priority::default_ || priority > thread_priority::high) {
        return "unknown";
    }
    return strings::thread_priority_names[static_cast<std::size_t>(priority)];
}

namespace strings {
// clang-format off
        char const* const stack_size_names[] = {
            "small",
            "medium",
            "large",
            "huge",
            "nostack",
        };
// clang-format on
} // namespace strings

auto get_stack_size_enum_name(thread_stacksize size) -> char const * {
    if (size == thread_stacksize::unknown)
        return "unknown";

    if (size < thread_stacksize::small_ || size > thread_stacksize::nostack)
        return "custom";

    return strings::stack_size_names[static_cast<std::size_t>(size) - 1];
}
} // namespace detail

auto operator<<(std::ostream &os, thread_stacksize const t) -> std::ostream & {
    os << detail::get_stack_size_enum_name(t) << " (" << static_cast<std::size_t>(t) << ")";
    return os;
}

auto operator<<(std::ostream &os, thread_priority const t) -> std::ostream & {
    os << detail::get_thread_priority_name(t) << " (" << static_cast<std::size_t>(t) << ")";
    return os;
}
} // namespace einsums::execution
