//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/debugging/attach_debugger.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/runtime/state.hpp>

#if !defined(EINSUMS_WINDOWS)
#    include <csignal>
#    include <cstdlib>
#    include <cstring>

namespace einsums {

[[noreturn]] EINSUMS_EXPORT void termination_handler(int signum) {
    if (signum != SIGINT /* && need to check runtime configuration */) {
        debug::detail::attach_debugger();
    }

    std::abort();
}

} // namespace einsums

#endif

namespace einsums {
namespace strings {

char const *const runtime_state_names[] = {
    "runtime_state::invalid",      // -1
    "runtime_state::initialized",  // 0
    "runtime_state::pre_startup",  // 1
    "runtime_state::startup",      // 2
    "runtime_state::pre_main",     // 3
    "runtime_state::starting",     // 4
    "runtime_state::running",      // 5
    "runtime_state::suspended",    // 6
    "runtime_state::pre_sleep",    // 7
    "runtime_state::sleeping",     // 8
    "runtime_state::pre_shutdown", // 9
    "runtime_state::shutdown",     // 10
    "runtime_state::stopping",     // 11
    "runtime_state::terminating",  // 12
    "runtime_state::stopped"       // 13
};

} // namespace strings

namespace detail {
auto get_runtime_state_name(runtime_state st) -> char const * {
    if (st < runtime_state::invalid || st >= runtime_state::last_valid_runtime)
        return "invalid (value out of bounds)";
    return strings::runtime_state_names[static_cast<std::int8_t>(st) + 1];
}
} // namespace detail

runtime::runtime() {
    init();
}

void runtime::init() {
    set_state(runtime_state::initialized);
}

auto runtime::get_state() const -> runtime_state {
    return _state.load();
}

void runtime::set_state(runtime_state s) {
    _state.store(s);
}

} // namespace einsums