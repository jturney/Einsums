//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/runtime/Runtime.hpp>
#include <einsums/runtime/StartupFunction.hpp>
#include <einsums/runtime/State.hpp>

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
    "RuntimeState::Invalid",     // -1
    "RuntimeState::Initialized", // 0
    "RuntimeState::PreStartup",  // 1
    "RuntimeState::Startup",     // 2
    "RuntimeState::PreMain",     // 3
    "RuntimeState::tarting",     // 4
    "RuntimeState::Running",     // 5
    "RuntimeState::Suspended",   // 6
    "RuntimeState::PreSleep",    // 7
    "RuntimeState::Sleeping",    // 8
    "RuntimeState::PreShutdown", // 9
    "RuntimeState::Shutdown",    // 10
    "RuntimeState::Stopping",    // 11
    "RuntimeState::Terminating", // 12
    "RuntimeState::Stopped"      // 13
};

} // namespace strings

namespace detail {
auto get_runtime_state_name(RuntimeState st) -> char const * {
    if (st < RuntimeState::Invalid || st >= RuntimeState::LastValidRuntime)
        return "invalid (value out of bounds)";
    return strings::runtime_state_names[static_cast<std::int8_t>(st) + 1];
}
} // namespace detail

runtime::runtime() {
    init();
}

void runtime::init() {
    set_state(RuntimeState::Initialized);
}

auto runtime::get_state() const -> RuntimeState {
    return _state.load();
}

void runtime::set_state(RuntimeState s) {
    _state.store(s);
}

} // namespace einsums