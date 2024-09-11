//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/debugging/attach_debugger.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/runtime.hpp>
#include <einsums/runtime/runtime_fwd.hpp>
#include <einsums/string_util/from_string.hpp>
#include <einsums/version.hpp>

#include <iostream>

#if defined(EINSUMS_WINDOWS)

namespace einsums::detail {
///////////////////////////////////////////////////////////////////////////
void handle_termination(char const *reason) {
    if (get_config_entry("einsums.attach_debugger", "") == "exception") {
        debug::detail::attach_debugger();
    }

    if (get_config_entry("einsums.diagnostics_on_terminate", "1") == "1") {
        int const verbosity = detail::from_string<int>(get_config_entry("einsums.exception_verbosity", "1"));

        if (verbosity >= 2) {
            std::cerr << einsums::full_build_string() << "\n";
        }

#    if defined(EINSUMS_HAVE_STACKTRACES)
        if (verbosity >= 1) {
            std::size_t const trace_depth =
                detail::from_string<std::size_t>(get_config_entry("einsums.trace_depth", EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH));
            std::cerr << "{stack-trace}: " << einsums::debug::detail::trace(trace_depth) << "\n";
        }
#    endif

        std::cerr << "{what}: " << (reason ? reason : "Unknown reason") << "\n";
    }
}

EINSUMS_EXPORT BOOL WINAPI termination_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
    case CTRL_C_EVENT:
        handle_termination("Ctrl-C");
        return TRUE;

    case CTRL_BREAK_EVENT:
        handle_termination("Ctrl-Break");
        return TRUE;

    case CTRL_CLOSE_EVENT:
        handle_termination("Ctrl-Close");
        return TRUE;

    case CTRL_LOGOFF_EVENT:
        handle_termination("Logoff");
        return TRUE;

    case CTRL_SHUTDOWN_EVENT:
        handle_termination("Shutdown");
        return TRUE;

    default:
        break;
    }
    return FALSE;
}
} // namespace einsums::detail

#else

#    include <signal.h>
#    include <stdlib.h>
#    include <string.h>

namespace einsums::detail {

[[noreturn]] EINSUMS_EXPORT void termination_handler(int signum) {
    if (signum != SIGINT && get_config_entry("einsums.attach_debugger", "") == "exception") {
        debug::detail::attach_debugger();
    }

    if (get_config_entry("einsums.diagnostics_on_terminate", "1") == "1") {
        int const verbosity = string_util::from_string<int>(get_config_entry("einsums.exception_verbosity", "1"));
        char     *reason    = strsignal(signum);

        if (verbosity >= 2) {
            std::cerr << einsums::full_build_string() << "\n";
        }

#    if defined(EINSUMS_HAVE_STACKTRACES)

#    endif

        std::cerr << "{what}: " << (reason ? reason : "Unknown reason") << "\n";
    }
    std::abort();
}

} // namespace einsums::detail

#endif

namespace einsums::detail {

EINSUMS_EXPORT void EINSUMS_CDECL new_handler() {
    EINSUMS_THROW_EXCEPTION(error::out_of_memory, "new allocator failed to allocate memory");
}

static bool exit_called = false;

void on_exit() noexcept {
    exit_called = true;
}

void on_abort(int) noexcept {
    exit_called = true;
    std::exit(-1);
}

void set_signal_handlers() {
#if defined(EINSUMS_WINDOWS)
    SetControlCtrlHandler(einsums::detail::termination_handler, TRUE);
#else
    struct sigaction new_action;
    new_action.sa_handler = einsums::detail::termination_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;

    sigaction(SIGINT, &new_action, nullptr);  // Interrupted
    sigaction(SIGBUS, &new_action, nullptr);  // Bus error
    sigaction(SIGFPE, &new_action, nullptr);  // Floating point exception
    sigaction(SIGILL, &new_action, nullptr);  // Illegal instruction
    sigaction(SIGPIPE, &new_action, nullptr); // Bad pipe
    sigaction(SIGSEGV, &new_action, nullptr); // Segmentation fault
    sigaction(SIGSYS, &new_action, nullptr);  // Bad syscall
#endif

    std::set_new_handler(einsums::detail::new_handler);
}

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

}

char const *get_runtime_state_name(runtime_state st) {
    if (st < runtime_state::invalid || st >= runtime_state::last_valid_runtime)
        return "invalid (value out of bounds)";
    return strings::runtime_state_names[static_cast<std::int8_t>(st) + 1];
}

einsums::threads::callback_notifier::on_startstop_type global_on_start_func;
einsums::threads::callback_notifier::on_startstop_type global_on_stop_func;
einsums::threads::callback_notifier::on_error_type     global_on_error_func;

runtime::runtime(einsums::util::runtime_configuration &rtcfg, bool initialize) : _rtcfg(rtcfg) {
}

#if 0

runtime &get_runtime() {
    EINSUMS_ASSERT(get_runtime_ptr() != nullptr);
    return *get_runtime_ptr();
}

runtime *&get_runtime_ptr() {
    static runtime *_runtime = nullptr;
    return _runtime;
}

std::string get_config_entry(std::string const &key, std::string const &dflt) {
    if (get_runtime_ptr() != nullptr) {
        return get_runtime().get_config().get_entry(key, dflt);
    }
    return dflt;
}

std::string get_config_entry(std::string const &key, std::size_t dflt) {
    if (get_runtime_ptr() != nullptr) {
        return get_runtime().get_config().get_entry(key, dflt);
    }

    return std::to_string(dflt);
}

void set_config_entry(std::string const &key, std::string const &value) {
    if (get_runtime_ptr() != nullptr) {
        get_runtime_ptr()->get_config().add_entry(key, value);
        return;
    }
}

void set_config_entry(std::string const &key, std::size_t value) {
    set_config_entry(key, std::to_string(value));
}

void set_config_entry_callback(std::string const &key, std::function<void(std::string const &, std::string const &)> const &callback) {
    if (get_runtime_ptr() != nullptr) {
        get_runtime_ptr()->get_config().add_notification_callback(key, callback);
        return;
    }
}

runtime::runtime(util::runtime_configuration &rtcfg, bool initialize) : _rtcfg(rtcfg) {
    init_global_data();

    if (initialize) {
        init();
    }
}

// this constructor is called by the distributed runtime only
runtime::runtime(util::runtime_configuration &rtcfg) : _rtcfg(rtcfg) {
    init_global_data();
}

void runtime::init() {
}

runtime::~runtime() {
}

void runtime::init_global_data() {
    runtime *&_runtime = get_runtime_ptr();
    EINSUMS_ASSERT(!_runtime);

    _runtime = this;
}

util::runtime_configuration &runtime::get_config() {
    return _rtcfg;
}
util::runtime_configuration const &runtime::get_config() const {
    return _rtcfg;
}

#endif

} // namespace einsums::detail
