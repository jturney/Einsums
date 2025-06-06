//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Debugging/AttachDebugger.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <csignal>

#if defined(EINSUMS_WINDOWS)
#    include <Windows.h>
#endif

namespace einsums {
namespace detail {

EINSUMS_SINGLETON_IMPL(RuntimeVars)

#if defined(EINSUMS_WINDOWS)

void handle_termination(char const *reason) {
    if (runtime_config().einsums.attach_debugger) {
        util::attach_debugger();
    }

    if (runtime_config().einsums.diagnostics_on_terminate) {
        // Add more information here.
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

#else
[[noreturn]] EINSUMS_EXPORT void termination_handler(int signum) {
    bool attach = true;

    try {
        auto &global_config = GlobalConfigMap::get_singleton();
        attach              = global_config.get_bool("attach-debugger", true);
    } catch (...) {
        attach = true;
    }

    if (signum != SIGINT && attach) {
        util::attach_debugger();
    }

    /// @todo If einsums.diagnostics_on_terminate is true then print out a lot of information.

    std::abort();
}
#endif

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
    SetConsoleCtrlHandler(termination_handler, TRUE);
#else
    struct sigaction new_action {};
    new_action.sa_handler = termination_handler;
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
}

Runtime::Runtime(RuntimeConfiguration &&rtcfg, bool initialize) : _rtcfg(std::move(rtcfg)) {
    init_global_data();

    if (initialize) {
        init();
    }
}

RuntimeState Runtime::state() const {
    return _state;
}

void Runtime::state(RuntimeState state) {
    EINSUMS_LOG_INFO("Runtime state changed from {} to {}", _state, state);
    _state = state;
}

RuntimeConfiguration &Runtime::config() {
    return _rtcfg;
}

RuntimeConfiguration const &Runtime::config() const {
    return _rtcfg;
}

void Runtime::init() {
    EINSUMS_LOG_INFO("Runtime::init: initializing...");
    try {
        /// @todo This would be a good place to create and initialize a thread pool

        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> vars_guard(runtime_vars); // Lock the variables.

        // Copy over all startup functions registered so far.
        for (StartupFunctionType &f : runtime_vars.global_pre_startup_functions) {
            add_pre_startup_function(f);
        }

        for (StartupFunctionType &f : runtime_vars.global_startup_functions) {
            add_startup_function(f);
        }

        for (ShutdownFunctionType &f : runtime_vars.global_pre_shutdown_functions) {
            add_pre_shutdown_function(f);
        }

        for (ShutdownFunctionType &f : runtime_vars.global_shutdown_functions) {
            add_shutdown_function(f);
        }
    } catch (std::exception const &e) {
        /// @todo report_exception_and_terminate(e);
    } catch (...) {
        /// @todo report_exception_and_terminate(std::current_exception());
    }
}

void Runtime::init_global_data() {
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(!runtime_);

    runtime_ = this;
}

void Runtime::deinit_global_data() {
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(runtime_);
    runtime_ = nullptr;
}

void Runtime::add_pre_shutdown_function(ShutdownFunctionType f) {
    std::lock_guard l(this->lock_);
    _pre_shutdown_functions.push_back(f);
}

void Runtime::add_shutdown_function(ShutdownFunctionType f) {
    std::lock_guard l(this->lock_);
    _shutdown_functions.push_back(f);
}

void Runtime::add_pre_startup_function(StartupFunctionType f) {
    std::lock_guard l(this->lock_);
    _pre_startup_functions.push_back(f);
}

void Runtime::add_startup_function(StartupFunctionType f) {
    std::lock_guard l(this->lock_);
    _startup_functions.push_back(f);
}

void Runtime::call_startup_functions(bool pre_startup) {
    if (pre_startup) {
        EINSUMS_LOG_TRACE("Calling pre-startup routines");
        state(RuntimeState::PreStartup);
        for (StartupFunctionType &f : _pre_startup_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling startup routines");
        state(RuntimeState::Startup);
        for (StartupFunctionType &f : _startup_functions) {
            f();
        }
    }
}

void Runtime::call_shutdown_functions(bool pre_shutdown) {
    if (pre_shutdown) {
        EINSUMS_LOG_TRACE("Calling pre-shutdown routines");
        state(RuntimeState::PreShutdown);
        for (ShutdownFunctionType &f : _pre_shutdown_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling shutdown routines");
        state(RuntimeState::Shutdown);
        for (ShutdownFunctionType &f : _shutdown_functions) {
            f();
        }
    }
}

int Runtime::run(std::function<EinsumsMainFunctionType> const &func) {
    call_startup_functions(true);
    call_startup_functions(false);

    // Set the state to running.
    state(RuntimeState::Running);
    // Once we start using a thread pool / threading manager we can
    // pass the function to the pool and have the manager handle it.
    EINSUMS_LOG_INFO("running user provided function");
    int result = func();

    return result;
}

int Runtime::run() {
    call_startup_functions(true);
    call_startup_functions(false);

    // Set the state to running.
    state(RuntimeState::Running);

    return 0;
}

} // namespace detail

bool is_running() {
    if (detail::Runtime *rt = runtime_ptr(); nullptr != rt)
        return rt->state() == RuntimeState::Running;
    return false;
}

detail::Runtime &runtime() {
    EINSUMS_ASSERT(runtime_ptr() != nullptr);
    return *runtime_ptr();
}

detail::Runtime *&runtime_ptr() {
    static detail::Runtime *runtime_ = nullptr;
    return runtime_;
}

RuntimeConfiguration &runtime_config() {
    return runtime().config();
}

void register_pre_startup_function(StartupFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::PreStartup) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a pre-startup function");
        }
        runtime->add_pre_startup_function(std::move(f));
    } else {
        auto           &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard guard(runtime_vars);
        runtime_vars.global_pre_startup_functions.emplace_back(std::move(f));
    }
}

void register_startup_function(StartupFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Startup) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a startup function");
        }
        runtime->add_startup_function(std::move(f));
    } else {
        auto           &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard guard(runtime_vars);
        runtime_vars.global_startup_functions.emplace_back(std::move(f));
    }
}

void register_pre_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::PreShutdown) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a pre-shutdown function");
        }
        runtime->add_pre_shutdown_function(std::move(f));
    } else {
        auto           &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard guard(runtime_vars);
        runtime_vars.global_pre_shutdown_functions.emplace_back(std::move(f));
    }
}

void register_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Shutdown) {
            EINSUMS_THROW_EXCEPTION(invalid_runtime_state, "Too late to register a shutdown function");
        }
        runtime->add_pre_shutdown_function(std::move(f));
    } else {
        auto           &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard guard(runtime_vars);
        runtime_vars.global_shutdown_functions.emplace_back(std::move(f));
    }
}

} // namespace einsums