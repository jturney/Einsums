//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Debugging/AttachDebugger.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Debugging/CrashHandler.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Options.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#if defined(EINSUMS_WINDOWS)
#    include <Windows.h>
#else
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN()
namespace detail {

EINSUMS_SINGLETON_IMPL(RuntimeVars)

#if defined(EINSUMS_WINDOWS)

void handle_termination(char const *reason) {
    // A descriptor read cannot throw a lookup error, but a handler running on
    // the way down should not be the thing that adds one either.
    bool attach      = true;
    bool diagnostics = true;
    try {
        attach      = config::get(option::AttachDebugger);
        diagnostics = config::get(option::DiagnosticsOnTerminate);
    } catch (...) {
    }

    // Diagnostics first, for the same reason as the POSIX handler below: whatever
    // attach_debugger() does, it is not guaranteed to come back.
    if (diagnostics) {
        std::cerr << "\n=== einsums: terminating ===\n" << (reason ? reason : "Unknown reason") << "\n";
        try {
            std::string const trace = util::backtrace();
            if (!trace.empty()) {
                std::cerr << "\nbacktrace:\n" << trace << "\n";
            }
        } catch (...) { // NOLINT
            std::cerr << "\nbacktrace: unavailable\n";
        }
        std::cerr << "=== end einsums termination ===\n";
        std::cerr.flush();
    }

    if (attach) {
        util::attach_debugger();
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
namespace {

/// Spelling for the signals @ref set_signal_handlers subscribes to.
char const *signal_name(int signum) {
    switch (signum) {
    case SIGINT:
        return "SIGINT (interrupted)";
    case SIGBUS:
        return "SIGBUS (bus error)";
    case SIGFPE:
        return "SIGFPE (floating point exception)";
    case SIGILL:
        return "SIGILL (illegal instruction)";
    case SIGSEGV:
        return "SIGSEGV (segmentation fault)";
    case SIGSYS:
        return "SIGSYS (bad syscall)";
    default:
        return "unrecognized signal";
    }
}

/// Leave, with a note, when symbolization does not come back.
///
/// @ref util::backtrace resolves through cpptrace, which allocates and takes
/// locks. Either can already be held by the frame this handler interrupted, and
/// cpptrace's symbol cache is shared between threads, so the walk can block
/// rather than fail. Waiting on a lock is not an exception, so the catch around
/// the call never fires: the process then hangs carrying no diagnostic at all,
/// which is worse than the crash it was trying to explain. An alarm is what
/// makes "allowed to fail" true.
void backtrace_timeout_handler(int) {
    static char const     message[] = "\nbacktrace: symbolizer blocked, leaving without one\n";
    [[maybe_unused]] auto ignored   = write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(EXIT_FAILURE);
}

/// How long a backtrace may take before the handler gives up on it. Generous,
/// because symbolizing a large binary cold is slow and the only cost of a high
/// bound is how long a genuinely wedged crash takes to fall over.
constexpr unsigned int backtrace_timeout_seconds = 10;

} // namespace

[[noreturn]] EINSUMS_EXPORT void termination_handler(int signum) {
    // One reporter at a time. Two threads taking a fatal signal together both
    // walk into the symbolizer, whose caches are not reentrant, and meet inside
    // them. A late arrival parks rather than racing: it has nothing to add to a
    // report already in flight, and the reporter ends the process for both. If
    // the reporter is the one that wedges, the alarm below gets everybody out.
    static std::atomic_flag reporting;
    if (reporting.test_and_set()) {
        while (true) {
            pause();
        }
    }

    bool attach      = true;
    bool diagnostics = true;

    try {
        attach      = config::get(option::AttachDebugger);
        diagnostics = config::get(option::DiagnosticsOnTerminate);
    } catch (...) {
        attach = true;
    }

    // Report before offering the debugger, not after. attach_debugger() spins until
    // someone attaches, so anything printed behind it is printed only for a developer
    // who was already sitting at the process - which is the one case that did not need
    // it. An unattended run wants the report and never reaches the loop.
    if (diagnostics) {
        // write(2) rather than the iostreams: this runs in a signal handler, where the
        // stream objects may be mid-teardown or the lock behind them already held by
        // the thread we interrupted. The backtrace below allocates and takes locks and
        // so carries the opposite risk, which is why it is attempted second, bounded
        // by an alarm, and allowed to fail. The catch alone was not enough: a
        // symbolizer that blocks never throws.
        auto emit = [](char const *text) { [[maybe_unused]] auto ignored = write(STDERR_FILENO, text, std::strlen(text)); };
        emit("\n=== einsums: fatal signal ===\n");
        emit(signal_name(signum));
        emit("\n");

        struct sigaction alarm_action;
        alarm_action.sa_handler = backtrace_timeout_handler;
        sigemptyset(&alarm_action.sa_mask);
        alarm_action.sa_flags = 0;
        sigaction(SIGALRM, &alarm_action, nullptr);
        alarm(backtrace_timeout_seconds);

        try {
            std::string const trace = util::backtrace();
            if (!trace.empty()) {
                emit("\nbacktrace:\n");
                emit(trace.c_str());
                emit("\n");
            }
        } catch (...) { // NOLINT
            emit("\nbacktrace: unavailable\n");
        }

        // Cancel it before attach_debugger(), which spins on purpose and would
        // otherwise be shot by our own alarm.
        alarm(0);
        emit("=== end einsums fatal signal ===\n");
    }

    if (signum != SIGINT && attach) {
        util::attach_debugger();
    }

    std::abort();
}
#endif

static bool exit_called = false;

void on_exit() noexcept {
    exit_called = true;
}

void on_abort(int) noexcept {
    exit_called = true;
    // _Exit, not exit: this runs from a SIGABRT handler, where exit() would run
    // the static destructors. On a process already on its way down that reaches
    // ~Profiler with other threads still live, and a destructor that throws
    // there turns the abort into a std::terminate inside fwrite, waiting on a
    // stdio lock. Leave without unwinding anything.
    std::_Exit(-1);
}

void set_signal_handlers() {
#if defined(EINSUMS_WINDOWS)
    SetConsoleCtrlHandler(termination_handler, TRUE);
#else
    struct sigaction new_action;
    new_action.sa_handler = termination_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;

    sigaction(SIGINT, &new_action, nullptr);  // Interrupted
    sigaction(SIGBUS, &new_action, nullptr);  // Bus error
    sigaction(SIGFPE, &new_action, nullptr);  // Floating point exception
    sigaction(SIGILL, &new_action, nullptr);  // Illegal instruction
    sigaction(SIGSEGV, &new_action, nullptr); // Segmentation fault
    sigaction(SIGSYS, &new_action, nullptr);  // Bad syscall
#endif
}

void ignore_broken_pipe() {
#if !defined(EINSUMS_WINDOWS)
    // SIGPIPE is not a crash. A consumer that stops reading early is ordinary
    // use - `prog | head`, `prog | grep -q` - and it reaches the writer as a
    // signal whose default disposition is death mid-write. Ignoring it makes
    // the write fail with EPIPE instead, which a caller can see and act on.
    //
    // It must never share the fatal handler. That handler symbolizes, and the
    // write that raised SIGPIPE was inside stdio holding the very lock the
    // symbolizer then waited for, so piping any Einsums program into `head`
    // hung it forever instead of ending it.
    struct sigaction ignore_action;
    ignore_action.sa_handler = SIG_IGN;
    sigemptyset(&ignore_action.sa_mask);
    ignore_action.sa_flags = 0;
    sigaction(SIGPIPE, &ignore_action, nullptr);
#endif
}

Runtime::Runtime(RuntimeConfiguration &&rtcfg, bool initialize) : _rtcfg(std::move(rtcfg)) {
    LabeledSectionInternal("Runtime constructor");
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

        auto                 &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard const vars_guard(runtime_vars); // Lock the variables.

        // Copy over all startup functions registered so far.
        for (StartupFunctionType const &f : runtime_vars.global_pre_startup_functions) {
            add_pre_startup_function(f);
        }

        for (StartupFunctionType const &f : runtime_vars.global_startup_functions) {
            add_startup_function(f);
        }

        for (ShutdownFunctionType const &f : runtime_vars.global_pre_shutdown_functions) {
            add_pre_shutdown_function(f);
        }

        for (ShutdownFunctionType const &f : runtime_vars.global_shutdown_functions) {
            add_shutdown_function(f);
        }
    } catch (std::exception const &e) {
        /// @todo report_exception_and_terminate(e);
    } catch (...) {
        /// @todo report_exception_and_terminate(std::current_exception());
    }
}

Runtime::~Runtime() {
    // Guard against double-finalize (if finalize() was called explicitly before destruction).
    if (runtime_ptr() != this) {
        return; // Already finalized.
    }

    // Run shutdown functions (module cleanup, user-registered shutdown hooks).
    call_shutdown_functions(true); // pre-shutdown
    EINSUMS_LOG_INFO("ran pre-shutdown functions");
    call_shutdown_functions(false); // shutdown
    EINSUMS_LOG_INFO("ran shutdown functions");

#if defined(EINSUMS_HAVE_PROFILER)
    // Shutdown profiler: drain all events, stop consumer thread, stop server.
    profile::Profiler::instance().shutdown();

    try {
        if (config::get(option::ProfileReport)) {
            std::ofstream out(config::get(option::ProfileFilename), config::get(option::ProfileAppend) ? std::ios::ate : std::ios::trunc);
            profile::Profiler::instance().print(config::get(option::ProfileDetailed), out);
        }
    } catch (...) {
    }
#endif

    // Clear the global runtime pointer.
    deinit_global_data();

    EINSUMS_LOG_INFO("einsums shutdown completed");
}

void Runtime::init_global_data() {
    LabeledSectionInternal("Runtime::init_global_data");
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(!runtime_);

    runtime_ = this;
}

void Runtime::deinit_global_data() {
    LabeledSectionInternal("Runtime::deinit_global_data");
    Runtime *&runtime_ = runtime_ptr();
    EINSUMS_ASSERT(runtime_);
    runtime_ = nullptr;
}

void Runtime::add_pre_shutdown_function(ShutdownFunctionType f) {
    LabeledSectionInternal("Runtime::add_pre_shutdown_function");
    std::lock_guard const l(this->lock_);
    _pre_shutdown_functions.push_back(f);
}

void Runtime::add_shutdown_function(ShutdownFunctionType f) {
    LabeledSectionInternal("Runtime::add_shutdown_function");
    std::lock_guard const l(this->lock_);
    _shutdown_functions.push_back(f);
}

void Runtime::add_pre_startup_function(StartupFunctionType f) {
    LabeledSectionInternal("Runtime::add_pre_startup_function");
    std::lock_guard const l(this->lock_);
    _pre_startup_functions.push_back(f);
}

void Runtime::add_startup_function(StartupFunctionType f) {
    LabeledSectionInternal("Runtime::add_startup_function");
    std::lock_guard const l(this->lock_);
    _startup_functions.push_back(f);
}

void Runtime::call_startup_functions(bool pre_startup) {
    if (pre_startup) {
        EINSUMS_LOG_TRACE("Calling pre-startup routines");
        LabeledSectionInternal("Calling pre-startup routines");
        state(RuntimeState::PreStartup);
        for (StartupFunctionType const &f : _pre_startup_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling startup routines");
        LabeledSectionInternal("Calling startup routines");
        state(RuntimeState::Startup);
        for (StartupFunctionType const &f : _startup_functions) {
            f();
        }
    }
}

void Runtime::call_shutdown_functions(bool pre_shutdown) {
    if (pre_shutdown) {
        EINSUMS_LOG_TRACE("Calling pre-shutdown routines");
        LabeledSectionInternal("Calling pre-shutdown routines");
        state(RuntimeState::PreShutdown);
        for (ShutdownFunctionType const &f : _pre_shutdown_functions) {
            f();
        }
    } else {
        EINSUMS_LOG_TRACE("Calling shutdown routines");
        LabeledSectionInternal("Calling shutdown routines");
        state(RuntimeState::Shutdown);
        for (ShutdownFunctionType const &f : _shutdown_functions) {
            f();
        }
    }
}

int Runtime::run(std::function<EinsumsMainFunctionType> const &func) {
    call_startup_functions(true);
    call_startup_functions(false);

    // Set the state to running.
    state(RuntimeState::Running);

    // Wait for profiler viewer to connect if requested. Compiled out with the
    // profiler: there is no Profiler type to ask, and nothing to wait for.
#if defined(EINSUMS_HAVE_PROFILER)
    {
        bool const wait_viewer = config::get(option::ProfileWaitForViewer);
        if (wait_viewer) {
            auto *server = profile::Profiler::instance().server();
            if (server && server->is_running()) {
                EINSUMS_LOG_INFO("Waiting for profiler viewer to connect (--einsums:profile:wait-for-viewer)...");
                std::fprintf(stderr, "\n*** Waiting for profiler viewer to connect on port %d ***\n",
                             static_cast<int>(config::get(option::ProfilePort)));
                std::fprintf(stderr, "*** Launch the viewer and connect, then execution will begin ***\n\n");
                while (!server->has_client()) {
                    server->tick();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                EINSUMS_LOG_INFO("Viewer connected, proceeding with execution");
                std::fprintf(stderr, "*** Viewer connected, starting execution ***\n\n");
            }
        }
    }
#endif

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
    detail::Runtime *rt = runtime_ptr();
    if (nullptr != rt)
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
            EINSUMS_THROW_EXCEPTION(InvalidRuntimeState, "Too late to register a pre-startup function");
            return;
        }
        runtime->add_pre_startup_function(std::move(f));
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.global_pre_startup_functions.emplace_back(std::move(f));
    }
}

void register_startup_function(StartupFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Startup) {
            EINSUMS_THROW_EXCEPTION(InvalidRuntimeState, "Too late to register a startup function");
            return;
        }
        runtime->add_startup_function(std::move(f));
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.global_startup_functions.emplace_back(std::move(f));
    }
}

void register_pre_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::PreShutdown) {
            EINSUMS_THROW_EXCEPTION(InvalidRuntimeState, "Too late to register a pre-shutdown function");
            return;
        }
        runtime->add_pre_shutdown_function(std::move(f));
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.global_pre_shutdown_functions.emplace_back(std::move(f));
    }
}

void register_shutdown_function(ShutdownFunctionType f) {
    auto *runtime = runtime_ptr();
    if (runtime != nullptr) {
        if (runtime->state() > RuntimeState::Shutdown) {
            EINSUMS_THROW_EXCEPTION(InvalidRuntimeState, "Too late to register a shutdown function");
            return;
        }
        runtime->add_pre_shutdown_function(std::move(f));
    } else {
        auto                                &runtime_vars = detail::RuntimeVars::get_singleton();
        std::lock_guard<detail::RuntimeVars> guard(runtime_vars);
        runtime_vars.global_shutdown_functions.emplace_back(std::move(f));
    }
}

EINSUMS_NAMESPACE_END()