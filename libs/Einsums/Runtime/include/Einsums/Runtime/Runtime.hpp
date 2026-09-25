//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/RuntimeConfiguration.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <cstdint>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @struct InvalidRuntimeState
 *
 * Indicates that the code is handling data that is uninitialized.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT InvalidRuntimeState : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @enum RuntimeState
 *
 * @brief Holds the possible states for the runtime.
 *
 * @versionadded{1.0.0}
 */
enum class APIARY_EXPOSE RuntimeState : std::int8_t {
    Invalid        = -1,      /**< The state is invalid. */
    Initialized    = 0,       /**< The runtime has been initialized. */
    PreStartup     = 1,       /**< The runtime is running the pre-startup functions. */
    Startup        = 2,       /**< The runtime is running the startup functions. */
    PreMain        = 3,       /**< The runtime is preparing to run the main function. */
    Starting       = 4,       /**< The runtime is starting the main function. */
    Running        = 5,       /**< The main function is running. */
    PreShutdown    = 6,       /**< The pre-shutdown functions are running. */
    Shutdown       = 7,       /**< The shutdown functions are running. */
    Stopping       = 8,       /**< The runtime is stopping. */
    Terminating    = 9,       /**< The runtime is terminating. */
    Stopped        = 10,      /**< The runtime has stopped. */
    LastValidState = Stopped, /**< Indicates the last valid state. Anything past this is considered invalid. */
};

namespace detail {

class EINSUMS_EXPORT RuntimeVars : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(RuntimeVars)

  public:
    std::list<StartupFunctionType>  global_pre_startup_functions;
    std::list<StartupFunctionType>  global_startup_functions;
    std::list<ShutdownFunctionType> global_pre_shutdown_functions;
    std::list<ShutdownFunctionType> global_shutdown_functions;

  private:
    explicit RuntimeVars() = default;
};

struct EINSUMS_EXPORT Runtime : public design_pats::Lockable<std::recursive_mutex> {
    virtual ~Runtime();

    /// The \a EinsumsMainFunctionType is the default function type used as
    /// the main Einsums function.
    /// @versionadded{1.0.0}
    using EinsumsMainFunctionType = int();

    /// Construct a new Einsums runtime instance
    /// @versionadded{1.0.0}
    Runtime(RuntimeConfiguration &&rtcfg, bool initialize);

    RuntimeState state() const;
    void         state(RuntimeState s);

    RuntimeConfiguration       &config();
    RuntimeConfiguration const &config() const;

    /// Add a function to be executed before einsums_main
    /// but guaranteed to be executed before any startup function registered
    /// with \a add_startup_function.
    ///
    /// \param  f   The function 'f' will be called  before pika_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    ///
    /// \note       The difference to a startup function is that all
    ///             pre-startup functions will be (system-wide) executed
    ///             before any startup function.
    /// @versionadded{1.0.0}
    virtual void add_pre_startup_function(StartupFunctionType f);

    /// Add a function to be executed before einsums_main
    ///
    /// \param  f   The function 'f' will be called before einsums_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    /// @versionadded{1.0.0}
    virtual void add_startup_function(StartupFunctionType f);

    /// Add a function to be executed during
    /// einsums::finalize, but guaranteed before any of the shutdown functions
    /// is executed.
    ///
    /// \param  f   The function 'f' will be called while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    ///
    /// \note       The difference to a shutdown function is that all
    ///             pre-shutdown functions will be (system-wide) executed
    ///             before any shutdown function.
    /// @versionadded{1.0.0}
    virtual void add_pre_shutdown_function(ShutdownFunctionType f);

    /// Add a function to be executed during einsums::finalize
    ///
    /// \param  f   The function 'f' will be called while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    /// @versionadded{1.0.0}
    virtual void add_shutdown_function(ShutdownFunctionType f);

    virtual int run(std::function<EinsumsMainFunctionType> const &func);
    virtual int run();

  protected:
    /// Common initialization for different constructors
    void init();
    void init_global_data();
    void deinit_global_data();

  private:
    void call_startup_functions(bool pre_startup);
    void call_shutdown_functions(bool pre_shutdown);

    friend int einsums::finalize();

    std::list<StartupFunctionType>  _pre_startup_functions;
    std::list<StartupFunctionType>  _startup_functions;
    std::list<ShutdownFunctionType> _pre_shutdown_functions;
    std::list<ShutdownFunctionType> _shutdown_functions;

  protected:
    RuntimeConfiguration _rtcfg;

    std::atomic<RuntimeState> _state{RuntimeState::Invalid};
};

EINSUMS_EXPORT void on_exit() noexcept;
EINSUMS_EXPORT void on_abort(int signal) noexcept;
EINSUMS_EXPORT void set_signal_handlers();

/// Make a write to a pipe nobody is reading fail rather than kill the process.
///
/// Deliberately not part of @ref set_signal_handlers: that installs crash
/// reporting, and a caller who turns crash reporting off has not thereby asked
/// to be killed mid-write by `head`. The whole test suite runs with
/// `--einsums:debug:no-install-signal-handlers`, which is precisely where the
/// two concerns come apart.
EINSUMS_EXPORT void ignore_broken_pipe();

/// Export the profiler session, stop the profiler, and write its report.
///
/// Both teardown paths call this: @ref einsums::finalize when a caller finalizes explicitly, and
/// `~Runtime` when nobody does. They ran identical copies of this sequence, which is how the
/// session export came to exist on only one of them.
///
/// Order is the contract. The session export needs a live server, so it runs before
/// `Profiler::shutdown`; the text report reads the aggregated tree, which outlives shutdown, so
/// it keeps running after as it always has.
///
/// A no-op when the profiler is compiled out. Never throws: teardown is not a place to fail.
EINSUMS_EXPORT void shutdown_profiler_and_report() noexcept;
} // namespace detail

/**
 * @brief Returns a reference to the current Runtime structure
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT detail::Runtime &runtime();

/**
 * @brief Returns a pointer to the current Runtime structure.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT detail::Runtime *&runtime_ptr();

/**
 * @brief Gets a reference to the current runtime configuration structure.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT RuntimeConfiguration &runtime_config();

/// \brief Test whether the runtime system is currently running.
///
/// This function returns whether the runtime system is currently running
/// or not, e.g.  whether the current state of the runtime system is
/// \a einsums::RuntimeState::Running
///
/// \note   This function needs to be executed on an einsums-thread. It will
///         return false otherwise.
/// @versionadded{1.0.0}
APIARY_EXPOSE EINSUMS_EXPORT bool is_running();

/// The name of a runtime state, for diagnostics: ``"Running"``, ``"Stopped"`` and so on, or
/// ``"Unknown"`` for a value outside the enumeration.
/// @versionadded{2.0.0}
[[nodiscard]] constexpr std::string_view runtime_state_name(RuntimeState state) noexcept {
    switch (state) {
    case RuntimeState::Invalid:
        return "Invalid";
    case RuntimeState::Initialized:
        return "Initialized";
    case RuntimeState::PreStartup:
        return "PreStartup";
    case RuntimeState::Startup:
        return "Startup";
    case RuntimeState::PreMain:
        return "PreMain";
    case RuntimeState::Starting:
        return "Starting";
    case RuntimeState::Running:
        return "Running";
    case RuntimeState::PreShutdown:
        return "PreShutdown";
    case RuntimeState::Shutdown:
        return "Shutdown";
    case RuntimeState::Stopping:
        return "Stopping";
    case RuntimeState::Terminating:
        return "Terminating";
    case RuntimeState::Stopped:
        return "Stopped";
    }
    return "Unknown";
}

/// @c fmt formats a @ref RuntimeState as its name, so ``fmt::format("{}", state)`` needs no call to
/// @ref runtime_state_name.
[[nodiscard]] constexpr std::string_view format_as(RuntimeState state) noexcept {
    return runtime_state_name(state);
}

EINSUMS_NAMESPACE_END()
