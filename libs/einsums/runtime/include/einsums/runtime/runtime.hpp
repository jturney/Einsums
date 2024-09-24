//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/modules/program_options.hpp>
#include <einsums/modules/thread_manager.hpp>
#include <einsums/modules/topology.hpp>
#include <einsums/runtime/runtime_fwd.hpp>
#include <einsums/runtime/shutdown_function.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/runtime/state.hpp>
#include <einsums/runtime/thread_hooks.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/threading_base/callback_notifier.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <einsums/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

///////////////////////////////////////////////////////////////////////
// There is no need to protect these global from thread concurrent
// access as they are accessed during early startup only.
extern std::list<startup_function_type>  global_pre_startup_functions;
extern std::list<startup_function_type>  global_startup_functions;
extern std::list<shutdown_function_type> global_pre_shutdown_functions;
extern std::list<shutdown_function_type> global_shutdown_functions;

///////////////////////////////////////////////////////////////////////////
class EINSUMS_EXPORT runtime {
  public:
    /// Generate a new notification policy instance for the given thread
    /// name prefix
    using notification_policy_type = einsums::threads::callback_notifier;
    virtual notification_policy_type get_notification_policy(char const *prefix);

    runtime_state get_state() const;
    void          set_state(runtime_state s);

    /// The \a einsums_main_function_type is the default function type usable
    /// as the main einsums thread function.
    using einsums_main_function_type = int();

    using einsums_errorsink_function_type = void(std::uint32_t, std::string const &);

    /// Construct a new einsums runtime instance
    explicit runtime(einsums::util::runtime_configuration &rtcfg, bool initialize);

  protected:
    explicit runtime(einsums::util::runtime_configuration &rtcfg);

    void set_notification_policies(notification_policy_type &&notifier);

    /// Common initialization for different constructors
    void init();

  public:
    /// \brief The destructor makes sure all einsums runtime services are
    ///        properly shut down before exiting.
    virtual ~runtime();

    /// \brief Manage list of functions to call on exit
    void on_exit(einsums::util::detail::function<void()> const &f);

    /// \brief Manage runtime 'stopped' state
    void starting();

    /// \brief Call all registered on_exit functions
    void stopping();

    /// This accessor returns whether the runtime instance has been stopped
    bool stopped() const;

    /// \brief access configuration information
    einsums::util::runtime_configuration &get_config();

    einsums::util::runtime_configuration const &get_config() const;

    /// \brief Return the system uptime measure on the thread executing this call
    static std::uint64_t get_system_uptime();

    einsums::threads::detail::topology const &get_topology() const;

    /// \brief Run the einsums runtime system, use the given function for the
    ///        main \a thread and block waiting for all threads to
    ///        finish
    ///
    /// \param func       [in] This is the main function of a einsums
    ///                   application. It will be scheduled for execution
    ///                   by the thread manager as soon as the runtime has
    ///                   been initialized. This function is expected to
    ///                   expose an interface as defined by the typedef
    ///                   \a einsums_main_function_type. This parameter is
    ///                   optional and defaults to none main thread
    ///                   function, in which case all threads have to be
    ///                   scheduled explicitly.
    ///
    /// \note             The parameter \a func is optional. If no function
    ///                   is supplied, the runtime system will simply wait
    ///                   for the shutdown action without explicitly
    ///                   executing any main thread.
    ///
    /// \returns          This function will return the value as returned
    ///                   as the result of the invocation of the function
    ///                   object given by the parameter \p func.
    virtual int run(einsums::util::detail::function<einsums_main_function_type> const &func);

    /// \brief Run the einsums runtime system, initially use the given number
    ///        of (OS) threads in the thread-manager and block waiting for
    ///        all threads to finish.
    ///
    /// \returns          This function will always return 0 (zero).
    virtual int run();

    /// Rethrow any stored exception (to be called after stop())
    virtual void rethrow_exception();

    /// \brief Start the runtime system
    ///
    /// \param func       [in] This is the main function of a einsums
    ///                   application. It will be scheduled for execution
    ///                   by the thread manager as soon as the runtime has
    ///                   been initialized. This function is expected to
    ///                   expose an interface as defined by the typedef
    ///                   \a einsums_main_function_type.
    /// \param blocking   [in] This allows to control whether this
    ///                   call blocks until the runtime system has been
    ///                   stopped. If this parameter is \a true the
    ///                   function \a runtime#start will call
    ///                   \a runtime#wait internally.
    ///
    /// \returns          If a blocking is a true, this function will
    ///                   return the value as returned as the result of the
    ///                   invocation of the function object given by the
    ///                   parameter \p func. Otherwise it will return zero.
    virtual int start(einsums::util::detail::function<einsums_main_function_type> const &func, bool blocking = false);

    /// \brief Start the runtime system
    ///
    /// \param blocking   [in] This allows to control whether this
    ///                   call blocks until the runtime system has been
    ///                   stopped. If this parameter is \a true the
    ///                   function \a runtime#start will call
    ///                   \a runtime#wait internally .
    ///
    /// \returns          If a blocking is a true, this function will
    ///                   return the value as returned as the result of the
    ///                   invocation of the function object given by the
    ///                   parameter \p func. Otherwise it will return zero.
    virtual int start(bool blocking = false);

    /// \brief Wait for the shutdown action to be executed
    ///
    /// \returns          This function will return the value as returned
    ///                   as the result of the invocation of the function
    ///                   object given by the parameter \p func.
    virtual int wait();

    /// \brief Initiate termination of the runtime system
    ///
    /// \param blocking   [in] This allows to control whether this
    ///                   call blocks until the runtime system has been
    ///                   fully stopped. If this parameter is \a false then
    ///                   this call will initiate the stop action but will
    ///                   return immediately. Use a second call to stop
    ///                   with this parameter set to \a true to wait for
    ///                   all internal work to be completed.
    virtual void stop(bool blocking = true);

    /// \brief Suspend the runtime system
    virtual void suspend();

    ///    \brief Resume the runtime system
    virtual void resume();

    virtual void finalize();

    /// \brief Allow access to the thread manager instance used by the einsums
    ///        runtime.
    virtual einsums::threads::detail::thread_manager &get_thread_manager();

    /// \brief Report a non-recoverable error to the runtime system
    ///
    /// \param num_thread [in] The number of the operating system thread
    ///                   the error has been detected in.
    /// \param e          [in] This is an instance encapsulating an
    ///                   exception which lead to this function call.
    virtual bool report_error(std::size_t num_thread, std::exception_ptr const &e, bool terminate_all = true);

    /// \brief Report a non-recoverable error to the runtime system
    ///
    /// \param e          [in] This is an instance encapsulating an
    ///                   exception which lead to this function call.
    ///
    /// \note This function will retrieve the number of the current
    ///       shepherd thread and forward to the report_error function
    ///       above.
    virtual bool report_error(std::exception_ptr const &e, bool terminate_all = true);

    /// Add a function to be executed inside a einsums thread before einsums_main
    /// but guaranteed to be executed before any startup function registered
    /// with \a add_startup_function.
    ///
    /// \param  f   The function 'f' will be called from inside a einsums
    ///             thread before einsums_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    ///
    /// \note       The difference to a startup function is that all
    ///             pre-startup functions will be (system-wide) executed
    ///             before any startup function.
    virtual void add_pre_startup_function(startup_function_type f);

    /// Add a function to be executed inside a einsums thread before einsums_main
    ///
    /// \param  f   The function 'f' will be called from inside a einsums
    ///             thread before einsums_main is executed. This is very useful
    ///             to setup the runtime environment of the application
    ///             (install performance counters, etc.)
    virtual void add_startup_function(startup_function_type f);

    /// Add a function to be executed inside a einsums thread during
    /// einsums::finalize, but guaranteed before any of the shutdown functions
    /// is executed.
    ///
    /// \param  f   The function 'f' will be called from inside a einsums
    ///             thread while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    ///
    /// \note       The difference to a shutdown function is that all
    ///             pre-shutdown functions will be (system-wide) executed
    ///             before any shutdown function.
    virtual void add_pre_shutdown_function(shutdown_function_type f);

    /// Add a function to be executed inside a einsums thread during einsums::finalize
    ///
    /// \param  f   The function 'f' will be called from inside a einsums
    ///             thread while einsums::finalize is executed. This is very
    ///             useful to tear down the runtime environment of the
    ///             application (uninstall performance counters, etc.)
    virtual void add_shutdown_function(shutdown_function_type f);

    /// \brief Register an external OS-thread with einsums
    ///
    /// This function should be called from any OS-thread which is external to
    /// einsums (not created by einsums), but which needs to access einsums functionality,
    /// such as setting a value on a promise or similar.
    ///
    /// \param name             [in] The name to use for thread registration.
    /// \param num              [in] The sequence number to use for thread
    ///                         registration. The default for this parameter
    ///                         is zero.
    /// \note The function will compose a thread name of the form
    ///       '<name>-thread#<num>' which is used to register the thread. It
    ///       is the user's responsibility to ensure that each (composed)
    ///       thread name is unique. einsums internally uses the following names
    ///       for the threads it creates, do not reuse those:
    ///
    ///         'main', 'io', 'timer', 'parcel', 'worker'
    ///
    /// \note This function should be called for each thread exactly once. It
    ///       will fail if it is called more than once.
    ///
    /// \returns This function will return whether the requested operation
    ///          succeeded or not.
    ///
    virtual bool register_thread(char const *name, std::size_t num = 0, error_code &ec = throws);

    /// \brief Unregister an external OS-thread with einsums
    ///
    /// This function will unregister any external OS-thread from einsums.
    ///
    /// \note This function should be called for each thread exactly once. It
    ///       will fail if it is called more than once. It will fail as well
    ///       if the thread has not been registered before (see
    ///       \a register_thread).
    ///
    /// \returns This function will return whether the requested operation
    ///          succeeded or not.
    ///
    virtual bool unregister_thread();

    notification_policy_type::on_startstop_type on_start_func() const;
    notification_policy_type::on_startstop_type on_stop_func() const;
    notification_policy_type::on_error_type     on_error_func() const;

    notification_policy_type::on_startstop_type on_start_func(notification_policy_type::on_startstop_type &&);
    notification_policy_type::on_startstop_type on_stop_func(notification_policy_type::on_startstop_type &&);
    notification_policy_type::on_error_type     on_error_func(notification_policy_type::on_error_type &&);

    virtual std::size_t get_num_worker_threads() const;

    virtual std::uint32_t assign_cores(std::string const &, std::uint32_t) { return std::uint32_t(-1); }

    virtual std::uint32_t assign_cores() { return std::uint32_t(-1); }

  protected:
    void init_global_data();
    void deinit_global_data();

    einsums::threads::detail::thread_result_type
    run_helper(einsums::util::detail::function<runtime::einsums_main_function_type> const &func, int &result, bool call_startup_functions);

    void wait_helper(std::mutex &mtx, std::condition_variable &cond, bool &running);

    // list of functions to call on exit
    using on_exit_type = std::vector<einsums::util::detail::function<void()>>;
    on_exit_type       on_exit_functions_;
    mutable std::mutex mtx_;

    einsums::util::runtime_configuration rtcfg_;

    // topology and affinity data
    einsums::threads::detail::topology &topology_;

    std::atomic<runtime_state> state_;

    // support tying in external functions to be called for thread events
    notification_policy_type::on_startstop_type on_start_func_;
    notification_policy_type::on_startstop_type on_stop_func_;
    notification_policy_type::on_error_type     on_error_func_;

    int result_;

    std::exception_ptr exception_;

    notification_policy_type                                  notifier_;
    std::unique_ptr<einsums::threads::detail::thread_manager> thread_manager_;

  private:
    /// \brief Helper function to stop the runtime.
    ///
    /// \param blocking   [in] This allows to control whether this
    ///                   call blocks until the runtime system has been
    ///                   fully stopped. If this parameter is \a false then
    ///                   this call will initiate the stop action but will
    ///                   return immediately. Use a second call to stop
    ///                   with this parameter set to \a true to wait for
    ///                   all internal work to be completed.
    void stop_helper(bool blocking, std::condition_variable &cond, std::mutex &mtx);

    void deinit_tss_helper(char const *context, std::size_t num);

    void init_tss_ex(char const *context, std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name,
                     char const *postfix);

    void init_tss_helper(char const *context, std::size_t local_thread_num, std::size_t global_thread_num, char const *pool_name,
                         char const *postfix);

    void notify_finalize();
    void wait_finalize();

    void call_startup_functions(bool pre_startup);
    void call_shutdown_functions(bool pre_shutdown);

    std::list<startup_function_type>  pre_startup_functions_;
    std::list<startup_function_type>  startup_functions_;
    std::list<shutdown_function_type> pre_shutdown_functions_;
    std::list<shutdown_function_type> shutdown_functions_;

    bool                    stop_called_;
    bool                    stop_done_;
    std::condition_variable wait_condition_;
};

EINSUMS_EXPORT void set_signal_handlers();

EINSUMS_EXPORT char const *get_runtime_state_name(einsums::runtime_state st);

namespace util {
///////////////////////////////////////////////////////////////////////////
// retrieve the command line arguments for the current process
EINSUMS_EXPORT bool retrieve_commandline_arguments(einsums::program_options::options_description const &app_options,
                                                   einsums::program_options::variables_map             &vm);

///////////////////////////////////////////////////////////////////////////
// retrieve the command line arguments for the current process
EINSUMS_EXPORT bool retrieve_commandline_arguments(std::string const &appname, einsums::program_options::variables_map &vm);
} // namespace util

namespace threads {
/// \brief Returns the stack size name.
///
/// Get the readable string representing the given stack size constant.
///
/// \param size this represents the stack size
EINSUMS_EXPORT char const *get_stack_size_name(std::ptrdiff_t size);

/// \brief Returns the default stack size.
///
/// Get the default stack size in bytes.
EINSUMS_EXPORT std::ptrdiff_t get_default_stack_size();

/// \brief Returns the stack size corresponding to the given stack size
///        enumeration.
///
/// Get the stack size corresponding to the given stack size enumeration.
///
/// \param size this represents the stack size
EINSUMS_EXPORT std::ptrdiff_t get_stack_size(execution::thread_stacksize);
} // namespace threads

} // namespace einsums::detail

namespace einsums {
/// Returns true when the runtime is initialized, false otherwise.
///
/// Returns true while in a @ref einsums::init call, or between calls of @ref einsums::start and @ref
/// einsums::stop, otherwise false.
///
/// Added in 0.22.0.
EINSUMS_EXPORT bool is_runtime_initialized() noexcept;
} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>
