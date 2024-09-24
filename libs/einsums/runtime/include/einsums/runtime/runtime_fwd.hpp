//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/functional/function.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/modules/runtime_configuration.hpp>
#include <einsums/runtime/config_entry.hpp>
#include <einsums/runtime/get_os_thread_count.hpp>
#include <einsums/runtime/get_thread_name.hpp>
#include <einsums/runtime/report_error.hpp>
#include <einsums/runtime/shutdown_function.hpp>
#include <einsums/runtime/startup_function.hpp>
#include <einsums/runtime/thread_hooks.hpp>
#include <einsums/runtime/thread_pool_helpers.hpp>
#include <einsums/thread_manager/thread_manager_fwd.hpp>
#include <einsums/threading_base/scheduler_base.hpp>
#include <einsums/threading_base/thread_num_tss.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace einsums::detail {

class EINSUMS_EXPORT runtime;

/// The function \a get_runtime returns a reference to the (thread
/// specific) runtime instance.
EINSUMS_EXPORT runtime  &get_runtime();
EINSUMS_EXPORT runtime *&get_runtime_ptr();

/// Register the current kernel thread with einsums, this should be done once
/// for each external OS-thread intended to invoke einsums functionality.
/// Calling this function more than once will return false.
EINSUMS_EXPORT bool register_thread(runtime *rt, char const *name, error_code &ec = throws);

/// Unregister the thread from einsums, this should be done once in
/// the end before the external thread exists.
EINSUMS_EXPORT void unregister_thread(runtime *rt);

/// Register a function to be called during system shutdown
EINSUMS_EXPORT bool register_on_exit(einsums::util::detail::function<void()> const &);

///////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT einsums::util::runtime_configuration const &get_config();

/// \endcond

///////////////////////////////////////////////////////////////////////////
/// \brief Test whether the runtime system is currently being started.
///
/// This function returns whether the runtime system is currently being
/// started or not, e.g. whether the current state of the runtime system is
/// \a einsums:runtime_state::startup
///
/// \note   This function needs to be executed on a einsums-thread. It will
///         return false otherwise.
EINSUMS_EXPORT bool is_starting();

///////////////////////////////////////////////////////////////////////////
/// \brief Test if einsums runs in fault-tolerant mode
///
/// This function returns whether the runtime system is running
/// in fault-tolerant mode
EINSUMS_EXPORT bool tolerate_node_faults();

///////////////////////////////////////////////////////////////////////////
/// \brief Test whether the runtime system is currently running.
///
/// This function returns whether the runtime system is currently running
/// or not, e.g.  whether the current state of the runtime system is
/// \a einsums:runtime_state::running
///
/// \note   This function needs to be executed on a einsums-thread. It will
///         return false otherwise.
EINSUMS_EXPORT bool is_running();

///////////////////////////////////////////////////////////////////////////
/// \brief Test whether the runtime system is currently stopped.
///
/// This function returns whether the runtime system is currently stopped
/// or not, e.g.  whether the current state of the runtime system is
/// \a einsums:runtime_state::stopped
///
/// \note   This function needs to be executed on a einsums-thread. It will
///         return false otherwise.
EINSUMS_EXPORT bool is_stopped();

///////////////////////////////////////////////////////////////////////////
/// \brief Test whether the runtime system is currently being shut down.
///
/// This function returns whether the runtime system is currently being
/// shut down or not, e.g.  whether the current state of the runtime system
/// is \a einsums:runtime_state::stopped or \a einsums:runtime_state::shutdown
///
/// \note   This function needs to be executed on a einsums-thread. It will
///         return false otherwise.
EINSUMS_EXPORT bool is_stopped_or_shutting_down();

///////////////////////////////////////////////////////////////////////////
/// \brief Return the system uptime measure on the thread executing this call.
///
/// This function returns the system uptime measured in nanoseconds for the
/// thread executing this call. If the function is called while no einsums
/// runtime system is active, it will return zero.
EINSUMS_EXPORT std::uint64_t get_system_uptime();

/// \cond NOINTERNAL
/// Reset internal (round robin) thread distribution scheme
EINSUMS_EXPORT void reset_thread_distribution();

/// Set the new scheduler mode
EINSUMS_EXPORT void set_scheduler_mode(threads::scheduler_mode new_mode);

/// Add the given flags to the scheduler mode
EINSUMS_EXPORT void add_scheduler_mode(threads::scheduler_mode to_add);

/// Remove the given flags from the scheduler mode
EINSUMS_EXPORT void remove_scheduler_mode(threads::scheduler_mode to_remove);

/// Get the global topology instance
EINSUMS_EXPORT einsums::threads::detail::topology const &get_topology();
/// \endcond

EINSUMS_EXPORT void on_exit() noexcept;
EINSUMS_EXPORT void on_abort(int signal) noexcept;
EINSUMS_EXPORT void handle_print_bind(std::size_t num_threads);
} // namespace einsums::detail

namespace einsums {
///////////////////////////////////////////////////////////////////////////
/// \brief Return the number of worker OS- threads used to execute einsums
///        threads
///
/// This function returns the number of OS-threads used to execute einsums
/// threads. If the function is called while no einsums runtime system is active,
/// it will return zero.
EINSUMS_EXPORT std::size_t get_num_worker_threads();
} // namespace einsums
