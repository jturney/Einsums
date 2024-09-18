//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/config/warnings_prefix.hpp>

#include <cstddef>

namespace einsums::threads::detail {
/// Set the global thread id to thread local storage.
EINSUMS_EXPORT std::size_t set_global_thread_num_tss(std::size_t num);

/// Get the global thread id from thread local storage.
EINSUMS_EXPORT std::size_t get_global_thread_num_tss();

/// Set the local thread id to thread local storage.
EINSUMS_EXPORT std::size_t set_local_thread_num_tss(std::size_t num);

/// Get the local thread id from thread local storage.
EINSUMS_EXPORT std::size_t get_local_thread_num_tss();

/// Set the thread pool id to thread local storage.
EINSUMS_EXPORT std::size_t set_thread_pool_num_tss(std::size_t num);

/// Get the thread pool id from thread local storage.
EINSUMS_EXPORT std::size_t get_thread_pool_num_tss();

///////////////////////////////////////////////////////////////////////////
struct reset_tss_helper {
    reset_tss_helper(std::size_t global_thread_num) : global_thread_num_(set_global_thread_num_tss(global_thread_num)) {}

    ~reset_tss_helper() { set_global_thread_num_tss(global_thread_num_); }

    std::size_t previous_global_thread_num() const { return global_thread_num_; }

  private:
    std::size_t global_thread_num_;
};
} // namespace einsums::threads::detail

namespace einsums {
///////////////////////////////////////////////////////////////////////////
/// \brief Return the number of the current OS-thread running in the
///        runtime instance the current einsums-thread is executed with.
///
/// This function returns the zero based index of the OS-thread which
/// executes the current einsums-thread.
///
/// \note   The returned value is zero based and its maximum value is
///         smaller than the overall number of OS-threads executed (as
///         returned by \a get_os_thread_count().
///
/// \note   This function needs to be executed on a einsums-thread. It will
///         fail otherwise (it will return -1).
EINSUMS_EXPORT std::size_t get_worker_thread_num();

///////////////////////////////////////////////////////////////////////////
/// \brief Return the number of the current OS-thread running in the current
///        thread pool the current einsums-thread is executed with.
///
/// This function returns the zero based index of the OS-thread on the
/// current thread pool which executes the current einsums-thread.
///
/// \note The returned value is zero based and its maximum value is smaller
///       than the number of OS-threads executed on the current thread pool.
///       It will return -1 if the current thread is not a known thread or
///       if the runtime is not in running state.
///
/// \note This function needs to be executed on a einsums-thread. It will fail
///         otherwise (it will return -1).
EINSUMS_EXPORT std::size_t get_local_worker_thread_num();

///////////////////////////////////////////////////////////////////////////
/// \brief Return the number of the current thread pool the current
/// einsums-thread is executed with.
///
/// This function returns the zero based index of the thread pool which
/// executes the current einsums-thread.
///
/// \note The returned value is zero based and its maximum value is smaller
///       than the number of thread pools started by the runtime. It will
///       return -1 if the current thread pool is not a known thread pool or
///       if the runtime is not in running state.
///
/// \note This function needs to be executed on a einsums-thread. It will fail
///         otherwise (it will return -1).
EINSUMS_EXPORT std::size_t get_thread_pool_num();

} // namespace einsums

#include <einsums/config/warnings_suffix.hpp>
