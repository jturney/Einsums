//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/modules/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <utility>

namespace einsums::detail {

// Stores the information about the hostname of the process the exception
// has been raised on. This information will show up in error messages
// under the [hostname] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_hostname, std::string);

// Stores the information about the pid of the OS process the exception
// has been raised on. This information will show up in error messages
// under the [pid] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_pid, std::int64_t);

// Stores the information about the shepherd thread the exception has been
// raised on. This information will show up in error messages under the
// [shepherd] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_shepherd, std::size_t);

// Stores the information about the einsums thread the exception has been
// raised on. This information will show up in error messages under the
// [thread_id] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_thread_id, std::size_t);

// Stores the information about the einsums thread name the exception has been
// raised on. This information will show up in error messages under the
// [thread_name] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_thread_name, std::string);

// Stores the information about the stack backtrace at the point the
// exception has been raised at. This information will show up in error
// messages under the [stack_trace] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_stacktrace, std::string);

// Stores the full execution environment of the process the exception
// has been raised in. This information will show up in error messages
// under the [env] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_env, std::string);

// Stores the full einsums configuration information of the process the
// exception has been raised in. This information will show up in error
// messages under the [config] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_config, std::string);

// Stores the current runtime state. This information will show up in
// error messages under the [state] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_state, std::string);

// Stores additional auxiliary information (such as information about
// the current parcel). This information will show up in error messages
// under the [auxinfo] tag.
EINSUMS_DEFINE_ERROR_INFO(throw_auxinfo, std::string);

// Portably extract the current execution environment
EINSUMS_EXPORT std::string get_execution_environment();

// Report an early or late exception and locally abort execution. There
// isn't anything more we could do.
[[noreturn]] EINSUMS_EXPORT void report_exception_and_terminate(std::exception const &);
[[noreturn]] EINSUMS_EXPORT void report_exception_and_terminate(std::exception_ptr const &);
[[noreturn]] EINSUMS_EXPORT void report_exception_and_terminate(einsums::exception const &);

// Report an early or late exception and locally exit execution. There
// isn't anything more we could do. The exception will be re-thrown
// from einsums::init
EINSUMS_EXPORT void report_exception_and_continue(std::exception const &);
EINSUMS_EXPORT void report_exception_and_continue(std::exception_ptr const &);
EINSUMS_EXPORT void report_exception_and_continue(einsums::exception const &);

EINSUMS_EXPORT einsums::exception_info construct_exception_info(std::string const &func, std::string const &file, long line,
                                                                std::string const &back_trace, std::uint32_t node,
                                                                std::string const &hostname, std::int64_t pid, std::size_t shepherd,
                                                                std::size_t thread_id, std::string const &thread_name,
                                                                std::string const &env, std::string const &config,
                                                                std::string const &state_name, std::string const &auxinfo);

template <typename Exception>
EINSUMS_EXPORT std::exception_ptr construct_exception(Exception const &e, einsums::exception_info info);

EINSUMS_EXPORT void pre_exception_handler();

using get_full_build_string_type = std::function<std::string()>;
EINSUMS_EXPORT void set_get_full_build_string(get_full_build_string_type f);
EINSUMS_EXPORT std::string get_full_build_string();

EINSUMS_EXPORT einsums::exception_info custom_exception_info(std::source_location const& location,
                                                             std::string const &auxinfo);

///////////////////////////////////////////////////////////////////////////
/// \brief Extract the diagnostic information embedded in the given
/// exception and return a string holding a formatted message.
///
/// The function \a einsums::detail::diagnostic_information can be used to extract all
/// diagnostic information stored in the given exception instance as a
/// formatted string. This simplifies debug output as it composes the
/// diagnostics into one, easy to use function call. This includes
/// the name of the source file and line number, the sequence number of the
/// OS-thread and the einsums-thread id and the stack backtrace
/// of the point where the original exception was thrown.
///
/// \param xi   The parameter \p e will be inspected for all diagnostic
///             information elements which have been stored at the point
///             where the exception was thrown. This parameter can be one
///             of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \returns    The formatted string holding all of the available
///             diagnostic information stored in the given exception
///             instance.
///
/// \throws     std#bad_alloc (if any of the required allocation operations
///             fail)
///
/// \see        \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::string diagnostic_information(exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string diagnostic_information(E const &e) {
    return invoke_with_exception_info(e,
                                      [](exception_info const *xi) { return xi ? diagnostic_information(*xi) : std::string("<unknown>"); });
}
/// \endcond

///////////////////////////////////////////////////////////////////////////
// Extract elements of the diagnostic information embedded in the given
// exception.

/// \brief Return the hostname of the process where the exception was
///        thrown.
///
/// The function \a einsums::get_error_host_name can be used to extract the
/// diagnostic information element representing the host name as stored in
/// the given exception instance.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \returns    The hostname of the process where the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return and empty string.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information()
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error()
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::string get_error_host_name(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_host_name(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_host_name(*xi) : std::string(); });
}
/// \endcond

/// \brief Return the (operating system) process id of the process where
///        the exception was thrown.
///
/// The function \a einsums::get_error_process_id can be used to extract the
/// diagnostic information element representing the process id as stored in
/// the given exception instance.
///
/// \returns    The process id of the OS-process which threw the exception
///             If the exception instance does not hold
///             this information, the function will return 0.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     nothing
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::int64_t get_error_process_id(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::int64_t get_error_process_id(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_process_id(*xi) : -1; });
}
/// \endcond

/// \brief Return the environment of the OS-process at the point the
///        exception was thrown.
///
/// The function \a einsums::get_error_env can be used to extract the
/// diagnostic information element representing the environment of the
/// OS-process collected at the point the exception was thrown.
///
/// \returns    The environment from the point the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::string get_error_env(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_env(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_env(*xi) : std::string("<unknown>"); });
}
/// \endcond

/// \brief Return the stack backtrace from the point the exception was thrown.
///
/// The function \a einsums::get_error_backtrace can be used to extract the
/// diagnostic information element representing the stack backtrace
/// collected at the point the exception was thrown.
///
/// \returns    The stack back trace from the point the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::string get_error_backtrace(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_backtrace(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_backtrace(*xi) : std::string(); });
}
/// \endcond

/// \brief Return the sequence number of the OS-thread used to execute
///        einsums-threads from which the exception was thrown.
///
/// The function \a einsums::get_error_os_thread can be used to extract the
/// diagnostic information element representing the sequence number  of the
/// OS-thread as stored in the given exception instance.
///
/// \returns    The sequence number of the OS-thread used to execute the
///             einsums-thread from which the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return std::size(-1).
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     nothing
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_thread_id(),
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::size_t get_error_os_thread(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::size_t get_error_os_thread(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_os_thread(*xi) : std::size_t(-1); });
}
/// \endcond

/// \brief Return the unique thread id of the einsums-thread from which the
///        exception was thrown.
///
/// The function \a einsums::get_error_thread_id can be used to extract the
/// diagnostic information element representing the einsums-thread id
/// as stored in the given exception instance.
///
/// \returns    The unique thread id of the einsums-thread from which the
///             exception was thrown. If the exception instance
///             does not hold this information, the function will return
///             std::size_t(0).
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     nothing
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread()
///             \a einsums::get_error_thread_description(), \a einsums::get_error(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error_what(), \a einsums::get_error_config(),
///             \a einsums::get_error_state()
///
EINSUMS_EXPORT std::size_t get_error_thread_id(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::size_t get_error_thread_id(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_thread_id(*xi) : std::size_t(-1); });
}
/// \endcond

/// \brief Return any additionally available thread description of the
///        einsums-thread from which the exception was thrown.
///
/// The function \a einsums::get_error_thread_description can be used to extract the
/// diagnostic information element representing the additional thread
/// description as stored in the given exception instance.
///
/// \returns    Any additionally available thread description of the
///             einsums-thread from which the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error(), \a einsums::get_error_state(),
///             \a einsums::get_error_what(), \a einsums::get_error_config()
///
EINSUMS_EXPORT std::string get_error_thread_description(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_thread_description(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_thread_description(*xi) : std::string(); });
}
/// \endcond

/// \brief Return the einsums configuration information point from which the
///        exception was thrown.
///
/// The function \a einsums::get_error_config can be used to extract the
/// einsums configuration information element representing the full einsums
/// configuration information as stored in the given exception instance.
///
/// \returns    Any additionally available einsums configuration information
///             the point from which the exception was
///             thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error(), \a einsums::get_error_state()
///             \a einsums::get_error_what(), \a einsums::get_error_thread_description()
///
EINSUMS_EXPORT std::string get_error_config(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_config(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_config(*xi) : std::string(); });
}
/// \endcond

/// \brief Return the einsums runtime state information at which the exception
///        was thrown.
///
/// The function \a einsums::get_error_state can be used to extract the
/// einsums runtime state information element representing the state the
/// runtime system is currently in as stored in the given exception
/// instance.
///
/// \returns    The point runtime state at the point at which the exception
///             was thrown. If the exception instance does not hold
///             this information, the function will return an empty string.
///
/// \param xi   The parameter \p e will be inspected for the requested
///             diagnostic information elements which have been stored at
///             the point where the exception was thrown. This parameter
///             can be one of the following types: \a einsums::exception_info,
///             \a einsums::error_code, \a std::exception, or
///             \a std::exception_ptr.
///
/// \throws     std#bad_alloc (if one of the required allocations fails)
///
/// \see        \a einsums::detail::diagnostic_information(), \a einsums::get_error_host_name(),
///             \a einsums::get_error_process_id(), \a einsums::get_error_function_name(),
///             \a einsums::get_error_file_name(), \a einsums::get_error_line_number(),
///             \a einsums::get_error_os_thread(), \a einsums::get_error_thread_id(),
///             \a einsums::get_error_backtrace(), \a einsums::get_error_env(),
///             \a einsums::get_error(),
///             \a einsums::get_error_what(), \a einsums::get_error_thread_description()
///
EINSUMS_EXPORT std::string get_error_state(einsums::exception_info const &xi);

/// \cond NOINTERNAL
template <typename E>
std::string get_error_state(E const &e) {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_state(*xi) : std::string(); });
}
/// \endcond

///////////////////////////////////////////////////////////////////////////
// \cond NOINTERNAL
// For testing purposes we sometime expect to see exceptions, allow those
// to go through without attaching a debugger.
//
// This should be used carefully as it disables the possible attaching of
// a debugger for all exceptions, not only the expected ones.
EINSUMS_EXPORT bool expect_exception(bool flag = true);
/// \endcond

} // namespace einsums::detail

#include <einsums/modules/errors.hpp>
