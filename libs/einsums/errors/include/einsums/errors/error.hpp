//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <fmt/format.h>

#include <string>
#include <system_error>

namespace einsums {

/// \brief Possible error conditions
///
/// This enumeration lists all possible error conditions that can be reported from any of the API functions.
enum class error {
    /// The operation was successful.
    success = 0,
    /// The operation failed, but not in an unexpected manner
    no_success = 1,
    /// The operation is not implemented.
    not_implemented = 2,
    /// The operation caused an out of memory condition
    out_of_memory = 3,
    /// The operation was executed in an invalid status
    invalid_status = 4,
    /// One of the supplied parameters is invalid
    bad_parameter = 5,
    ///
    lock_error = 6,
    ///
    startup_timed_out = 7,
    ///
    uninitialized_value = 8,
    ///
    bad_response_type = 9,
    ///
    deadlock = 10,
    ///
    assertion_failure = 11,
    /// Attempt to invoke a function that requires a pika thread from a non-pika thread
    null_thread_id = 12,
    ///
    invalid_data = 13,
    /// The yield operation was aborted
    yield_aborted = 14,
    ///
    dynamic_link_failure = 15,
    /// One of the options given on the command line is erroneous
    commandline_option_error = 16,
    /// An unhandled exception has been caught
    unhandled_exception = 17,
    /// The OS kernel reported an error
    kernel_error = 18,
    /// The task associated with this future object is not available anymore
    broken_task = 19,
    /// The task associated with this future object has been moved
    task_moved = 20,
    /// The task associated with this future object has already been started
    task_already_started = 21,
    /// The future object has already been retrieved
    future_already_retrieved = 22,
    /// The value for this future object has already been set
    promise_already_satisfied = 23,
    /// The future object does not support cancellation
    future_does_not_support_cancellation = 24,
    /// The future can't be canceled at this time
    future_can_not_be_cancelled = 25,
    /// The future object has no valid shared state
    no_state = 26,
    /// The promise has been deleted
    broken_promise = 27,
    ///
    thread_resource_error = 28,
    ///
    future_cancelled = 29,
    ///
    thread_cancelled = 30,
    ///
    thread_not_interruptable = 31,
    /// An unknown error occurred
    unknown_error = 32,
    /// equivalent of std::bad_function_call
    bad_function_call = 33,
    /// parallel::task_canceled_exception
    task_canceled_exception = 34,
    /// task_region is not active
    task_block_not_active = 35,
    /// Equivalent to std::out_of_range
    out_of_range = 36,
    /// Equivalent to std::length_error
    length_error = 37,

    /// \cond NOINTERNAL
    last_error,

    system_error_flag = 0x4000L,
    error_upper_bound = 0x7fffL // force this enum type to be at least 16 bits.
                                /// \endcond
};

namespace detail {
/// \cond NOINTERNAL
char const *const error_names[] = {
    /*  0 */ "success",
    /*  1 */ "no_success",
    /*  2 */ "not_implemented",
    /*  3 */ "out_of_memory",
    /*  4 */ "invalid_status",
    /*  5 */ "bad_parameter",
    /*  6 */ "lock_error",
    /*  7 */ "startup_timed_out",
    /*  8 */ "uninitialized_value",
    /*  9 */ "bad_response_type",
    /* 10 */ "deadlock",
    /* 11 */ "assertion_failure",
    /* 12 */ "null_thread_id",
    /* 13 */ "invalid_data",
    /* 14 */ "yield_aborted",
    /* 15 */ "dynamic_link_failure",
    /* 16 */ "commandline_option_error",
    /* 17 */ "unhandled_exception",
    /* 18 */ "kernel_error",
    /* 19 */ "broken_task",
    /* 20 */ "task_moved",
    /* 21 */ "task_already_started",
    /* 22 */ "future_already_retrieved",
    /* 23 */ "promise_already_satisfied",
    /* 24 */ "future_does_not_support_cancellation",
    /* 25 */ "future_can_not_be_cancelled",
    /* 26 */ "no_state",
    /* 27 */ "broken_promise",
    /* 28 */ "thread_resource_error",
    /* 29 */ "future_cancelled",
    /* 30 */ "thread_cancelled",
    /* 31 */ "thread_not_interruptable",
    /* 32 */ "unknown_error",
    /* 33 */ "bad_function_call",
    /* 34 */ "task_canceled_exception",
    /* 35 */ "task_block_not_active",
    /* 36 */ "out_of_range",
    /* 37 */ "length_error",

    /*    */ ""};

inline auto error_code_has_system_error(int e) -> bool {
    return e & static_cast<int>(einsums::error::system_error_flag);
}
/// \endcond
} // namespace detail

} // namespace einsums

template <>
struct fmt::formatter<einsums::error> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(einsums::error e, FormatContext &ctx) {
        int e_int = static_cast<int>(e);
        if (e_int >= static_cast<int>(einsums::error::success) &&
            e_int < static_cast<int>(einsums::error::last_error)) {
            return fmt::formatter<std::string>::format(einsums::detail::error_names[e_int], ctx);
        } else {
            return fmt::formatter<std::string>::format(fmt::format("invalid einsums::error ({})", e_int), ctx);
        }
    }
};

/// \cond NOINTERNAL
namespace std {

// make sure our errors get recognized by the standard library
template <>
struct is_error_code_enum<einsums::error> : true_type {};

} // namespace std
  /// \endcond