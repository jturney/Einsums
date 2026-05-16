//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/StringLiteral.hpp>

#include <source_location>
#include <stdexcept>
#include <string>

namespace einsums {

namespace detail {

/**
 * Construct a message that contains the type of error being produced, the location that error is being emitted,
 * and the actual message for the error.
 *
 * @param type_name The name of the type producing the error.
 * @param str The message for the error.
 * @param location The source location that the error is being emitted.
 *
 * @return A message with this extra debugging info.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, char const *str, std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, char const *str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, std::string const &str,
                                              std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, std::string const &str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

} // namespace detail

/**
 * @struct CodedError
 *
 * This error type is used when a function can emit several different instances of the
 * same error. This allows the user to either catch the class the code is based on,
 * or the CodedError with the code specified. This means that the user can
 * handle all errors with a similar cause together, or gain more fine-grained control
 * if needed.
 *
 * @versionadded{1.0.0}
 */
template <class ErrorClass, int ErrorCode>
struct CodedError : ErrorClass {
    using ErrorClass::ErrorClass;

    /**
     * Get the error code for this exception
     *
     * @versionadded{1.0.0}
     */
    constexpr int get_code() const { return ErrorCode; }
};

/**
 * @struct rank_error
 *
 * Indicates that the rank of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.1.0}
 */
struct EINSUMS_EXPORT rank_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct dimension_error
 *
 * Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT dimension_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct tensor_compat_error
 *
 * Indicates that two or more tensors are incompatible to be operated with each other for a reason other
 * than their dimensions.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT tensor_compat_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct num_argument_error
 *
 * Indicates that a function did not receive the correct amount of arguments.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT num_argument_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct not_enough_args
 *
 * Indicates that a function did not receive enough arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT not_enough_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct too_many_args
 *
 * Indicates that a function received too many arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT too_many_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct access_denied
 *
 * Indicates that an operation was stopped due to access restrictions, for instance writing to read-only data.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT access_denied : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct todo_error
 *
 * Indicates that a certain code path is not yet finished.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT todo_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct not_implemented
 *
 * Indicates that a certain code path is not implemented.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT not_implemented : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct bad_logic
 *
 * Indicates that an error occurred for some unspecified reason. It means
 * the same as std::logic_error. However, since so many exceptions are derived from
 * std::logic_error, this acts as a way to not break things.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT bad_logic : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct uninitialized_error
 *
 * Indicates that the code is handling data that is uninitialized.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT uninitialized_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct system_error
 *
 * Indicates that an error happened when making a system call.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT system_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct enum_error
 *
 * Indicates that an invalid enum value was passed to a function.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT enum_error : std::domain_error {
    using std::domain_error::domain_error;
};

/**
 * @struct complex_conversion_error
 *
 * Thrown when trying to convert a complex number to a real number. Instead, the input
 * data should be transformed into a real value in a way that makes sense for the operation
 * being performed. This is often either the magnitude or the real part.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT complex_conversion_error : std::logic_error {
    using std::logic_error::logic_error;
};

} // namespace einsums
