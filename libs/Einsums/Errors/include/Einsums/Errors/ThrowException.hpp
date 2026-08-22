//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <exception>       // IWYU pragma: keep
#include <source_location> // IWYU pragma: keep
#include <string>          // IWYU pragma: keep
#include <system_error>    // IWYU pragma: keep

/**
 * @def EINSUMS_THROW_STD_EXCEPTION
 *
 * Throws an exception without a message string. Only reports its location.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_THROW_STD_EXCEPTION(except)                                                                                                \
    throw except(einsums::detail::make_error_message(einsums::type_name<except>(), "", std::source_location::current())) /**/

/**
 * @def EINSUMS_THROW_EXCEPTION
 *
 * Throws an exception with a formatted error message. It will also report its location.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_THROW_EXCEPTION(except, ...)                                                                                               \
    throw except(                                                                                                                          \
        einsums::detail::make_error_message(einsums::type_name<except>(), fmt::format(__VA_ARGS__), std::source_location::current())) /**/

/**
 * @def EINSUMS_THROW_CODED_EXCEPTION
 *
 * Throws an exception with a code to distinguish between exceptions of the same type within the same function. It can also
 * have a custom error message, and it will report the location in the code.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_THROW_CODED_EXCEPTION(except, code, ...)                                                                                   \
    throw einsums::CodedError<except, code>(                                                                                               \
        einsums::detail::make_error_message(einsums::type_name<except>(), fmt::format(__VA_ARGS__), std::source_location::current())) /**/

/**
 * @def EINSUMS_THROW_NOT_IMPLEMENTED
 *
 * This will throw a not_implemented exception.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_THROW_NOT_IMPLEMENTED                                                                                                      \
    throw not_implemented(                                                                                                                 \
        einsums::detail::make_error_message(einsums::type_name<not_implemented>(), "", std::source_location::current())) /**/

/**
 * @def EINSUMS_THROW_NESTED
 *
 * This will throw a nested exception.
 *
 * @versionadded{2.0.0}
 */
#define EINSUMS_THROW_NESTED(except, ...)                                                                                                  \
    std::throw_with_nested(except(einsums::detail::make_error_message(einsums::type_name<except>(), fmt::format(__VA_ARGS__),              \
                                                                      std::source_location::current()))) /**/
