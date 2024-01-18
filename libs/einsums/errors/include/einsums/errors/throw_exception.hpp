//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/preprocessor/cat.hpp>
#include <einsums/preprocessor/expand.hpp>
#include <einsums/preprocessor/nargs.hpp>

#include <einsums/assertion/current_function.hpp>
#include <einsums/errors/error.hpp>
#include <einsums/errors/exception_fwd.hpp>

#include <fmt/format.h>

#include <exception>
#include <string>
#include <system_error>

// clang-format off
#include <einsums/config/warnings_prefix.hpp>
// clang-format on

/// \cond NODETAIL
namespace einsums::detail {

template <typename Exception>
[[noreturn]] EINSUMS_EXPORT void throw_exception(Exception const &e, std::string const &func, std::string const &file,
                                                 long line);

[[noreturn]] EINSUMS_EXPORT void throw_exception(error errcode, std::string const &msg, std::string const &func,
                                                 std::string const &file, long line);

[[noreturn]] EINSUMS_EXPORT void rethrow_exception(exception const &e, std::string const &func);

template <typename Exception>
EINSUMS_EXPORT auto get_exception(Exception const &e, std::string const &func = "<unknown>",
                                  std::string const &file = "<unknown>", long line = -1,
                                  std::string const &auxinfo = "") -> std::exception_ptr;

EINSUMS_EXPORT auto get_exception(error errcode, std::string const &msg, throwmode mode,
                                  std::string const &func = "<unknown>", std::string const &file = "<unknown>",
                                  long line = -1, std::string const &auxinfo = "") -> std::exception_ptr;

EINSUMS_EXPORT auto get_exception(std::error_code const &ec, std::string const &msg, throwmode mode,
                                  std::string const &func = "<unknown>", std::string const &file = "<unknown>",
                                  long line = -1, std::string const &auxinfo = "") -> std::exception_ptr;

EINSUMS_EXPORT void throws_if(einsums::error_code &ec, error errcode, std::string const &msg, std::string const &func,
                              std::string const &file, long line);

EINSUMS_EXPORT void rethrows_if(einsums::error_code &ec, exception const &e, std::string const &func);

[[noreturn]] EINSUMS_EXPORT void throw_thread_interrupted_exception();

} // namespace einsums::detail
/// \endcond

namespace einsums {
/// \cond NOINTERNAL

/// \brief Throw an einsums::exception initialzied from the given arguments
[[noreturn]] inline void throw_exception(error e, std::string const &msg, std::string const &func,
                                         std::string const &file = "", long line = -1) {
    detail::throw_exception(e, msg, func, file, line);
}

/// \endcond
} // namespace einsums

/// \cond NOINTERNAL
////////////////////////////////////////////////////////////////////////////////
// helper macro allowing to prepend file name and line number to a generated exception
#define EINSUMS_THROW_STD_EXCEPTION(except, func)                                                                      \
    einsums::detail::throw_exception(except, func, __FILE__, __LINE__) /**/

#define EINSUMS_RETHROW_EXCEPTION(e, f) einsums::detail::rethrow_exception(e, f) /**/

#define EINSUMS_RETHROWS_IF(ec, e, f) einsums::detail::rethrows_if(ec, e, f) /**/

///////////////////////////////////////////////////////////////////////////////
#define EINSUMS_GET_EXCEPTION(...)                                                                                     \
    EINSUMS_GET_EXCEPTION_(__VA_ARGS__)                                                                                \
    /**/

#define EINSUMS_GET_EXCEPTION_(...)                                                                                    \
    EINSUMS_PP_EXPAND(EINSUMS_PP_CAT(EINSUMS_GET_EXCEPTION_, EINSUMS_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))              \
/**/
#define EINSUMS_GET_EXCEPTION_3(errcode, f, msg)                                                                       \
    EINSUMS_GET_EXCEPTION_4(errcode, einsums::throwmode::plain, f, msg)                                                \
/**/
#define EINSUMS_GET_EXCEPTION_4(errcode, mode, f, msg)                                                                 \
    einsums::detail::get_exception(errcode, msg, mode, f, __FILE__, __LINE__) /**/

///////////////////////////////////////////////////////////////////////////////
#define EINSUMS_THROW_IN_CURRENT_FUNC(errcode, msg)                                                                    \
    EINSUMS_THROW_EXCEPTION(errcode, EINSUMS_ASSERTION_CURRENT_FUNCTION, msg)                                          \
    /**/

#define EINSUMS_RETHROW_IN_CURRENT_FUNC(errcode, msg)                                                                  \
    EINSUMS_RETHROW_EXCEPTION(errcode, EINSUMS_ASSERTION_CURRENT_FUNCTION, msg)                                        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define EINSUMS_THROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                                                            \
    EINSUMS_THROWS_IF(ec, errcode, EINSUMS_ASSERTION_CURRENT_FUNCTION, msg)                                            \
    /**/

#define EINSUMS_RETHROWS_IN_CURRENT_FUNC_IF(ec, errcode, msg)                                                          \
    EINSUMS_RETHROWS_IF(ec, errcode, EINSUMS_ASSERTION_CURRENT_FUNCTION, msg)                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define EINSUMS_THROW_THREAD_INTERRUPTED_EXCEPTION() einsums::detail::throw_thread_interrupted_exception() /**/
/// \endcond

///////////////////////////////////////////////////////////////////////////////
/// \def EINSUMS_THROW_EXCEPTION(errcode, f, msg)
/// \brief Throw a einsums::exception initialized from the given parameters
///
/// The macro \a EINSUMS_THROW_EXCEPTION can be used to throw a einsums::exception.
/// The purpose of this macro is to prepend the source file name and line number
/// of the position where the exception is thrown to the error message.
/// Moreover, this associates additional diagnostic information with the
/// exception, such as file name and line number, thread id,
/// and stack backtrace from the point where the exception was thrown.
///
/// The parameter \p errcode holds the einsums::error code the new exception should
/// encapsulate. The parameter \p f is expected to hold the name of the
/// function exception is thrown from and the parameter \p msg holds the error
/// message the new exception should encapsulate.
///
/// \par Example:
///
/// \code
///      void raise_exception()
///      {
///          // Throw a einsums::exception initialized from the given parameters.
///          // Additionally associate with this exception some detailed
///          // diagnostic information about the throw-site.
///          EINSUMS_THROW_EXCEPTION(einsums::error:no_success, "raise_exception", "simulated error");
///      }
/// \endcode
///
#define EINSUMS_THROW_EXCEPTION(errcode, f, ...)                                                                       \
    einsums::detail::throw_exception(errcode, fmt::format(__VA_ARGS__), f, __FILE__, __LINE__) /**/

/// \def EINSUMS_THROWS_IF(ec, errcode, f, msg)
/// \brief Either throw a einsums::exception or initialize \a einsums::error_code from
///        the given parameters
///
/// The macro \a EINSUMS_THROWS_IF can be used to either throw a \a einsums::exception
/// or to initialize a \a einsums::error_code from the given parameters. If
/// &ec == &einsums::throws, the semantics of this macro are equivalent to
/// \a EINSUMS_THROW_EXCEPTION. If &ec != &einsums::throws, the \a einsums::error_code
/// instance \p ec is initialized instead.
///
/// The parameter \p errcode holds the einsums::error code from which the new
/// exception should be initialized. The parameter \p f is expected to hold the
/// name of the function exception is thrown from and the parameter \p msg
/// holds the error message the new exception should encapsulate.
///
#define EINSUMS_THROWS_IF(ec, errcode, f, ...)                                                                         \
    einsums::detail::throws_if(ec, errcode, fmt::format(__VA_ARGS__), f, __FILE__, __LINE__) /**/

#include <einsums/config/warnings_suffix.hpp>
