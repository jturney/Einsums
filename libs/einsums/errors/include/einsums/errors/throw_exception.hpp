// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/exception_fwd.hpp>

#include <exception>
#include <source_location>
#include <string>
#include <system_error>

namespace einsums {
namespace detail {

template <typename Exception>
[[noreturn]] EINSUMS_EXPORT void throw_exception(Exception const            &e,
                                                 std::source_location const &location = std::source_location::current());

[[noreturn]] EINSUMS_EXPORT void throw_exception(error errcode, std::string const &msg,
                                                 std::source_location const &location = std::source_location::current());

[[noreturn]] EINSUMS_EXPORT void rethrow_exception(exception const            &e,
                                                   std::source_location const &location = std::source_location::current());

template <typename Exception>
EINSUMS_EXPORT auto get_exception(Exception const &e, std::source_location const &location = std::source_location::current(),
                                  std::string const &auxinfo = "") -> std::exception_ptr;

EINSUMS_EXPORT auto get_exception(error errcode, std::string const &msg,
                                  std::source_location const &location = std::source_location::current(), std::string const &auxinfo = "")
    -> std::exception_ptr;

EINSUMS_EXPORT auto get_exception(std::error_code const &ec, std::string const &msg,
                                  std::source_location const &location = std::source_location::current(), std::string const &auxinfo = "")
    -> std::exception_ptr;

EINSUMS_EXPORT void throws_if(einsums::error_code &ec, error errcode, std::string const &msg, std::source_location const &location);

EINSUMS_EXPORT void rethrows_if(einsums::error_code &ec, exception const &e, std::source_location const &location);

} // namespace detail

[[noreturn]] inline void throw_exception(einsums::error e, std::string const &msg, std::source_location const &location) {
    detail::throw_exception(e, msg, location);
}

} // namespace einsums

#define EINSUMS_THROW_STD_EXCEPTION(except) einsums::detail::throw_exception(except, std::source_location::current()) /**/

#define EINSUMS_RETHROW_EXCEPTION(e) einsums::detail::rethrow_exception(e, std::source_location::current()) /**/

#define EINSUMS_RETHROWS_IF(ec, e) einsums::detail::rethrows_if(ec, e, std::source_location::current()) /**/

#define EINSUMS_GET_EXCEPTION(...) EINSUMS_GET_EXCEPTION_(__VA_ARGS__) /**/

#define EINSUMS_GET_EXCEPTION_(...)                                                                                                        \
    EINSUMS_PP_EXPAND(EINSUMS_PP_CAT(EINSUMS_GET_EXCEPTION_, EINSUMS_PP_NARGS(__VA_ARGS__))(__VA_ARGS__)) /**/

#define EINSUMS_GET_EXCEPTION_3(errcode, msg)                                                                                              \
    EINSUMS_GET_EXCEPTION_4(errcode, einsums::throws::plain, std::source_location::current(), msg) /**/

#define EINSUMS_GET_EXCEPTION_4(errcode, mode, msg) einsums::detail::get_exception(errcode, msg, mode, std::source_location::current()) /**/

#define EINSUMS_THROW_IN_CURRENT_FUNC(errcode, msg) EINSUMS_THROW_EXCEPTION(errcode, msg, std::source_location::current()) /**/

#define EINSUMS_THROW_EXCEPTION(errcode, ...)                                                                                              \
    einsums::detail::throw_exception(errcode, fmt::format(__VA_ARGS__), std::source_location::current()) /**/

#define EINSUMS_THROWS_IF(ec, errcode, ...)                                                                                                \
    einsums::detail::throws_if(ec, errcode, fmt::format(__VA_ARGS__), std::source_location::current()) /**/