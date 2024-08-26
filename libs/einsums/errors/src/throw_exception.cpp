// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/exception.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <system_error>

namespace einsums {
error_code throws; // "throw on error" special error_code;
}

namespace einsums::detail {

[[noreturn]] void throw_exception(error errcode, std::string const &msg, std::source_location const &location) {
    einsums::detail::throw_exception(einsums::exception(errcode, msg, einsums::throw_mode::plain), location);
}

[[noreturn]] void rethrow_exception(exception const &e, std::source_location const &location) {
    einsums::detail::throw_exception(einsums::exception(e.get_error(), e.what(), einsums::throw_mode::plain), location);
}

auto get_exception(error errcode, std::string const &msg, throw_mode mode, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    return einsums::detail::get_exception(einsums::exception(errcode, msg, mode), location, auxinfo);
}

auto get_exception(std::error_code const &ec, std::string const & /* msg */, throw_mode /* mode */, std::source_location const &location,
                   std::string const &auxinfo) -> std::exception_ptr {
    return einsums::detail::get_exception(einsums::exception(ec), location, auxinfo);
}

void throws_if(einsums::error_code &ec, error errcode, std::string const &msg, std::source_location const &location) {
    if (&ec == &einsums::throws) {
        einsums::detail::throw_exception(errcode, msg, location);
    } else {
        ec = make_error_code(static_cast<einsums::error>(errcode), msg, location,
                             (ec.category() == get_lightweight_einsums_category()) ? einsums::throw_mode::lightweight
                                                                                   : einsums::throw_mode::plain);
    }
}

void rethrows_if(einsums::error_code &ec, exception const &e, std::source_location const &location) {
    if (&ec == &einsums::throws) {
        einsums::detail::rethrow_exception(e, location);
    } else {
        ec = make_error_code(e.get_error(), e.what(), location,
                             (ec.category() == get_lightweight_einsums_category()) ? einsums::throw_mode::lightweight_rethrow
                                                                                   : einsums::throw_mode::rethrow);
    }
}

[[noreturn]] void throw_thread_interrupted_exception() {
    throw einsums::thread_interrupted();
}

} // namespace einsums::detail