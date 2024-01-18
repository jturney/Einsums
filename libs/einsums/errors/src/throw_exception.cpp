//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/exception.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <system_error>

namespace einsums::detail {
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
[[noreturn]] void throw_exception(error errcode, std::string const &msg, std::string const &func,
                                  std::string const &file, long line)
// NOLINTEND(bugprone-easily-swappable-parameters)
{
    std::filesystem::path p(file);
    einsums::detail::throw_exception(einsums::exception(errcode, msg, einsums::throwmode::plain), func, p.string(),
                                     line);
}

[[noreturn]] void rethrow_exception(exception const &e, std::string const &func) {
    einsums::detail::throw_exception(einsums::exception(e.get_error(), e.what(), einsums::throwmode::rethrow), func,
                                     einsums::get_error_file_name(e), einsums::get_error_line_number(e));
}

auto get_exception(error errcode, std::string const &msg, throwmode mode, std::string const & /* func */,
                   std::string const &file, long line, std::string const &auxinfo) -> std::exception_ptr {
    std::filesystem::path p(file);
    return einsums::detail::get_exception(einsums::exception(errcode, msg, mode), p.string(), file, line, auxinfo);
}

auto get_exception(std::error_code const &ec, std::string const & /* msg */, throwmode /* mode */,
                   std::string const &func, std::string const &file, long line, std::string const &auxinfo)
    -> std::exception_ptr {
    return einsums::detail::get_exception(einsums::exception(ec), func, file, line, auxinfo);
}

void throws_if(einsums::error_code &ec, error errcode, std::string const &msg, std::string const &func,
               std::string const &file, long line) {
    if (&ec == &einsums::throws) {
        einsums::detail::throw_exception(errcode, msg, func, file, line);
    } else {
        ec = make_error_code(static_cast<einsums::error>(errcode), msg, func.c_str(), file.c_str(), line,
                             (ec.category() == get_lightweight_einsums_category()) ? einsums::throwmode::lightweight
                                                                                   : einsums::throwmode::plain);
    }
}

void rethrows_if(einsums::error_code &ec, exception const &e, std::string const &func) {
    if (&ec == &einsums::throws) {
        einsums::detail::rethrow_exception(e, func);
    } else {
        ec = make_error_code(e.get_error(), e.what(), func.c_str(), einsums::get_error_file_name(e).c_str(),
                             einsums::get_error_line_number(e),
                             (ec.category() == get_lightweight_einsums_category())
                                 ? einsums::throwmode::lightweight_rethrow
                                 : einsums::throwmode::rethrow);
    }
}

[[noreturn]] void throw_thread_interrupted_exception() {
    throw einsums::thread_interrupted();
}
} // namespace einsums::detail
