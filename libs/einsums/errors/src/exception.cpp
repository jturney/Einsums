// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/errors/error.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception.hpp>
#include <einsums/errors/exception_info.hpp>

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#elif defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace einsums {

exception::exception(error e) : std::system_error(make_error_code(e, throw_mode::plain)) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

exception::exception(std::system_error const &e) : std::system_error(e) {
}

exception::exception(std::error_code const &e) : std::system_error(e) {
}

exception::exception(error e, char const *msg, throw_mode mode) : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

exception::exception(error e, std::string const &msg, throw_mode mode) : std::system_error(detail::make_system_error_code(e, mode), msg) {
    EINSUMS_ASSERT((e >= einsums::error::success && e < einsums::error::last_error) ||
                   (detail::error_code_has_system_error(static_cast<int>(e))));
}

exception::~exception() noexcept = default;

auto exception::get_error() const noexcept -> error {
    return static_cast<error>(code().value());
}

auto exception::get_error_code(throw_mode mode) const noexcept -> error_code {
    (void)mode;
    return {code().value(), *this};
}

namespace detail {

namespace {
custom_exception_info_handler_type custom_exception_info_handler;
pre_exception_handler_type         pre_exception_handler;
} // namespace

void set_custom_exception_info_handler(custom_exception_info_handler_type f) {
    custom_exception_info_handler = std::move(f);
}

void set_pre_exception_handler(pre_exception_handler_type f) {
    pre_exception_handler = std::move(f);
}

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e, std::source_location const &location) -> std::exception_ptr {
    try {
        throw_with_info(e, std::move(einsums::exception_info().set(einsums::detail::throw_function(location.function_name()),
                                                                   einsums::detail::throw_file(location.file_name()),
                                                                   einsums::detail::throw_line(location.line()))));
    } catch (...) {
        return std::current_exception();
    }

    return {};
}

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e) -> std::exception_ptr {
    try {
        einsums::throw_with_info(e);
    } catch (...) {
        return std::current_exception();
    }

    return {};
}

template EINSUMS_EXPORT auto construct_lightweight_exception(einsums::thread_interrupted const &) -> std::exception_ptr;

template <typename Exception>
EINSUMS_EXPORT auto construct_custom_exception(Exception const &e, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    if (!custom_exception_info_handler) {
        return construct_lightweight_exception(e, location);
    }

    // create a std::exception_ptr object encapsulating the Exception to
    // be thrown and annotate it with information provided by the hook
    try {
        throw_with_info(e, custom_exception_info_handler(location, auxinfo));
    } catch (...) {
        return std::current_exception();
    }

    return {};
}

template <typename Exception>
inline auto is_of_lightweight_einsums_category(Exception const &) -> bool {
    return false;
}

inline auto is_of_lightweight_einsums_category(einsums::exception const &e) -> bool {
    return e.get_error_code().category() == get_lightweight_einsums_category();
}

auto access_exception(error_code const &e) -> std::exception_ptr {
    return e._exception;
}

template <typename Exception>
EINSUMS_EXPORT auto get_exception(Exception const &e, std::source_location const &location, std::string const &auxinfo)
    -> std::exception_ptr {
    if (is_of_lightweight_einsums_category(e)) {
        return construct_lightweight_exception(e, location);
    }

    return construct_custom_exception(e, location, auxinfo);
}

template <typename Exception>
EINSUMS_EXPORT void throw_exception(Exception const &e, std::source_location const &location) {
    if (pre_exception_handler) {
        pre_exception_handler();
    }

    std::rethrow_exception(get_exception(e, location));
}

template EINSUMS_EXPORT auto get_exception(einsums::exception const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;

template EINSUMS_EXPORT auto get_exception(std::system_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;

template EINSUMS_EXPORT auto get_exception(std::exception const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(einsums::detail::std_exception const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_exception const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(einsums::detail::bad_exception const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_typeid const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(einsums::detail::bad_typeid const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_cast const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(einsums::detail::bad_cast const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::bad_alloc const &, std::source_location const &, std::string const &) -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(einsums::detail::bad_alloc const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::logic_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::runtime_error const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::out_of_range const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;
template EINSUMS_EXPORT auto get_exception(std::invalid_argument const &, std::source_location const &, std::string const &)
    -> std::exception_ptr;

template EINSUMS_EXPORT void throw_exception(einsums::exception const &, std::source_location const &);

template EINSUMS_EXPORT void throw_exception(std::system_error const &, std::source_location const &);

template EINSUMS_EXPORT void throw_exception(std::exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(einsums::detail::std_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_exception const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_typeid const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_typeid const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_cast const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_cast const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::bad_alloc const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(einsums::detail::bad_alloc const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::logic_error const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::runtime_error const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::out_of_range const &, std::source_location const &);
template EINSUMS_EXPORT void throw_exception(std::invalid_argument const &, std::source_location const &);

} // namespace detail

auto get_error_what(einsums::exception_info const &xi) -> std::string {
    auto const *se = dynamic_cast<std::exception const *>(&xi);
    return se ? se->what() : "<unknown>";
}

auto get_error(einsums::exception const &e) -> error {
    return static_cast<einsums::error>(e.get_error());
}

auto get_error(std::exception_ptr const &e) -> error {
    try {
        std::rethrow_exception(e);
    } catch (einsums::thread_interrupted const &) {
        return einsums::error::thread_cancelled;
    } catch (einsums::exception const &he) {
        return he.get_error();
    } catch (std::system_error const &e) {
        int code = e.code().value();
        if (code < static_cast<int>(einsums::error::success) || code >= static_cast<int>(einsums::error::last_error)) {
            code |= static_cast<int>(einsums::error::system_error_flag);
        }
        return static_cast<einsums::error>(code);
    } catch (...) {
        return einsums::error::unknown_error;
    }
}

auto get_error_function_name(einsums::exception_info const &xi) -> std::string {
    std::string const *function = xi.get<einsums::detail::throw_function>();
    if (function) {
        return *function;
    }

    return {};
}

auto get_error_file_name(einsums::exception_info const &xi) -> std::string {
    std::string const *file = xi.get<einsums::detail::throw_file>();
    if (file) {
        return *file;
    }

    return "<unknown>";
}

auto get_error_line_number(einsums::exception_info const &xi) -> long {
    long const *line = xi.get<einsums::detail::throw_line>();
    if (line) {
        return *line;
    }
    return -1;
}

} // namespace einsums