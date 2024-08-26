// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception_fwd.hpp>
#include <einsums/errors/exception_info.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <system_error>
#include <utility>

namespace einsums {

struct EINSUMS_EXPORT exception : public std::system_error {
    explicit exception(error e = einsums::error::success);

    explicit exception(std::system_error const &e);

    explicit exception(std::error_code const &e);

    exception(error e, char const *msg, throw_mode mode = throw_mode::plain);

    exception(error e, std::string const &msg, throw_mode mode = throw_mode::plain);

    ~exception() noexcept override;

    [[nodiscard]] auto get_error() const noexcept -> error;

    [[nodiscard]] auto get_error_code(throw_mode mode = throw_mode::plain) const noexcept -> error_code;
};

namespace detail {

using custom_exception_info_handler_type = std::function<einsums::exception_info(std::source_location const &, std::string const &)>;

EINSUMS_EXPORT void set_custom_exception_info_handler(custom_exception_info_handler_type f);

using pre_exception_handler_type = std::function<void()>;

EINSUMS_EXPORT void set_pre_exception_handler(pre_exception_handler_type f);

} // namespace detail

struct EINSUMS_EXPORT thread_interrupted : std::exception {};

namespace detail {

EINSUMS_DEFINE_ERROR_INFO(throw_function, std::string);
EINSUMS_DEFINE_ERROR_INFO(throw_file, std::string);
EINSUMS_DEFINE_ERROR_INFO(throw_line, long);

struct EINSUMS_EXPORT std_exception : std::exception {
    explicit std_exception(std::string w) : _what(std::move(w)) {}

    ~std_exception() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> const char * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_alloc : std::bad_alloc {
    explicit bad_alloc(std::string w) : _what(std::move(w)) {}

    ~bad_alloc() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> const char * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_exception : std::bad_exception {

  public:
    explicit bad_exception(std::string w) : _what(std::move(w)) {}

    ~bad_exception() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> const char * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_cast : std::bad_cast {

  public:
    explicit bad_cast(std::string w) : _what(std::move(w)) {}

    ~bad_cast() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> const char * override { return _what.c_str(); }

  private:
    std::string _what;
};

struct EINSUMS_EXPORT bad_typeid : std::bad_typeid {
    explicit bad_typeid(std::string w) : _what(std::move(w)) {}

    ~bad_typeid() noexcept override = default;

    [[nodiscard]] auto what() const noexcept -> const char * override { return _what.c_str(); }

  private:
    std::string _what;
};

template <typename Exception>
EINSUMS_EXPORT auto get_exception(einsums::exception const &e, std::string const &func, std::string const &file, long line,
                                  std::string const &auxinfo) -> std::exception_ptr;

template <typename Exception>
EINSUMS_EXPORT auto construct_lightweight_exception(Exception const &e) -> std::exception_ptr;

} // namespace detail

EINSUMS_EXPORT auto get_error_what(exception_info const &xi) -> std::string;

template <typename E>
auto get_error_what(E const &e) -> std::string {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_what(*xi) : std::string("<unknown>"); });
}

inline auto get_error_what(einsums::error_code const &e) -> std::string {
    if (e.category() == detail::get_lightweight_einsums_category()) {
        return e.message();
    }

    return get_error_what<einsums::error_code>(e);
}

inline auto get_error_what(std::exception const &e) -> std::string {
    return e.what();
}

EINSUMS_EXPORT auto get_error(einsums::exception const &e) -> error;

EINSUMS_EXPORT auto get_error(einsums::error_code const &e) -> error;

EINSUMS_EXPORT auto get_error(std::exception_ptr const &e) -> error;

EINSUMS_EXPORT auto get_error_function_name(einsums::exception_info const &xi) -> std::string;

template <typename E>
auto get_error_function_name(E const &e) -> std::string {
    return invoke_with_exception_info(
        e, [](exception_info const *xi) { return xi ? get_error_function_name(*xi) : std::string("<unknown>"); });
}

EINSUMS_EXPORT auto get_error_file_name(einsums::exception_info const &xi) -> std::string;

template <typename E>
auto get_error_file_name(E const &e) -> std::string {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_file_name(*xi) : std::string("<unknown>"); });
}

EINSUMS_EXPORT auto get_error_line_number(einsums::exception_info const &xi) -> long;

template <typename E>
auto get_error_line_number(E const &e) -> long {
    return invoke_with_exception_info(e, [](exception_info const *xi) { return xi ? get_error_line_number(*xi) : -1; });
}

} // namespace einsums

#include <einsums/errors/throw_exception.hpp>
