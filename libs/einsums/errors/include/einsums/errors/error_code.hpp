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
#include <stdexcept>
#include <string>
#include <system_error>

namespace einsums {

namespace detail {

EINSUMS_EXPORT auto access_exception(error_code const &) -> std::exception_ptr;

struct command_line_error : std::logic_error {
    explicit command_line_error(char const *msg) : std::logic_error(msg) {}

    explicit command_line_error(std::string const &msg) : std::logic_error(msg) {}
};

} // namespace detail

EINSUMS_EXPORT auto get_einsums_category() -> std::error_category const &;

EINSUMS_EXPORT auto get_einsums_rethrow_category() -> std::error_category const &;

namespace detail {

EINSUMS_EXPORT auto get_lightweight_einsums_category() -> std::error_category const &;

EINSUMS_EXPORT auto get_einsums_category(throw_mode mode) -> std::error_category const &;

inline auto make_system_error_code(error e, throw_mode mode = throw_mode::plain) -> std::error_code {
    return {static_cast<int>(e), get_einsums_category(mode)};
}

inline auto make_error_condition(error e, throw_mode mode) -> std::error_condition {
    return {static_cast<int>(e), get_einsums_category(mode)};
}

} // namespace detail

struct error_code : public std::error_code {
    explicit error_code(throw_mode mode = throw_mode::plain)
        : std::error_code(detail::make_system_error_code(einsums::error::success, mode)) {}

    EINSUMS_EXPORT explicit error_code(error e, throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT error_code(error e, std::source_location const &location = std::source_location::current(),
                              throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT error_code(error e, char const *msg, throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT error_code(error e, char const *msg, std::source_location const &location = std::source_location::current(),
                              throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT error_code(error e, std::string const &msg, throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT error_code(error e, std::string const &msg, std::source_location const &location = std::source_location::current(),
                              throw_mode mode = throw_mode::plain);

    EINSUMS_EXPORT [[nodiscard]] auto get_message() const -> std::string;

    void clear() {
        std::error_code::assign(static_cast<int>(einsums::error::success), get_einsums_category());
        _exception = std::exception_ptr();
    }

    EINSUMS_EXPORT error_code(error_code const &rhs);

    EINSUMS_EXPORT auto operator=(error_code const &rhs) -> error_code &;

  private:
    friend auto detail::access_exception(error_code const &) -> std::exception_ptr;
    friend struct exception;
    friend auto make_error_code(std::exception_ptr const &) -> error_code;

    EINSUMS_EXPORT error_code(int err, einsums::exception const &e);
    EINSUMS_EXPORT explicit error_code(std::exception_ptr const &e);

    std::exception_ptr _exception;
};

inline auto make_error_code(error e, throw_mode mode = throw_mode::plain) -> error_code {
    return error_code(e, mode);
}

inline auto make_error_code(error e, std::source_location const &location, throw_mode mode = throw_mode::plain) -> error_code {
    return {e, location, mode};
}

inline auto make_error_code(error e, char const *msg, throw_mode mode = throw_mode::plain) -> error_code {
    return {e, msg, mode};
}

inline auto make_error_code(error e, char const *msg, std::source_location const &location,
                            throw_mode mode = throw_mode::plain) -> error_code {
    return {e, msg, location, mode};
}

inline auto make_error_code(error e, std::string const &msg, throw_mode mode = throw_mode::plain) -> error_code {
    return {e, msg, mode};
}

inline auto make_error_code(error e, std::string const &msg, std::source_location const &location,
                            throw_mode mode = throw_mode::plain) -> error_code {
    return {e, msg, location, mode};
}

inline auto make_error_code(std::exception_ptr const &e) -> error_code {
    return error_code(e);
}

inline auto make_success_code(throw_mode mode = throw_mode::plain) {
    return error_code(mode);
}

} // namespace einsums