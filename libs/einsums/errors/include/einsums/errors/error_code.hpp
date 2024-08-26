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

EINSUMS_EXPORT auto get_einsums_category() -> std::error_category const &;

namespace detail {

inline auto make_system_error_code(error e) -> std::error_code {
    return {static_cast<int>(e), get_einsums_category()};
}

inline auto make_error_condition(error e) -> std::error_condition {
    return {static_cast<int>(e), get_einsums_category()};
}

} // namespace detail

struct error_code : std::error_code {
    explicit error_code() : std::error_code(detail::make_system_error_code(einsums::error::success)) {}

    EINSUMS_EXPORT explicit error_code(error e);

    EINSUMS_EXPORT error_code(error e, std::source_location const &location);

    EINSUMS_EXPORT error_code(error e, char const *msg);

    EINSUMS_EXPORT error_code(error e, char const *msg, std::source_location const &location);

    EINSUMS_EXPORT error_code(error e, std::string const &msg);

    EINSUMS_EXPORT error_code(error e, std::string const &msg, std::source_location const &location);

    EINSUMS_EXPORT [[nodiscard]] auto get_message() const -> std::string;

    void clear() {
        std::error_code::assign(static_cast<int>(einsums::error::success), get_einsums_category());
        _exception = std::exception_ptr();
    }

    EINSUMS_EXPORT error_code(error_code const &rhs);

  private:
    friend auto detail::access_exception(error_code const &) -> std::exception_ptr;
    friend struct exception;
    friend auto make_error_code(std::exception_ptr const &) -> error_code;

    EINSUMS_EXPORT error_code(int err, einsums::exception const &e);
    EINSUMS_EXPORT explicit error_code(std::exception_ptr const &e);

    std::exception_ptr _exception;
};

inline auto make_error_code(error e) -> error_code {
    return error_code(e);
}

inline auto make_error_code(error e, std::source_location const &location) -> error_code {
    return {e, location};
}

inline auto make_error_code(error e, char const *msg) -> error_code {
    return {e, msg};
}

inline auto make_error_code(error e, char const *msg, std::source_location const &location) -> error_code {
    return {e, msg, location};
}

inline auto make_error_code(error e, std::string const &msg) -> error_code {
    return {e, msg};
}

inline auto make_error_code(error e, std::string const &msg, std::source_location const &location) -> error_code {
    return {e, msg, location};
}

inline auto make_error_code(std::exception_ptr const &e) -> error_code {
    return error_code(e);
}

inline auto make_success_code() {
    return error_code();
}

} // namespace einsums
