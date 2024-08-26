// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception.hpp>
#include <einsums/errors/throw_exception.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>

namespace einsums {

namespace detail {

struct einsums_category : std::error_category {
    [[nodiscard]] auto name() const noexcept -> const char * override { return "einsums"; }

    [[nodiscard]] auto message(int value) const -> std::string override {
        if (value >= static_cast<int>(einsums::error::success) && value < static_cast<int>(einsums::error::last_error)) {
            return std::string("einsums(") + error_names[value] + ")";
        }
        if (error_code_has_system_error(value)) {
            return {"einsums(system_error)"};
        }
        return "einsums(unknown_error)";
    }
};

} // namespace detail

auto get_einsums_category() -> std::error_category const & {
    static detail::einsums_category einsums_category;
    return einsums_category;
}

error_code::error_code(error e) : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, "");
    }
}

error_code::error_code(error e, std::source_location const &location) : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, "", location);
    }
}

error_code::error_code(error e, char const *msg) : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, msg);
    }
}

error_code::error_code(error e, char const *msg, std::source_location const &location)
    : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, msg, location);
    }
}

error_code::error_code(error e, std::string const &msg) : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, msg);
    }
}

error_code::error_code(error e, std::string const &msg, std::source_location const &location)
    : std::error_code(detail::make_system_error_code(e)) {
    if (e != einsums::error::success && e != einsums::error::no_success) {
        _exception = detail::get_exception(e, msg, location);
    }
}

error_code::error_code(int err, einsums::exception const &e) {
    assign(err, get_einsums_category());
    _exception = std::make_exception_ptr(e);
}

error_code::error_code(std::exception_ptr const &e) : std::error_code(detail::make_system_error_code(get_error(e))), _exception(e) {
}

auto error_code::get_message() const -> std::string {
    if (_exception) {
        try {
            std::rethrow_exception(_exception);
        } catch (std::exception const &be) {
            return be.what();
        }
    }
    return get_error_what(*this);
}

error_code::error_code(error_code const &rhs)
    : std::error_code(static_cast<einsums::error>(rhs.value()) == einsums::error::success ? make_success_code() : rhs),
      _exception(rhs._exception) {
}

} // namespace einsums