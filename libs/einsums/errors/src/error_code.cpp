// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception.hpp>

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

struct lightweight_einsums_category : einsums_category {};

struct einsums_category_rethrow : std::error_category {
    [[nodiscard]] auto name() const noexcept -> const char * override { return ""; }

    [[nodiscard]] auto message(int) const noexcept -> std::string override { return ""; }
};

struct lightweight_einsums_category_rethrow : einsums_category_rethrow {};

} // namespace detail

auto get_einsums_category() -> std::error_category const & {
    static detail::einsums_category einsums_category;
    return einsums_category;
}

auto get_einsums_rethrow_category() -> std::error_category const & {
    static detail::einsums_category_rethrow einsums_category_rethrow;
    return einsums_category_rethrow;
}

namespace detail {

auto get_lightweight_einsums_category() -> std::error_category const & {
    static detail::lightweight_einsums_category lightweight_einsums_category;
    return lightweight_einsums_category;
}

auto get_einsums_category(throw_mode mode) -> std::error_category const & {
    switch (mode) {
    case throw_mode::rethrow:
        return get_einsums_rethrow_category();

    case throw_mode::lightweight:
    case throw_mode::lightweight_rethrow:
        return get_lightweight_einsums_category();

    case throw_mode::plain:
    default:
        break;
    }
    return einsums::get_einsums_category();
}

auto throw_mode_is_lightweight(throw_mode mode) {
    return static_cast<int>(mode) & static_cast<int>(throw_mode::lightweight);
}

} // namespace detail

error_code::error_code(error e, throw_mode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, "", mode);
    }
}

error_code::error_code(error e, std::source_location const &location, throw_mode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, "", mode, location);
    }
}

error_code::error_code(error e, char const *msg, throw_mode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, msg, mode);
    }
}

error_code::error_code(error e, char const *msg, std::source_location const &location, throw_mode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, msg, mode, location);
    }
}

error_code::error_code(error e, std::string const &msg, throw_mode mode) : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, msg, mode);
    }
}

error_code::error_code(error e, std::string const &msg, std::source_location const &location, throw_mode mode)
    : std::error_code(detail::make_system_error_code(e, mode)) {
    if (e != einsums::error::success && e != einsums::error::no_success && !(detail::throw_mode_is_lightweight(mode))) {
        _exception = detail::get_exception(e, msg, mode, location);
    }
}

error_code::error_code(int err, einsums::exception const &e) {
    assign(err, get_einsums_category());
    _exception = std::make_exception_ptr(e);
}

error_code::error_code(std::exception_ptr const &e)
    : std::error_code(detail::make_system_error_code(get_error(e), throw_mode::rethrow)), _exception(e) {
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
    : std::error_code(static_cast<einsums::error>(rhs.value()) == einsums::error::success
                          ? make_success_code((category() == detail::get_lightweight_einsums_category()) ? einsums::throw_mode::lightweight
                                                                                                         : einsums::throw_mode::plain)
                          : rhs),
      _exception(rhs._exception) {
}

auto error_code::operator=(error_code const &rhs) -> error_code & {
    if (this != &rhs) {
        if (static_cast<einsums::error>(rhs.value()) == einsums::error::success) {
            // if the rhs is a success code, we maintain our throw mode
            this->std::error_code::operator=(make_success_code((category() == detail::get_lightweight_einsums_category())
                                                                   ? einsums::throw_mode::lightweight
                                                                   : einsums::throw_mode::plain));
        } else {
            this->std::error_code::operator=(rhs);
        }
        _exception = rhs._exception;
    }
    return *this;
}

} // namespace einsums