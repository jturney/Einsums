// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error.hpp>
#include <einsums/errors/error_code.hpp>
#include <einsums/errors/exception_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <source_location>
#include <string>
#include <system_error>
#include <utility>

namespace einsums {

struct EINSUMS_EXPORT exception : std::system_error {
    explicit exception(error e = einsums::error::success);

    explicit exception(std::system_error const &e);

    explicit exception(std::error_code const &e);

    exception(error e, char const *msg);

    exception(error e, std::string const &msg);

    ~exception() noexcept override;

    [[nodiscard]] auto get_error() const noexcept -> error;

    [[nodiscard]] auto get_error_code() const noexcept -> error_code;
};

namespace detail {

template <typename Exception>
EINSUMS_EXPORT auto get_exception(einsums::exception const &e, const std::source_location &location, std::string const &auxinfo)
    -> std::exception_ptr;

}

} // namespace einsums