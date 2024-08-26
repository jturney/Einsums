//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <source_location>
#include <string>

namespace einsums::assertion::detail {

EINSUMS_EXPORT void handle_assert(std::source_location const &loc, const char *expr, std::string const &msg) noexcept;

}