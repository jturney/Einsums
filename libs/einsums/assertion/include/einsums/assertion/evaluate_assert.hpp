//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assertion/source_location.hpp>

#include <string>

namespace einsums::detail {
/// \cond NOINTERNAL
EINSUMS_EXPORT void handle_assert(source_location const &loc, char const *expr, std::string const &msg) noexcept;
/// \endcond
} // namespace einsums::detail