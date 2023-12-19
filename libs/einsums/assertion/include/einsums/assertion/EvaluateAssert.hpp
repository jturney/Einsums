//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/Config.hpp>
#include <einsums/assertion/SourceLocation.hpp>
#include <string>

namespace einsums::detail {
/// \cond NOINTERNAL
EINSUMS_EXPORT void handle_assert(SourceLocation const &loc, const char *expr, std::string const &msg) noexcept;
/// \endcond
} // namespace einsums::detail