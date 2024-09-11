//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <string>

namespace einsums::detail {
/// Attaches a debugger if \c category is equal to the configuration entry
/// einsums.attach-debugger.
void EINSUMS_EXPORT may_attach_debugger(std::string const &category);
} // namespace einsums::detail