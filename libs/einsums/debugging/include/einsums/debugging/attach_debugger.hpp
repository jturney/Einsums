//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

namespace einsums::debug::detail {

/// Tries to break an attached debugger, if not supported a loop is
/// invoked which gives enough time to attach a debugger manually.
EINSUMS_EXPORT void attach_debugger();

} // namespace einsums::debug::detail