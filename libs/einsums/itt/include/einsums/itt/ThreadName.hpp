//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <string>

namespace einsums::detail {

/// Helper utility to set and store a name for the current operating system thread. Returns a reference to the name for
/// the current thread.
EINSUMS_EXPORT auto thread_name() -> std::string &;

} // namespace einsums::detail