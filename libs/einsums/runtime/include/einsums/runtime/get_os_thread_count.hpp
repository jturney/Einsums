//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>

namespace einsums {

/// Return the number of OS-threads running in the runtime instance
/// the current einsums-thread is associated with.
EINSUMS_EXPORT std::size_t get_os_thread_count();

} // namespace einsums