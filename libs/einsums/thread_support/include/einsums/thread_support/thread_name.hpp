//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#    include <windows.h>

namespace einsums::detail {
/// Helper utility to set and store a name for the current operating system
/// thread. Returns a reference to the name for the current thread.
PIKA_EXPORT std::string &thread_name();

EINSUMS_EXPORT void set_thread_name(char const * /*threadName*/, DWORD /*dwThreadID*/ = DWORD(-1));
} // namespace einsums::detail

#else

namespace einsums::detail {
/// Helper utility to set and store a name for the current operating system
/// thread. Returns a reference to the name for the current thread.
EINSUMS_EXPORT std::string &thread_name();

inline void set_thread_name(char const * /*thread_name*/) {
}
} // namespace einsums::detail

#endif
