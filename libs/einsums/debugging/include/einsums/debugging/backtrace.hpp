//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STACKTRACES)
#    include <einsums/debugging/detail/backtrace.hpp>
#else

#    include <cstddef>
#    include <string>

namespace einsums::debug::detail {
class backtrace {};

inline std::string trace(std::size_t = 0) {
    return "";
}
} // namespace einsums::debug::detail

#endif
