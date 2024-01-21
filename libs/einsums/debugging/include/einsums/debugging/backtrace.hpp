//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_HAVE_STACKTRACES)
#    include <einsums/debugging/backtrace/backtrace.hpp>
#else

#    include <cstddef>
#    include <string>

namespace einsums::debug::detail {

struct backtrace {};

inline auto trace(std::size_t frames_no = PIKA_HAVE_THREAD_BACKTRACE_DEPTH) -> std::string {
    return "";
}

} // namespace einsums::debug::detail

#endif
