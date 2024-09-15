//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_WINDOWS)

#    include <cstdint>
#    include <intrin.h>
#    include <windows.h>

#    if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_CODE)
#        include <einsums/timing/detail/timestamp/cuda.hpp>
#    endif

namespace einsums::chrono::detail {
EINSUMS_HOST_DEVICE inline std::uint64_t timestamp() {
#    if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_DEVICE_CODE)
    return timestamp_cuda();
#    else
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return static_cast<std::uint64_t>(now.QuadPart);
#    endif
}
} // namespace einsums::chrono::detail

#endif
