//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstdint>
#include <time.h>

#if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_CODE)
#    include <einsums/timing/detail/timestamp/cuda.hpp>
#endif

namespace einsums::chrono::detail {
EINSUMS_HOST_DEVICE inline std::uint64_t timestamp() {
#if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_DEVICE_CODE)
    return timestamp_cuda();
#else
    struct timespec res;
    clock_gettime(CLOCK_MONOTONIC, &res);
    return 1000 * res.tv_sec + res.tv_nsec / 1000000;
#endif
}
} // namespace einsums::chrono::detail
