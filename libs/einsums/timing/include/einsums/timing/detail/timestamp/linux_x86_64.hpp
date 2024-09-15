//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstdint>

#if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_CODE)
#    include <einsums/timing/detail/timestamp/cuda.hpp>
#endif

namespace einsums::chrono::detail {
EINSUMS_HOST_DEVICE inline std::uint64_t timestamp() {
#if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_DEVICE_CODE)
    return timestamp_cuda();
#else
    std::uint32_t lo = 0, hi = 0;
#    if defined(EINSUMS_HAVE_RDTSCP)
    __asm__ __volatile__("rdtscp ;\n" : "=a"(lo), "=d"(hi) : : "rcx");
#    elif defined(EINSUMS_HAVE_RDTSC)
    __asm__ __volatile__("cpuid ;\n"
                         "rdtsc ;\n"
                         : "=a"(lo), "=d"(hi)
                         :
                         : "rbx", "rcx");
#    endif
    return ((static_cast<std::uint64_t>(hi)) << 32) | lo;
#endif
}
} // namespace einsums::chrono::detail
