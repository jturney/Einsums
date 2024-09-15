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
    std::uint64_t r = 0;

#    if defined(EINSUMS_HAVE_RDTSCP)
    __asm__ __volatile__("rdtscp ;\n" : "=A"(r) : : "%ecx");
#    elif defined(EINSUMS_HAVE_RDTSC)
    __asm__ __volatile__("movl %%ebx, %%edi ;\n"
                         "xorl %%eax, %%eax ;\n"
                         "cpuid ;\n"
                         "rdtsc ;\n"
                         "movl %%edi, %%ebx ;\n"
                         : "=A"(r)
                         :
                         : "%edi", "%ecx");
#    endif
    return r;
#endif
}
} // namespace einsums::chrono::detail
