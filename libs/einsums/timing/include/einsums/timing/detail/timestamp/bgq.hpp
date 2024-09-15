//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(__bgq__)

// Hardware cycle-accurate timer on BGQ.
// see https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#High-Resolution_Timers

#    include <einsums/config.hpp>

#    include <cstdint>
#    include <hwi/include/bqc/A2_inlines.h>

#    if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_CODE)
#        include <einsums/timing/detail/timestamp/cuda.hpp>
#    endif

namespace einsums::chrono::detail {
EINSUMS_HOST_DEVICE inline std::uint64_t timestamp() {
#    if defined(EINSUMS_HAVE_CUDA) && defined(EINSUMS_COMPUTE_DEVICE_CODE)
    return timestamp_cuda();
#    else
    return GetTimeBase();
#    endif
}
} // namespace einsums::chrono::detail

#endif
