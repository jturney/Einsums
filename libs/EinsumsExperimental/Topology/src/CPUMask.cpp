//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Topology/CPUMask.hpp>

#include <iomanip>
#include <sstream>
#include <string>

#if !defined(EINSUMS_HAVE_MORE_THAN_64_THREADS) || (defined(EINSUMS_HAVE_MAX_CPU_COUNT) && EINSUMS_HAVE_MAX_CPU_COUNT <= 64)
#    define EINSUMS_CPU_MASK_PREFIX "0x"
#else
#    define EINSUMS_CPU_MASK_PREFIX "0b"
#endif

namespace einsums::topology::detail {

std::string to_string(MaskCRefType val) {
    std::ostringstream ostr;
    ostr << std::hex << EINSUMS_CPU_MASK_PREFIX << val;
    return ostr.str();
}

} // namespace einsums::topology::detail