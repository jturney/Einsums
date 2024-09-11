//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/topology/cpu_mask.hpp>

#include <iomanip>
#include <sstream>
#include <string>

#if !EINSUMS_HAVE_MORE_THAN_64_THREADS && EINSUMS_MAX_CPU_COUNT <= 64
#    define EINSUMS_CPU_MASK_PREFIX "0x"
#else
#    define EINSUMS_CPU_MASK_PREFIX "0b"
#endif

namespace einsums::threads::detail {

std::string to_string(mask_cref_type val) {
    std::ostringstream ostr;
    ostr << std::hex << EINSUMS_CPU_MASK_PREFIX << val;
    return ostr.str();
}

} // namespace einsums::threads::detail