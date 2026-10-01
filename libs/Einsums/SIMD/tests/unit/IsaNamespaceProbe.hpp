//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/SIMD/Vec.hpp>

/// What one translation unit sees of the SIMD headers: the lane count it was compiled with, and the
/// native_lanes<float> object it links to, read through its address so the compiler cannot fold it.
struct IsaProbe {
    int         compiled_lanes;
    int const  *linked_object;
    int         linked_lanes;
    char const *tag;
};

#define EINSUMS_ISA_PROBE_STR2(x) #x
#define EINSUMS_ISA_PROBE_STR(x)  EINSUMS_ISA_PROBE_STR2(x)

// static, so each translation unit has its own copy compiled at its own flags.
static IsaProbe isa_probe_here() {
    int const *object = &einsums::simd::native_lanes<float>;
    return {einsums::simd::Vec<float>::lanes, object, *static_cast<int const volatile *>(object),
            EINSUMS_ISA_PROBE_STR(EINSUMS_SIMD_ISA_NS)};
}

IsaProbe isa_probe_wide();
