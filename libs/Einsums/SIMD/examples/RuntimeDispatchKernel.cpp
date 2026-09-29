//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file RuntimeDispatchKernel.cpp
/// @brief The kernel half of RuntimeDispatch.cpp: compiled once per instruction-set rung.
///
/// einsums_add_simd_dispatch_sources (see this directory's CMakeLists.txt) builds this file several
/// times, once per rung, each copy with that rung's -march flags and with EINSUMS_SIMD_ARCH_NS
/// defined to a namespace of its own: arch_baseline, arch_v2, arch_v3, arch_v4. Inside each copy
/// Vec<float> is as wide as that rung allows, so the same source becomes an SSE2, an AVX2 and an
/// AVX-512 kernel. Where there is no ladder (a non-x86 target, or a build pinned to one CPU) it is
/// built once, as arch_native.
///
/// Nothing here may be called directly: only the dispatcher, which checks what the CPU supports,
/// knows which copy is safe to run.

#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <cstddef>

namespace simd_example {
namespace EINSUMS_SIMD_ARCH_NS {

using namespace einsums::simd;

/// The sum of x[i]^2 at this rung's vector width.
float sum_of_squares(float const *x, std::size_t n) {
    constexpr std::size_t L   = lanes<float>;
    Vec<float>            acc = broadcast(0.0f);
    std::size_t           i   = 0;
    for (; i + L <= n; i += L) {
        Vec<float> const v = loadu(x + i);
        acc                = fmadd(v, v, acc);
    }
    if (i < n) {
        Vec<float> const v = loadu_partial(x + i, n - i);
        acc                = fmadd(v, v, acc);
    }
    return reduce_add(acc);
}

/// How many floats a Vec holds in this copy of the kernel.
int float_lanes() {
    return lanes<float>;
}

} // namespace EINSUMS_SIMD_ARCH_NS
} // namespace simd_example
