//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Per-rung stream-kernel translation unit. NOT compiled directly: the thin
// wrappers einsums_add_simd_dispatch_sources() generates (one per
// instruction-set rung) each define EINSUMS_SIMD_ARCH_NS and add that rung's
// -march flags, then include this file. The kernel template therefore
// compiles once per rung, in that rung's namespace, at that rung's ISA - so
// its Vec ops resolve to the rung's vector width. StreamKernelDispatch.cpp
// picks the best rung at runtime.

#include <Einsums/ComputeGraph/Passes/StreamKernel.hpp>

// StreamKernelBody.hpp is a private implementation header living next to this
// TU in src/Passes (not installed, not fed to the pybind codegen). The quoted
// include resolves relative to this file even though the SIMD-dispatch wrapper
// pulls this TU in by absolute path.
#define EINSUMS_STREAM_KERNEL_NS EINSUMS_SIMD_ARCH_NS
#include <complex>
#include <cstdint>

#include "StreamKernelBody.hpp"

namespace einsums::compute_graph::passes {
namespace EINSUMS_SIMD_ARCH_NS {

template void stream_inner<float>(float *, float const *, float const *, float, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                  int64_t);
template void stream_inner<double>(double *, double const *, double const *, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                   int64_t);
template void stream_inner<std::complex<float>>(std::complex<float> *, std::complex<float> const *, std::complex<float> const *,
                                                std::complex<float>, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
template void stream_inner<std::complex<double>>(std::complex<double> *, std::complex<double> const *, std::complex<double> const *,
                                                 std::complex<double>, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

} // namespace EINSUMS_SIMD_ARCH_NS
} // namespace einsums::compute_graph::passes
