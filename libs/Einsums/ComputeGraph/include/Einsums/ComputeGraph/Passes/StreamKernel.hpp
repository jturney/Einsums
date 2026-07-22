//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Public entry point for the SIMD-dispatched inner-loop kernel used by
// StreamContractionFusion. The fused stream walks the large tensor S once in
// storage order and, for each member, runs an innermost loop over the
// unit-stride axis:
//
//     C[co + i*dc] += alpha * S[si + i*ds] * W[wo + i*dw]     (i = 0..n)
//
// The (ds, dc, dw) stride triple on that axis is only known at execute time,
// which hides the unit-stride/broadcast structure from the compiler's
// autovectorizer (it emits scalar gather/scatter over the runtime strides).
// This kernel branches on the triple at runtime and hands each vectorizable
// case to the SIMD module's Vec ops:
//
//     (1,1,0)  scaled AXPY      C[i] += (alpha*W) * S[i]     (GEMV-shaped members: Fock J/K)
//     (1,1,1)  Hadamard FMA     C[i] += alpha * S[i] * W[i]
//     (1,0,1)  dot reduction    C[co] += alpha * sum_i S[i]*W[i]
//
// Real and complex types both vectorize these three triples (complex via
// the interleaved CVec ops). Any other pattern - non-unit strides, or an
// exotic element type - falls back to the general scalar strided loop.
//
// The kernel body (StreamKernelBody.hpp) is compiled once per instruction-set
// rung by einsums_add_simd_dispatch_sources() (see src/Passes/StreamKernelImpl.cpp),
// and the rung is resolved at runtime in src/Passes/StreamKernelDispatch.cpp,
// mirroring the PackedGemm micro-kernel and HPTT transpose dispatch. Unlike
// PackedGemm this kernel requests NO `sme` rung: the operation is a
// bandwidth-bound streaming FMA with unit arithmetic intensity, which the
// SME matrix unit cannot accelerate (it targets reuse-heavy outer products).

#include <Einsums/Config.hpp>

#include <cstdint>

namespace einsums::compute_graph::passes {

/// Function-pointer type for one member's innermost stream loop.
template <typename T>
using StreamInnerFn = void (*)(T *cb, T const *sp, T const *w, T alpha, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds,
                               int64_t dc, int64_t dw);

/// Resolve the best inner-loop kernel for @p T at einsums::simd::selected_arch().
/// Cached on first call; safe to invoke on the hot path but callers should
/// still hoist it out of inner loops.
template <typename T>
EINSUMS_EXPORT StreamInnerFn<T> stream_inner_entry();

} // namespace einsums::compute_graph::passes
