//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// Public entry point for the SIMD-dispatched inner-loop kernel of
// stream_contract (Stream.hpp). The stream walks the large tensor S once in
// storage order and, for each term, runs an innermost loop over the
// unit-stride axis:
//
//     C[co + i*dc] += alpha * S[si + i*ds] * W[wo + i*dw]     (i = 0..n)
//
// The (ds, dc, dw) stride triple on that axis is only known at execute time,
// which hides the unit-stride/broadcast structure from the compiler's
// autovectorizer (it emits scalar gather/scatter over the runtime strides).
// This kernel branches on the triple at runtime and hands each vectorizable
// case to Stripes' Vec ops:
//
//     (1,1,0)  scaled AXPY      C[i] += (alpha*W) * S[i]     (GEMV-shaped terms: Fock J/K)
//     (1,1,1)  Hadamard FMA     C[i] += alpha * S[i] * W[i]
//     (1,0,1)  dot reduction    C[co] += alpha * sum_i S[i]*W[i]
//
// Real and complex types both vectorize these three triples (complex via
// the interleaved CVec ops). Any other pattern - non-unit strides, or an
// exotic element type - falls back to the general scalar strided loop.
//
// Compiled per rung and dispatched at run time like the micro-kernel, but with no `sme` rung: a
// bandwidth-bound streaming FMA gains nothing from the matrix unit.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

/// Function-pointer type for one term's innermost stream loop.
template <typename T>
using StreamInnerFn = void (*)(T *cb, T const *sp, T const *w, T alpha, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds,
                               int64_t dc, int64_t dw);

/// Function-pointer type for one term's two innermost stream loops: m rows of the
/// StreamInnerFn loop, row r offset by (r*dc2, r*ds2, r*dw2).
template <typename T>
using StreamTileFn = void (*)(T *cb, T const *sp, T const *w, T alpha, int64_t m, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds,
                              int64_t dc, int64_t dw, int64_t ds2, int64_t dc2, int64_t dw2);

/// Resolve the best inner-loop kernel for @p T at stripes::selected_arch().
/// Cached on first call; safe to invoke on the hot path but callers should
/// still hoist it out of inner loops.
template <typename T>
EINSUMS_EXPORT StreamInnerFn<T> stream_inner_entry();

/// Resolve the best two-loop tile kernel for @p T, as stream_inner_entry does.
template <typename T>
EINSUMS_EXPORT StreamTileFn<T> stream_tile_entry();

EINSUMS_NAMESPACE_END(packed_gemm)
