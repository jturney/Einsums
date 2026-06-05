//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstddef>

namespace einsums::simd {

// ===========================================================================
// Prefetch hints
// ===========================================================================

// Prefetch hints. T0/T1/T2/NTA are read-prefetch hints (different cache
// levels). The Write* variants tell the cache "this line is about to be
// written" — on aarch64 they lower to ``prfm pst*``, on x86 to ``PREFETCHW``
// (or its T1 variant on Xeon Phi-class chips). The CPU then primes the line
// in the right state to avoid a read-for-ownership transaction at write time.
enum class PrefetchHint { T0, T1, T2, NTA, WriteT0, WriteT1, WriteT2, WriteStreaming };

// ===========================================================================
// Prefetch
// ===========================================================================

template <PrefetchHint Hint = PrefetchHint::T2>
EINSUMS_FORCEINLINE void prefetch(void const *ptr) {
#if defined(__SSE2__) || defined(__AVX__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    if constexpr (Hint == PrefetchHint::T0)
        _mm_prefetch(reinterpret_cast<char const *>(ptr), _MM_HINT_T0);
    else if constexpr (Hint == PrefetchHint::T1)
        _mm_prefetch(reinterpret_cast<char const *>(ptr), _MM_HINT_T1);
    else if constexpr (Hint == PrefetchHint::T2)
        _mm_prefetch(reinterpret_cast<char const *>(ptr), _MM_HINT_T2);
    else if constexpr (Hint == PrefetchHint::NTA)
        _mm_prefetch(reinterpret_cast<char const *>(ptr), _MM_HINT_NTA);
    else if constexpr (Hint == PrefetchHint::WriteT0 || Hint == PrefetchHint::WriteT1 || Hint == PrefetchHint::WriteT2 ||
                       Hint == PrefetchHint::WriteStreaming) {
#    if defined(__PRFCHW__) || defined(__3dNOW__) || (defined(__GNUC__) && !defined(__clang__))
        // GCC accepts the rw=1 form of __builtin_prefetch and lowers to PREFETCHW; older
        // x86 cores without 3DNow!/PRFCHW silently no-op.
        constexpr int loc = (Hint == PrefetchHint::WriteT0)   ? 3
                            : (Hint == PrefetchHint::WriteT1) ? 2
                            : (Hint == PrefetchHint::WriteT2) ? 1
                                                              : 0;
        __builtin_prefetch(ptr, 1, loc);
#    else
        // Fallback: a read prefetch is better than nothing.
        _mm_prefetch(reinterpret_cast<char const *>(ptr), _MM_HINT_T0);
#    endif
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    // __builtin_prefetch(addr, rw, locality)
    // rw: 0=read, 1=write; locality: 0=NTA, 1=L3, 2=L2, 3=L1
    if constexpr (Hint == PrefetchHint::T0)
        __builtin_prefetch(ptr, 0, 3);
    else if constexpr (Hint == PrefetchHint::T1)
        __builtin_prefetch(ptr, 0, 2);
    else if constexpr (Hint == PrefetchHint::T2)
        __builtin_prefetch(ptr, 0, 1);
    else if constexpr (Hint == PrefetchHint::NTA)
        __builtin_prefetch(ptr, 0, 0);
    else if constexpr (Hint == PrefetchHint::WriteT0)
        __builtin_prefetch(ptr, 1, 3);
    else if constexpr (Hint == PrefetchHint::WriteT1)
        __builtin_prefetch(ptr, 1, 2);
    else if constexpr (Hint == PrefetchHint::WriteT2)
        __builtin_prefetch(ptr, 1, 1);
    else if constexpr (Hint == PrefetchHint::WriteStreaming)
        __builtin_prefetch(ptr, 1, 0);
#else
    (void)ptr;
#endif
}

/// Prefetch N consecutive rows (pattern used by HPTT micro_kernel).
template <typename T, PrefetchHint Hint = PrefetchHint::T2>
EINSUMS_FORCEINLINE void prefetch_rows(T const *base, size_t stride, int nrows = Vec<T>::lanes) {
    for (int i = 0; i < nrows; ++i)
        prefetch<Hint>(base + i * stride);
}

// ===========================================================================
// Streaming store (non-temporal): bypasses cache for write-only patterns
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE void stream_store(T *ptr, Vec<T> v);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE void stream_store(float *p, Vec<float> v) {
    _mm512_stream_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void stream_store(double *p, Vec<double> v) {
    _mm512_stream_pd(p, v.reg);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE void stream_store(float *p, Vec<float> v) {
    _mm256_stream_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void stream_store(double *p, Vec<double> v) {
    _mm256_stream_pd(p, v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE void stream_store(float *p, Vec<float> v) {
    _mm_stream_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void stream_store(double *p, Vec<double> v) {
    _mm_stream_pd(p, v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON has no SIMD non-temporal store, but Clang's
// __builtin_nontemporal_store lowers to STNP (store-pair non-temporal)
// on aarch64, which writes through L1 to L2 (avoiding cache pollution
// for write-only patterns). On compilers that don't recognize the
// builtin, fall back to a plain SIMD store.
template <>
EINSUMS_FORCEINLINE void stream_store(float *p, Vec<float> v) {
#    if defined(__clang__) || __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(v.reg, reinterpret_cast<float32x4_t *>(p));
#    else
    vst1q_f32(p, v.reg);
#    endif
}
template <>
EINSUMS_FORCEINLINE void stream_store(double *p, Vec<double> v) {
#    if defined(__clang__) || __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(v.reg, reinterpret_cast<float64x2_t *>(p));
#    else
    vst1q_f64(p, v.reg);
#    endif
}
#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE void stream_store(half_t *p, Vec<half_t> v) {
#        if defined(__clang__) || __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(v.reg, reinterpret_cast<float16x8_t *>(p));
#        else
    vst1q_f16(p, v.reg);
#        endif
}
#    endif
#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE void stream_store(bfloat16_t *p, Vec<bfloat16_t> v) {
#        if defined(__clang__) || __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(v.reg, reinterpret_cast<bfloat16x8_t *>(p));
#        else
    vst1q_bf16(p, v.reg);
#        endif
}
#    endif
#else
template <>
EINSUMS_FORCEINLINE void stream_store(float *p, Vec<float> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void stream_store(double *p, Vec<double> v) {
    *p = v.reg;
}
#endif

} // namespace einsums::simd
