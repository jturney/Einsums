//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/SIMD/Vec.hpp>

namespace einsums::simd {

// ===========================================================================
// Broadcast: scalar → Vec<T>
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> broadcast(T val);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE Vec<float> broadcast(float val) {
    return _mm512_set1_ps(val);
}
template <>
EINSUMS_FORCEINLINE Vec<double> broadcast(double val) {
    return _mm512_set1_pd(val);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> broadcast(float val) {
    return _mm256_set1_ps(val);
}
template <>
EINSUMS_FORCEINLINE Vec<double> broadcast(double val) {
    return _mm256_set1_pd(val);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> broadcast(float val) {
    return _mm_set1_ps(val);
}
template <>
EINSUMS_FORCEINLINE Vec<double> broadcast(double val) {
    return _mm_set1_pd(val);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> broadcast(float val) {
    return vdupq_n_f32(val);
}
template <>
EINSUMS_FORCEINLINE Vec<double> broadcast(double val) {
    return vdupq_n_f64(val);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> broadcast(float val) {
    return {val};
}
template <>
EINSUMS_FORCEINLINE Vec<double> broadcast(double val) {
    return {val};
}
#endif

// ===========================================================================
// Load (unaligned)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> loadu(T const *ptr);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p) {
    return _mm512_loadu_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p) {
    return _mm512_loadu_pd(p);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p) {
    return _mm256_loadu_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p) {
    return _mm256_loadu_pd(p);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p) {
    return _mm_loadu_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p) {
    return _mm_loadu_pd(p);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p) {
    return vld1q_f32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p) {
    return vld1q_f64(p);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p) {
    return {*p};
}
#endif

// ===========================================================================
// Load (aligned)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> loada(T const *ptr);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE Vec<float> loada(float const *p) {
    return _mm512_load_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loada(double const *p) {
    return _mm512_load_pd(p);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> loada(float const *p) {
    return _mm256_load_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loada(double const *p) {
    return _mm256_load_pd(p);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> loada(float const *p) {
    return _mm_load_ps(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loada(double const *p) {
    return _mm_load_pd(p);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON vld1q does not require alignment on aarch64
template <>
EINSUMS_FORCEINLINE Vec<float> loada(float const *p) {
    return vld1q_f32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loada(double const *p) {
    return vld1q_f64(p);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> loada(float const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<double> loada(double const *p) {
    return {*p};
}
#endif

// ===========================================================================
// Store (unaligned)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE void storeu(T *ptr, Vec<T> v);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v) {
    _mm512_storeu_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v) {
    _mm512_storeu_pd(p, v.reg);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v) {
    _mm256_storeu_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v) {
    _mm256_storeu_pd(p, v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v) {
    _mm_storeu_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v) {
    _mm_storeu_pd(p, v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v) {
    vst1q_f32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v) {
    vst1q_f64(p, v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v) {
    *p = v.reg;
}
#endif

// ===========================================================================
// Store (aligned)
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE void storea(T *ptr, Vec<T> v);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE void storea(float *p, Vec<float> v) {
    _mm512_store_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(double *p, Vec<double> v) {
    _mm512_store_pd(p, v.reg);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE void storea(float *p, Vec<float> v) {
    _mm256_store_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(double *p, Vec<double> v) {
    _mm256_store_pd(p, v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE void storea(float *p, Vec<float> v) {
    _mm_store_ps(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(double *p, Vec<double> v) {
    _mm_store_pd(p, v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE void storea(float *p, Vec<float> v) {
    vst1q_f32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(double *p, Vec<double> v) {
    vst1q_f64(p, v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE void storea(float *p, Vec<float> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storea(double *p, Vec<double> v) {
    *p = v.reg;
}
#endif

// ===========================================================================
// Arithmetic: add, sub, mul
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> add(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> sub(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> mul(Vec<T> a, Vec<T> b);

#if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE Vec<float> add(Vec<float> a, Vec<float> b) {
    return _mm512_add_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> add(Vec<double> a, Vec<double> b) {
    return _mm512_add_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sub(Vec<float> a, Vec<float> b) {
    return _mm512_sub_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sub(Vec<double> a, Vec<double> b) {
    return _mm512_sub_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> mul(Vec<float> a, Vec<float> b) {
    return _mm512_mul_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> mul(Vec<double> a, Vec<double> b) {
    return _mm512_mul_pd(a, b);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> add(Vec<float> a, Vec<float> b) {
    return _mm256_add_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> add(Vec<double> a, Vec<double> b) {
    return _mm256_add_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sub(Vec<float> a, Vec<float> b) {
    return _mm256_sub_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sub(Vec<double> a, Vec<double> b) {
    return _mm256_sub_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> mul(Vec<float> a, Vec<float> b) {
    return _mm256_mul_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> mul(Vec<double> a, Vec<double> b) {
    return _mm256_mul_pd(a, b);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> add(Vec<float> a, Vec<float> b) {
    return _mm_add_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> add(Vec<double> a, Vec<double> b) {
    return _mm_add_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sub(Vec<float> a, Vec<float> b) {
    return _mm_sub_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sub(Vec<double> a, Vec<double> b) {
    return _mm_sub_pd(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> mul(Vec<float> a, Vec<float> b) {
    return _mm_mul_ps(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> mul(Vec<double> a, Vec<double> b) {
    return _mm_mul_pd(a, b);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> add(Vec<float> a, Vec<float> b) {
    return vaddq_f32(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> add(Vec<double> a, Vec<double> b) {
    return vaddq_f64(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sub(Vec<float> a, Vec<float> b) {
    return vsubq_f32(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sub(Vec<double> a, Vec<double> b) {
    return vsubq_f64(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<float> mul(Vec<float> a, Vec<float> b) {
    return vmulq_f32(a, b);
}
template <>
EINSUMS_FORCEINLINE Vec<double> mul(Vec<double> a, Vec<double> b) {
    return vmulq_f64(a, b);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> add(Vec<float> a, Vec<float> b) {
    return {a.reg + b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> add(Vec<double> a, Vec<double> b) {
    return {a.reg + b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<float> sub(Vec<float> a, Vec<float> b) {
    return {a.reg - b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> sub(Vec<double> a, Vec<double> b) {
    return {a.reg - b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<float> mul(Vec<float> a, Vec<float> b) {
    return {a.reg * b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> mul(Vec<double> a, Vec<double> b) {
    return {a.reg * b.reg};
}
#endif

// ===========================================================================
// FMA: a * b + c
// Uses hardware FMA when available, otherwise mul + add.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> fmadd(Vec<T> a, Vec<T> b, Vec<T> c);

// x86 with FMA3 support (available on AVX2+ and some AVX processors)
#if defined(__FMA__)
#    if defined(__AVX512F__)
template <>
EINSUMS_FORCEINLINE Vec<float> fmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return _mm512_fmadd_ps(a, b, c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return _mm512_fmadd_pd(a, b, c);
}
#    elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> fmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return _mm256_fmadd_ps(a, b, c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return _mm256_fmadd_pd(a, b, c);
}
#    else // SSE + FMA (rare but possible)
template <>
EINSUMS_FORCEINLINE Vec<float> fmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return _mm_fmadd_ps(a, b, c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return _mm_fmadd_pd(a, b, c);
}
#    endif
// ARM NEON: FMA is always available on aarch64
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> fmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return vfmaq_f32(c, a, b); // NEON fma: c + a*b
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return vfmaq_f64(c, a, b);
}
// Fallback: mul + add
#else
template <>
EINSUMS_FORCEINLINE Vec<float> fmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return add(mul(a, b), c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return add(mul(a, b), c);
}
#endif

// ===========================================================================
// Operator overloads (opt-out via EINSUMS_SIMD_NO_OPERATORS)
// ===========================================================================

#if !defined(EINSUMS_SIMD_NO_OPERATORS)
template <typename T>
EINSUMS_FORCEINLINE Vec<T> operator+(Vec<T> a, Vec<T> b) {
    return add(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Vec<T> operator-(Vec<T> a, Vec<T> b) {
    return sub(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Vec<T> operator*(Vec<T> a, Vec<T> b) {
    return mul(a, b);
}
#endif

// ===========================================================================
// Integer SIMD operations — Vec<int32_t>, Vec<uint32_t>, Vec<int64_t>,
// Vec<uint64_t>.
//
// On x86 all integer widths share the same register type (__m128i / __m256i /
// __m512i), so the load/store/broadcast intrinsics dispatch on register
// width regardless of element type. The per-element-width arithmetic
// intrinsics are different (epi32 vs epi64 etc.). On NEON each (signedness,
// width) pair has its own type and intrinsic family.
//
// Coverage in this section:
//   broadcast, loadu, loada, storeu, storea
//   add, sub
//   bitwise and / or / xor
//   logical shift left / right (immediate count)
//   compare-equal
//
// Skipped (intentionally) in this round:
//   stream_store on integers — no observed user; add when needed.
//   i64 multiply on SSE2 — no native instruction; emulation costs more
//   than scalar fallback for typical kernels. Add later if a user needs it.
//   i8 / i16 vectors — not in scope; saturation semantics double the surface.
// ===========================================================================

// ── broadcast ─────────────────────────────────────────────────────────────

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> broadcast(int32_t v) {
    return _mm512_set1_epi32(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> broadcast(uint32_t v) {
    return _mm512_set1_epi32(static_cast<int32_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> broadcast(int64_t v) {
    return _mm512_set1_epi64(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> broadcast(uint64_t v) {
    return _mm512_set1_epi64(static_cast<int64_t>(v));
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> broadcast(int32_t v) {
    return _mm256_set1_epi32(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> broadcast(uint32_t v) {
    return _mm256_set1_epi32(static_cast<int32_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> broadcast(int64_t v) {
    return _mm256_set1_epi64x(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> broadcast(uint64_t v) {
    return _mm256_set1_epi64x(static_cast<int64_t>(v));
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> broadcast(int32_t v) {
    return _mm_set1_epi32(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> broadcast(uint32_t v) {
    return _mm_set1_epi32(static_cast<int32_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> broadcast(int64_t v) {
    return _mm_set1_epi64x(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> broadcast(uint64_t v) {
    return _mm_set1_epi64x(static_cast<int64_t>(v));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> broadcast(int32_t v) {
    return vdupq_n_s32(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> broadcast(uint32_t v) {
    return vdupq_n_u32(v);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> broadcast(int64_t v) {
    return vdupq_n_s64(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> broadcast(uint64_t v) {
    return vdupq_n_u64(v);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<int32_t> broadcast(int32_t v) {
    return {v};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> broadcast(uint32_t v) {
    return {v};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> broadcast(int64_t v) {
    return {v};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> broadcast(uint64_t v) {
    return {v};
}
#endif

// ── loadu / loada / storeu / storea ───────────────────────────────────────
//
// Helper macros: x86 integer load/store all dispatch on register width
// regardless of element type (the same `__m{128,256,512}i const *` cast).
// One macro expands to all four element types per ISA tier.

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_INT_LDST_X86(LOADU, LOADA, STOREU, STOREA, RT)                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int32_t> loadu(int32_t const *p) {                                                                         \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint32_t> loadu(uint32_t const *p) {                                                                       \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int64_t> loadu(int64_t const *p) {                                                                         \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint64_t> loadu(uint64_t const *p) {                                                                       \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int32_t> loada(int32_t const *p) {                                                                         \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint32_t> loada(uint32_t const *p) {                                                                       \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int64_t> loada(int64_t const *p) {                                                                         \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint64_t> loada(uint64_t const *p) {                                                                       \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(int32_t *p, Vec<int32_t> v) {                                                                      \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(uint32_t *p, Vec<uint32_t> v) {                                                                    \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(int64_t *p, Vec<int64_t> v) {                                                                      \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(uint64_t *p, Vec<uint64_t> v) {                                                                    \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(int32_t *p, Vec<int32_t> v) {                                                                      \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(uint32_t *p, Vec<uint32_t> v) {                                                                    \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(int64_t *p, Vec<int64_t> v) {                                                                      \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(uint64_t *p, Vec<uint64_t> v) {                                                                    \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }
EINSUMS_SIMD_INT_LDST_X86(_mm512_loadu_si512, _mm512_load_si512, _mm512_storeu_si512, _mm512_store_si512, __m512i)
#    undef EINSUMS_SIMD_INT_LDST_X86
#elif defined(__AVX__)
// Same pattern as AVX-512; just narrower register type.
#    define EINSUMS_SIMD_INT_LDST_X86(LOADU, LOADA, STOREU, STOREA, RT)                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int32_t> loadu(int32_t const *p) {                                                                         \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint32_t> loadu(uint32_t const *p) {                                                                       \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int64_t> loadu(int64_t const *p) {                                                                         \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint64_t> loadu(uint64_t const *p) {                                                                       \
            return LOADU(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int32_t> loada(int32_t const *p) {                                                                         \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint32_t> loada(uint32_t const *p) {                                                                       \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<int64_t> loada(int64_t const *p) {                                                                         \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<uint64_t> loada(uint64_t const *p) {                                                                       \
            return LOADA(reinterpret_cast<RT const *>(p));                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(int32_t *p, Vec<int32_t> v) {                                                                      \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(uint32_t *p, Vec<uint32_t> v) {                                                                    \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(int64_t *p, Vec<int64_t> v) {                                                                      \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(uint64_t *p, Vec<uint64_t> v) {                                                                    \
            STOREU(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(int32_t *p, Vec<int32_t> v) {                                                                      \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(uint32_t *p, Vec<uint32_t> v) {                                                                    \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(int64_t *p, Vec<int64_t> v) {                                                                      \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storea(uint64_t *p, Vec<uint64_t> v) {                                                                    \
            STOREA(reinterpret_cast<RT *>(p), v.reg);                                                                                      \
        }
EINSUMS_SIMD_INT_LDST_X86(_mm256_loadu_si256, _mm256_load_si256, _mm256_storeu_si256, _mm256_store_si256, __m256i)
#    undef EINSUMS_SIMD_INT_LDST_X86
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loadu(int32_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loadu(uint32_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loadu(int64_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loadu(uint64_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loada(int32_t const *p) {
    return _mm_load_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loada(uint32_t const *p) {
    return _mm_load_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loada(int64_t const *p) {
    return _mm_load_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loada(uint64_t const *p) {
    return _mm_load_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE void storeu(int32_t *p, Vec<int32_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint32_t *p, Vec<uint32_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(int64_t *p, Vec<int64_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint64_t *p, Vec<uint64_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(int32_t *p, Vec<int32_t> v) {
    _mm_store_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(uint32_t *p, Vec<uint32_t> v) {
    _mm_store_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(int64_t *p, Vec<int64_t> v) {
    _mm_store_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(uint64_t *p, Vec<uint64_t> v) {
    _mm_store_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loadu(int32_t const *p) {
    return vld1q_s32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loadu(uint32_t const *p) {
    return vld1q_u32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loadu(int64_t const *p) {
    return vld1q_s64(p);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loadu(uint64_t const *p) {
    return vld1q_u64(p);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loada(int32_t const *p) {
    return vld1q_s32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loada(uint32_t const *p) {
    return vld1q_u32(p);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loada(int64_t const *p) {
    return vld1q_s64(p);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loada(uint64_t const *p) {
    return vld1q_u64(p);
}
template <>
EINSUMS_FORCEINLINE void storeu(int32_t *p, Vec<int32_t> v) {
    vst1q_s32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint32_t *p, Vec<uint32_t> v) {
    vst1q_u32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(int64_t *p, Vec<int64_t> v) {
    vst1q_s64(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint64_t *p, Vec<uint64_t> v) {
    vst1q_u64(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(int32_t *p, Vec<int32_t> v) {
    vst1q_s32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(uint32_t *p, Vec<uint32_t> v) {
    vst1q_u32(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(int64_t *p, Vec<int64_t> v) {
    vst1q_s64(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(uint64_t *p, Vec<uint64_t> v) {
    vst1q_u64(p, v.reg);
}
#else
// Scalar fallback — Vec<T>::reg is a single T.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loadu(int32_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loadu(uint32_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loadu(int64_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loadu(uint64_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> loada(int32_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> loada(uint32_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> loada(int64_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> loada(uint64_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE void storeu(int32_t *p, Vec<int32_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storeu(uint32_t *p, Vec<uint32_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storeu(int64_t *p, Vec<int64_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storeu(uint64_t *p, Vec<uint64_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storea(int32_t *p, Vec<int32_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storea(uint32_t *p, Vec<uint32_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storea(int64_t *p, Vec<int64_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storea(uint64_t *p, Vec<uint64_t> v) {
    *p = v.reg;
}
#endif

// ===========================================================================
// Integer arithmetic — add / sub / mul.
//
// `add` and `sub` are trivial across every ISA tier and signedness because
// two's-complement wraparound is bit-identical for signed and unsigned —
// the same instruction handles both.
//
// `mul` returns the low half of the elementwise product (matching every
// SIMD ISA's "low multiply" semantics; full 32×32→64 multiplies are a
// different operation we skip here).
//
// Coverage caveats — instructions that don't exist on the tier:
//   - SSE2 has no 32-bit integer multiply (added in SSE4.1 as PMULLD); we
//     skip Vec<int32_t>/Vec<uint32_t> mul on SSE2 unless SSE4.1 is on.
//   - SSE2/AVX/AVX2 have no native 64-bit element multiply. AVX-512DQ adds
//     VPMULLQ; without DQ we skip i64/u64 mul on those tiers.
//   - aarch64 NEON has no 64-bit integer vector multiply (vmulq_s64 only
//     exists under SVE2). We skip i64/u64 mul on NEON.
// Calls to a missing specialization produce a link error referencing the
// primary template — clear signal to the user that the op needs a wider
// ISA.
// ===========================================================================

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> add(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm512_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> add(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm512_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> add(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm512_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> add(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm512_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> sub(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm512_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> sub(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm512_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> sub(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm512_sub_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> sub(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm512_sub_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm512_mullo_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm512_mullo_epi32(a.reg, b.reg);
}
#    if defined(__AVX512DQ__)
// AVX-512DQ adds VPMULLQ for 64-bit element multiply.
template <>
EINSUMS_FORCEINLINE Vec<int64_t> mul(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm512_mullo_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> mul(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm512_mullo_epi64(a.reg, b.reg);
}
#    endif
#elif defined(__AVX2__)
// 256-bit integer arithmetic requires AVX2 — AVX1 only had float ops.
// Chips with __AVX__ but not __AVX2__ get a link error here, which is the
// right signal: the integer Vec on that tier won't work.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> add(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm256_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> add(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm256_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> add(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm256_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> add(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm256_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> sub(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm256_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> sub(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm256_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> sub(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm256_sub_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> sub(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm256_sub_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm256_mullo_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm256_mullo_epi32(a.reg, b.reg);
}
// i64/u64 mul not implemented — needs AVX-512DQ.
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> add(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> add(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm_add_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> add(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> add(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm_add_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> sub(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> sub(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm_sub_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> sub(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm_sub_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> sub(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm_sub_epi64(a.reg, b.reg);
}
#    if defined(__SSE4_1__)
// PMULLD is SSE4.1; SSE2 has no 32×32→32 multiply.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm_mullo_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm_mullo_epi32(a.reg, b.reg);
}
#    else
// SSE2 fallback (e.g. -march=nocona): no PMULLD. Emulate via two _mm_mul_epu32
// (32×32→64 on even lanes) and repack the low 32 bits of each product. The low half is
// identical for signed and unsigned, so the same sequence serves both.
EINSUMS_FORCEINLINE __m128i einsums_sse2_mullo_epi32(__m128i a, __m128i b) {
    __m128i e02 = _mm_mul_epu32(a, b);                                          // products of lanes 0,2
    __m128i e13 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));    // products of lanes 1,3
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(e02, _MM_SHUFFLE(0, 0, 2, 0)),  // low 32 bits of 0,2
                              _mm_shuffle_epi32(e13, _MM_SHUFFLE(0, 0, 2, 0))); // low 32 bits of 1,3
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return einsums_sse2_mullo_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return einsums_sse2_mullo_epi32(a.reg, b.reg);
}
#    endif
// i64/u64 mul not implemented — needs AVX-512DQ.
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> add(Vec<int32_t> a, Vec<int32_t> b) {
    return vaddq_s32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> add(Vec<uint32_t> a, Vec<uint32_t> b) {
    return vaddq_u32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> add(Vec<int64_t> a, Vec<int64_t> b) {
    return vaddq_s64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> add(Vec<uint64_t> a, Vec<uint64_t> b) {
    return vaddq_u64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> sub(Vec<int32_t> a, Vec<int32_t> b) {
    return vsubq_s32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> sub(Vec<uint32_t> a, Vec<uint32_t> b) {
    return vsubq_u32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> sub(Vec<int64_t> a, Vec<int64_t> b) {
    return vsubq_s64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> sub(Vec<uint64_t> a, Vec<uint64_t> b) {
    return vsubq_u64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return vmulq_s32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return vmulq_u32(a.reg, b.reg);
}
// i64/u64 mul not implemented — aarch64 NEON has no 64-bit element mul.
#else
// Scalar fallback: Vec<T>::reg is a single T. Two's-complement wraparound
// on signed/unsigned overflow matches every hardware path above.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> add(Vec<int32_t> a, Vec<int32_t> b) {
    return {static_cast<int32_t>(a.reg + b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> add(Vec<uint32_t> a, Vec<uint32_t> b) {
    return {static_cast<uint32_t>(a.reg + b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> add(Vec<int64_t> a, Vec<int64_t> b) {
    return {static_cast<int64_t>(a.reg + b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> add(Vec<uint64_t> a, Vec<uint64_t> b) {
    return {static_cast<uint64_t>(a.reg + b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> sub(Vec<int32_t> a, Vec<int32_t> b) {
    return {static_cast<int32_t>(a.reg - b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> sub(Vec<uint32_t> a, Vec<uint32_t> b) {
    return {static_cast<uint32_t>(a.reg - b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> sub(Vec<int64_t> a, Vec<int64_t> b) {
    return {static_cast<int64_t>(a.reg - b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> sub(Vec<uint64_t> a, Vec<uint64_t> b) {
    return {static_cast<uint64_t>(a.reg - b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> mul(Vec<int32_t> a, Vec<int32_t> b) {
    return {static_cast<int32_t>(a.reg * b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> mul(Vec<uint32_t> a, Vec<uint32_t> b) {
    return {static_cast<uint32_t>(a.reg * b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> mul(Vec<int64_t> a, Vec<int64_t> b) {
    return {static_cast<int64_t>(a.reg * b.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> mul(Vec<uint64_t> a, Vec<uint64_t> b) {
    return {static_cast<uint64_t>(a.reg * b.reg)};
}
#endif

// ===========================================================================
// Bitwise — and / or / xor.
//
// Single intrinsic per ISA per width — no element-type dispatch needed
// since the bit pattern is what matters. NEON exposes per-signedness/width
// names but they all alias the same vand/vor/veor instruction.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b);

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_INT_BITWISE(T)                                                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm512_and_si512(a.reg, b.reg);                                                                                         \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return _mm512_or_si512(a.reg, b.reg);                                                                                          \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm512_xor_si512(a.reg, b.reg);                                                                                         \
        }
#elif defined(__AVX2__)
#    define EINSUMS_SIMD_INT_BITWISE(T)                                                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm256_and_si256(a.reg, b.reg);                                                                                         \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return _mm256_or_si256(a.reg, b.reg);                                                                                          \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm256_xor_si256(a.reg, b.reg);                                                                                         \
        }
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define EINSUMS_SIMD_INT_BITWISE(T)                                                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm_and_si128(a.reg, b.reg);                                                                                            \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return _mm_or_si128(a.reg, b.reg);                                                                                             \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return _mm_xor_si128(a.reg, b.reg);                                                                                            \
        }
#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON: vand/vorr/veor — typed by signedness and width but all alias the same hw op.
#    define EINSUMS_SIMD_INT_BITWISE_ONE(T, AND_F, OR_F, XOR_F)                                                                            \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return AND_F(a.reg, b.reg);                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return OR_F(a.reg, b.reg);                                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return XOR_F(a.reg, b.reg);                                                                                                    \
        }
EINSUMS_SIMD_INT_BITWISE_ONE(int32_t, vandq_s32, vorrq_s32, veorq_s32)
EINSUMS_SIMD_INT_BITWISE_ONE(uint32_t, vandq_u32, vorrq_u32, veorq_u32)
EINSUMS_SIMD_INT_BITWISE_ONE(int64_t, vandq_s64, vorrq_s64, veorq_s64)
EINSUMS_SIMD_INT_BITWISE_ONE(uint64_t, vandq_u64, vorrq_u64, veorq_u64)
#    undef EINSUMS_SIMD_INT_BITWISE_ONE
#    define EINSUMS_SIMD_INT_BITWISE(T) /* nothing — NEON expanded above per type */
#else
// Scalar fallback.
#    define EINSUMS_SIMD_INT_BITWISE(T)                                                                                                    \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return {static_cast<T>(a.reg & b.reg)};                                                                                        \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return {static_cast<T>(a.reg | b.reg)};                                                                                        \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return {static_cast<T>(a.reg ^ b.reg)};                                                                                        \
        }
#endif

EINSUMS_SIMD_INT_BITWISE(int32_t)
EINSUMS_SIMD_INT_BITWISE(uint32_t)
EINSUMS_SIMD_INT_BITWISE(int64_t)
EINSUMS_SIMD_INT_BITWISE(uint64_t)
#undef EINSUMS_SIMD_INT_BITWISE

// ===========================================================================
// Logical shifts — shift_left<N>(v) and shift_right<N>(v).
//
// The shift count is a non-type template parameter so the intrinsics
// generated are immediate-form (best codegen). Variable-count shifts are a
// separate API not added in this round.
//
// `shift_right` is *logical* (zero-fill on the high bit). Arithmetic shift
// (sign-extending) is a separate operation we'll add when there's a use.
// ===========================================================================

template <int N, typename T>
EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v);
template <int N, typename T>
EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v);

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v) {
    return _mm512_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v) {
    return _mm512_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v) {
    return _mm512_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v) {
    return _mm512_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v) {
    return _mm512_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v) {
    return _mm512_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v) {
    return _mm512_srli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v) {
    return _mm512_srli_epi64(v.reg, N);
}
#elif defined(__AVX2__)
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v) {
    return _mm256_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v) {
    return _mm256_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v) {
    return _mm256_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v) {
    return _mm256_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v) {
    return _mm256_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v) {
    return _mm256_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v) {
    return _mm256_srli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v) {
    return _mm256_srli_epi64(v.reg, N);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v) {
    return _mm_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v) {
    return _mm_slli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v) {
    return _mm_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v) {
    return _mm_slli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v) {
    return _mm_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v) {
    return _mm_srli_epi32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v) {
    return _mm_srli_epi64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v) {
    return _mm_srli_epi64(v.reg, N);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON: vshlq_n_* and vshrq_n_* require N to be a literal in [1..bits].
// `vshlq_n_*` only accepts positive counts; for left shift by 0 the user
// should just not call it. Same for vshrq_n_*.
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v) {
    return vshlq_n_s32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v) {
    return vshlq_n_u32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v) {
    return vshlq_n_s64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v) {
    return vshlq_n_u64(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v) {
    // Logical right shift on signed: cast through unsigned and back.
    return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(v.reg), N));
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v) {
    return vshrq_n_u32(v.reg, N);
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v) {
    return vreinterpretq_s64_u64(vshrq_n_u64(vreinterpretq_u64_s64(v.reg), N));
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v) {
    return vshrq_n_u64(v.reg, N);
}
#else
// Scalar fallback. C++ semantics: logical shift right on signed is
// implementation-defined (uses arithmetic on most compilers); we cast
// through the unsigned counterpart to force zero-fill.
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v) {
    return {static_cast<int32_t>(static_cast<uint32_t>(v.reg) << N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v) {
    return {static_cast<uint32_t>(v.reg << N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v) {
    return {static_cast<int64_t>(static_cast<uint64_t>(v.reg) << N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v) {
    return {static_cast<uint64_t>(v.reg << N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v) {
    return {static_cast<int32_t>(static_cast<uint32_t>(v.reg) >> N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v) {
    return {static_cast<uint32_t>(v.reg >> N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v) {
    return {static_cast<int64_t>(static_cast<uint64_t>(v.reg) >> N)};
}
template <int N>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v) {
    return {static_cast<uint64_t>(v.reg >> N)};
}
#endif

// ===========================================================================
// Compare equal — cmp_eq.
//
// Returns a Vec<T> mask: each lane is all-1s (interpreted as -1 for signed,
// max value for unsigned) where a == b, otherwise 0. Suitable for use with
// bitwise ops to build conditional kernels without a branch.
//
// AVX-512 native compares produce a __mmask{8,16}; we round-trip through
// `maskz_set1` to deliver a Vec<T>-shaped result for cross-ISA consistency.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> cmp_eq(Vec<T> a, Vec<T> b);

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> cmp_eq(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a.reg, b.reg), -1);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> cmp_eq(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a.reg, b.reg), -1);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask(a.reg, b.reg), -1);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask(a.reg, b.reg), -1);
}
#elif defined(__AVX2__)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> cmp_eq(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm256_cmpeq_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> cmp_eq(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm256_cmpeq_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm256_cmpeq_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm256_cmpeq_epi64(a.reg, b.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> cmp_eq(Vec<int32_t> a, Vec<int32_t> b) {
    return _mm_cmpeq_epi32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> cmp_eq(Vec<uint32_t> a, Vec<uint32_t> b) {
    return _mm_cmpeq_epi32(a.reg, b.reg);
}
#    if defined(__SSE4_1__)
// PCMPEQQ is SSE4.1.
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return _mm_cmpeq_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return _mm_cmpeq_epi64(a.reg, b.reg);
}
#    else
// SSE2 fallback (e.g. -march=nocona): no PCMPEQQ. Compare the 32-bit halves, then AND each
// 64-bit lane with its half-swapped self so a lane is all-ones only when both halves matched.
EINSUMS_FORCEINLINE __m128i einsums_sse2_cmpeq_epi64(__m128i a, __m128i b) {
    __m128i t = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return einsums_sse2_cmpeq_epi64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return einsums_sse2_cmpeq_epi64(a.reg, b.reg);
}
#    endif
#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON: vceq returns a uint*x*_t mask; reinterpret to the typed result.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> cmp_eq(Vec<int32_t> a, Vec<int32_t> b) {
    return vreinterpretq_s32_u32(vceqq_s32(a.reg, b.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> cmp_eq(Vec<uint32_t> a, Vec<uint32_t> b) {
    return vceqq_u32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return vreinterpretq_s64_u64(vceqq_s64(a.reg, b.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return vceqq_u64(a.reg, b.reg);
}
#else
// Scalar fallback: -1 on match, 0 otherwise.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> cmp_eq(Vec<int32_t> a, Vec<int32_t> b) {
    return {a.reg == b.reg ? int32_t(-1) : int32_t(0)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> cmp_eq(Vec<uint32_t> a, Vec<uint32_t> b) {
    return {a.reg == b.reg ? ~uint32_t(0) : uint32_t(0)};
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> cmp_eq(Vec<int64_t> a, Vec<int64_t> b) {
    return {a.reg == b.reg ? int64_t(-1) : int64_t(0)};
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> cmp_eq(Vec<uint64_t> a, Vec<uint64_t> b) {
    return {a.reg == b.reg ? ~uint64_t(0) : uint64_t(0)};
}
#endif

// ===========================================================================
// 8-bit integer load / store / broadcast.
//
// Vec<int8_t> and Vec<uint8_t> exist primarily as inputs to the dot-product
// helpers below. We deliberately don't expose general lane-wise add/sub/mul
// for 8-bit because every ISA's 8-bit arithmetic has saturation surprises
// (PADDSB vs PADDB on x86; vqaddq vs vaddq on NEON) that would force users
// to memorize a per-platform behavior matrix. Quantized inner-loop kernels
// almost universally accumulate into int32 via dot products anyway.
// ===========================================================================

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<int8_t> broadcast(int8_t v) {
    return _mm512_set1_epi8(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> broadcast(uint8_t v) {
    return _mm512_set1_epi8(static_cast<int8_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int8_t> loadu(int8_t const *p) {
    return _mm512_loadu_si512(reinterpret_cast<__m512i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> loadu(uint8_t const *p) {
    return _mm512_loadu_si512(reinterpret_cast<__m512i const *>(p));
}
template <>
EINSUMS_FORCEINLINE void storeu(int8_t *p, Vec<int8_t> v) {
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint8_t *p, Vec<uint8_t> v) {
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(p), v.reg);
}
#elif defined(__AVX2__)
template <>
EINSUMS_FORCEINLINE Vec<int8_t> broadcast(int8_t v) {
    return _mm256_set1_epi8(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> broadcast(uint8_t v) {
    return _mm256_set1_epi8(static_cast<int8_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int8_t> loadu(int8_t const *p) {
    return _mm256_loadu_si256(reinterpret_cast<__m256i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> loadu(uint8_t const *p) {
    return _mm256_loadu_si256(reinterpret_cast<__m256i const *>(p));
}
template <>
EINSUMS_FORCEINLINE void storeu(int8_t *p, Vec<int8_t> v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint8_t *p, Vec<uint8_t> v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<int8_t> broadcast(int8_t v) {
    return _mm_set1_epi8(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> broadcast(uint8_t v) {
    return _mm_set1_epi8(static_cast<int8_t>(v));
}
template <>
EINSUMS_FORCEINLINE Vec<int8_t> loadu(int8_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> loadu(uint8_t const *p) {
    return _mm_loadu_si128(reinterpret_cast<__m128i const *>(p));
}
template <>
EINSUMS_FORCEINLINE void storeu(int8_t *p, Vec<int8_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint8_t *p, Vec<uint8_t> v) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<int8_t> broadcast(int8_t v) {
    return vdupq_n_s8(v);
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> broadcast(uint8_t v) {
    return vdupq_n_u8(v);
}
template <>
EINSUMS_FORCEINLINE Vec<int8_t> loadu(int8_t const *p) {
    return vld1q_s8(p);
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> loadu(uint8_t const *p) {
    return vld1q_u8(p);
}
template <>
EINSUMS_FORCEINLINE void storeu(int8_t *p, Vec<int8_t> v) {
    vst1q_s8(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(uint8_t *p, Vec<uint8_t> v) {
    vst1q_u8(p, v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<int8_t> broadcast(int8_t v) {
    return {v};
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> broadcast(uint8_t v) {
    return {v};
}
template <>
EINSUMS_FORCEINLINE Vec<int8_t> loadu(int8_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE Vec<uint8_t> loadu(uint8_t const *p) {
    return {*p};
}
template <>
EINSUMS_FORCEINLINE void storeu(int8_t *p, Vec<int8_t> v) {
    *p = v.reg;
}
template <>
EINSUMS_FORCEINLINE void storeu(uint8_t *p, Vec<uint8_t> v) {
    *p = v.reg;
}
#endif

// ===========================================================================
// Dot-product helpers — int8 × int8 → int32 accumulating.
//
// Three signedness combos:
//   - dot_product_ss(acc, a, b): signed × signed
//   - dot_product_uu(acc, a, b): unsigned × unsigned
//   - dot_product_us(acc, a, b): unsigned × signed (the standard ML-quant
//     shape; activations are unsigned, weights are signed)
//
// All three accumulate `acc + sum_of_4_byte_products` per int32 lane —
// 16 byte pairs → 4 int32 outputs per 128-bit register, scaling up to 64
// pairs → 16 outputs at AVX-512 width.
//
// Hardware coverage is uneven: ARM FEAT_DOTPROD gives ss + uu directly,
// FEAT_I8MM (M3+, requires `-mcpu=apple-m4` in this build) adds us.
// Intel AVX-VNNI / AVX-512 VNNI gives only the us shape natively. For the
// missing combos on each platform a kernel needs to either fall back to
// scalar or emulate via shuffle + sign-extend; we don't ship emulation
// here — calling a missing specialization fails to link, which is the
// cleanest signal that a wider ISA is needed.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_ss(Vec<int32_t> acc, Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_uu(Vec<int32_t> acc, Vec<T> a, Vec<T> b);
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b);

#if defined(__ARM_FEATURE_DOTPROD)
template <>
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_ss(Vec<int32_t> acc, Vec<int8_t> a, Vec<int8_t> b) {
    return vdotq_s32(acc.reg, a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_uu(Vec<int32_t> acc, Vec<uint8_t> a, Vec<uint8_t> b) {
    // vdotq_u32 returns uint32x4_t; reinterpret to the int32x4_t the
    // Vec<int32_t> traits use. Same bit pattern.
    return vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(acc.reg), a.reg, b.reg));
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b) {
    return vusdotq_s32(acc.reg, a.reg, b.reg);
}
#elif defined(__AVX512VNNI__) && defined(__AVX512F__)
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b) {
    return _mm512_dpbusd_epi32(acc.reg, a.reg, b.reg);
}
#elif defined(__AVXVNNI__) && defined(__AVX2__)
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b) {
    return _mm256_dpbusd_epi32(acc.reg, a.reg, b.reg);
}
#endif

// ===========================================================================
// IEEE half-precision (Vec<half_t>) — broadcast / load / store / arithmetic.
//
// Available when the build sees:
//   - `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` on aarch64 (M1+ default), or
//   - `__AVX512FP16__` on x86 (Sapphire Rapids+, requires -mavx512fp16).
//
// Operations covered: broadcast, loadu/loada, storeu/storea, add, sub, mul,
// fma. Other ops (gather, transpose, complex) are deferred — they need
// dedicated specializations in Shuffle.hpp / Gather.hpp / ComplexVec.hpp.
// ===========================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE Vec<half_t> broadcast(half_t v) {
    return vdupq_n_f16(v);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> loadu(half_t const *p) {
    return vld1q_f16(p);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> loada(half_t const *p) {
    return vld1q_f16(p);
}
template <>
EINSUMS_FORCEINLINE void storeu(half_t *p, Vec<half_t> v) {
    vst1q_f16(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(half_t *p, Vec<half_t> v) {
    vst1q_f16(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> add(Vec<half_t> a, Vec<half_t> b) {
    return vaddq_f16(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> sub(Vec<half_t> a, Vec<half_t> b) {
    return vsubq_f16(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> mul(Vec<half_t> a, Vec<half_t> b) {
    return vmulq_f16(a.reg, b.reg);
}
// `fma` is declared in the float/double section above with a signature of
// fma(a, b, c) = a*b + c. NEON's vfmaq_f16(acc, a, b) computes acc + a*b,
// which matches once we shuffle the args.
template <>
EINSUMS_FORCEINLINE Vec<half_t> fmadd(Vec<half_t> a, Vec<half_t> b, Vec<half_t> c) {
    return vfmaq_f16(c.reg, a.reg, b.reg);
}
#elif defined(__AVX512FP16__)
template <>
EINSUMS_FORCEINLINE Vec<half_t> broadcast(half_t v) {
    return _mm512_set1_ph(v);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> loadu(half_t const *p) {
    return _mm512_loadu_ph(p);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> loada(half_t const *p) {
    return _mm512_load_ph(p);
}
template <>
EINSUMS_FORCEINLINE void storeu(half_t *p, Vec<half_t> v) {
    _mm512_storeu_ph(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(half_t *p, Vec<half_t> v) {
    _mm512_store_ph(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> add(Vec<half_t> a, Vec<half_t> b) {
    return _mm512_add_ph(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> sub(Vec<half_t> a, Vec<half_t> b) {
    return _mm512_sub_ph(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> mul(Vec<half_t> a, Vec<half_t> b) {
    return _mm512_mul_ph(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> fmadd(Vec<half_t> a, Vec<half_t> b, Vec<half_t> c) {
    return _mm512_fmadd_ph(a.reg, b.reg, c.reg);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC || __AVX512FP16__

// ===========================================================================
// Brain-float bf16 (Vec<bfloat16_t>) — load / store only.
//
// Both ARM BF16 and AVX-512BF16 expose conversion to/from FP32 plus dot
// product, but no native lane-wise add/sub/mul on bf16. The convention
// in BF16 kernels is "convert to FP32, do math, convert back" — so we
// expose only what the hardware supports natively here. Convert helpers
// land in a follow-up alongside the `bf16_to_float` / `float_to_bf16`
// API, plus the `bf16_dot_product` helper that uses FEAT_BF16's BFDOT
// or AVX-512 VDPBF16PS.
// ===========================================================================

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> loadu(bfloat16_t const *p) {
    return vld1q_bf16(p);
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> loada(bfloat16_t const *p) {
    return vld1q_bf16(p);
}
template <>
EINSUMS_FORCEINLINE void storeu(bfloat16_t *p, Vec<bfloat16_t> v) {
    vst1q_bf16(p, v.reg);
}
template <>
EINSUMS_FORCEINLINE void storea(bfloat16_t *p, Vec<bfloat16_t> v) {
    vst1q_bf16(p, v.reg);
}
#elif defined(__AVX512BF16__)
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> loadu(bfloat16_t const *p) {
    // __m512bh and __m512i alias each other on Intel ICC/Clang — the load
    // is the integer variant since there's no `_mm512_loadu_pbh` intrinsic.
    return reinterpret_cast<__m512bh>(_mm512_loadu_si512(reinterpret_cast<__m512i const *>(p)));
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> loada(bfloat16_t const *p) {
    return reinterpret_cast<__m512bh>(_mm512_load_si512(reinterpret_cast<__m512i const *>(p)));
}
template <>
EINSUMS_FORCEINLINE void storeu(bfloat16_t *p, Vec<bfloat16_t> v) {
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(p), reinterpret_cast<__m512i>(v.reg));
}
template <>
EINSUMS_FORCEINLINE void storea(bfloat16_t *p, Vec<bfloat16_t> v) {
    _mm512_store_si512(reinterpret_cast<__m512i *>(p), reinterpret_cast<__m512i>(v.reg));
}
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC || __AVX512BF16__

} // namespace einsums::simd
