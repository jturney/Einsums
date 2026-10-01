//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ===========================================================================
// Broadcast: scalar → Vec<T>
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> broadcast(T val);

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

#if defined(__AVX512F__) && defined(__AVX512VL__)
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

// x86 with FMA3 support (available on AVX2+ and some AVX processors). See
// Platform.hpp for why this is not a bare __FMA__ test.
#if defined(EINSUMS_SIMD_HAVE_FMA)
#    if defined(__AVX512F__) && defined(__AVX512VL__)
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
// Arithmetic: div, sqrt, min, max, abs, neg (float and double).
//
// min(a, b) is exactly a < b ? a : b, and max(a, b) exactly a > b ? a : b: when
// either lane is NaN, or both are zeros of either sign, the result is b. That
// is what x86 MINPS/MAXPS compute; NEON's own vminq propagates NaN instead, so
// NEON builds the same selection from a compare. abs clears the sign bit and
// neg flips it, so neg of +0.0 is -0.0 and neither changes a NaN's payload.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> div(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> sqrt(Vec<T> a);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> min(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> max(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> abs(Vec<T> a);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> neg(Vec<T> a);

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<float> div(Vec<float> a, Vec<float> b) {
    return _mm512_div_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> div(Vec<double> a, Vec<double> b) {
    return _mm512_div_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sqrt(Vec<float> a) {
    return _mm512_sqrt_ps(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sqrt(Vec<double> a) {
    return _mm512_sqrt_pd(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> min(Vec<float> a, Vec<float> b) {
    return _mm512_min_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> min(Vec<double> a, Vec<double> b) {
    return _mm512_min_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> max(Vec<float> a, Vec<float> b) {
    return _mm512_max_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> max(Vec<double> a, Vec<double> b) {
    return _mm512_max_pd(a.reg, b.reg);
}
// The floating-point and/xor intrinsics are AVX-512DQ; the integer ones are F.
template <>
EINSUMS_FORCEINLINE Vec<float> abs(Vec<float> a) {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a.reg), _mm512_set1_epi32(0x7FFFFFFF)));
}
template <>
EINSUMS_FORCEINLINE Vec<double> abs(Vec<double> a) {
    return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a.reg), _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF)));
}
template <>
EINSUMS_FORCEINLINE Vec<float> neg(Vec<float> a) {
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.reg), _mm512_set1_epi32(INT32_MIN)));
}
template <>
EINSUMS_FORCEINLINE Vec<double> neg(Vec<double> a) {
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.reg), _mm512_set1_epi64(INT64_MIN)));
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> div(Vec<float> a, Vec<float> b) {
    return _mm256_div_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> div(Vec<double> a, Vec<double> b) {
    return _mm256_div_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sqrt(Vec<float> a) {
    return _mm256_sqrt_ps(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sqrt(Vec<double> a) {
    return _mm256_sqrt_pd(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> min(Vec<float> a, Vec<float> b) {
    return _mm256_min_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> min(Vec<double> a, Vec<double> b) {
    return _mm256_min_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> max(Vec<float> a, Vec<float> b) {
    return _mm256_max_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> max(Vec<double> a, Vec<double> b) {
    return _mm256_max_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> abs(Vec<float> a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> abs(Vec<double> a) {
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> neg(Vec<float> a) {
    return _mm256_xor_ps(a.reg, _mm256_set1_ps(-0.0f));
}
template <>
EINSUMS_FORCEINLINE Vec<double> neg(Vec<double> a) {
    return _mm256_xor_pd(a.reg, _mm256_set1_pd(-0.0));
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> div(Vec<float> a, Vec<float> b) {
    return _mm_div_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> div(Vec<double> a, Vec<double> b) {
    return _mm_div_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sqrt(Vec<float> a) {
    return _mm_sqrt_ps(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sqrt(Vec<double> a) {
    return _mm_sqrt_pd(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> min(Vec<float> a, Vec<float> b) {
    return _mm_min_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> min(Vec<double> a, Vec<double> b) {
    return _mm_min_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> max(Vec<float> a, Vec<float> b) {
    return _mm_max_ps(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> max(Vec<double> a, Vec<double> b) {
    return _mm_max_pd(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> abs(Vec<float> a) {
    return _mm_andnot_ps(_mm_set1_ps(-0.0f), a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> abs(Vec<double> a) {
    return _mm_andnot_pd(_mm_set1_pd(-0.0), a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> neg(Vec<float> a) {
    return _mm_xor_ps(a.reg, _mm_set1_ps(-0.0f));
}
template <>
EINSUMS_FORCEINLINE Vec<double> neg(Vec<double> a) {
    return _mm_xor_pd(a.reg, _mm_set1_pd(-0.0));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> div(Vec<float> a, Vec<float> b) {
    return vdivq_f32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> div(Vec<double> a, Vec<double> b) {
    return vdivq_f64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> sqrt(Vec<float> a) {
    return vsqrtq_f32(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> sqrt(Vec<double> a) {
    return vsqrtq_f64(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> min(Vec<float> a, Vec<float> b) {
    return vbslq_f32(vcltq_f32(a.reg, b.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> min(Vec<double> a, Vec<double> b) {
    return vbslq_f64(vcltq_f64(a.reg, b.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> max(Vec<float> a, Vec<float> b) {
    return vbslq_f32(vcgtq_f32(a.reg, b.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> max(Vec<double> a, Vec<double> b) {
    return vbslq_f64(vcgtq_f64(a.reg, b.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> abs(Vec<float> a) {
    return vabsq_f32(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> abs(Vec<double> a) {
    return vabsq_f64(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> neg(Vec<float> a) {
    return vnegq_f32(a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> neg(Vec<double> a) {
    return vnegq_f64(a.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> div(Vec<float> a, Vec<float> b) {
    return {a.reg / b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> div(Vec<double> a, Vec<double> b) {
    return {a.reg / b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<float> sqrt(Vec<float> a) {
    return {std::sqrt(a.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<double> sqrt(Vec<double> a) {
    return {std::sqrt(a.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<float> min(Vec<float> a, Vec<float> b) {
    return {a.reg < b.reg ? a.reg : b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> min(Vec<double> a, Vec<double> b) {
    return {a.reg < b.reg ? a.reg : b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<float> max(Vec<float> a, Vec<float> b) {
    return {a.reg > b.reg ? a.reg : b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> max(Vec<double> a, Vec<double> b) {
    return {a.reg > b.reg ? a.reg : b.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<float> abs(Vec<float> a) {
    return {std::fabs(a.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<double> abs(Vec<double> a) {
    return {std::fabs(a.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<float> neg(Vec<float> a) {
    return {-a.reg};
}
template <>
EINSUMS_FORCEINLINE Vec<double> neg(Vec<double> a) {
    return {-a.reg};
}
#endif

// ===========================================================================
// The other fused forms, with the x86 names and meanings:
//
//   fmsub(a, b, c)  =  a * b - c
//   fnmadd(a, b, c) = -(a * b) + c
//   fnmsub(a, b, c) = -(a * b) - c
//
// Each rounds once where fmadd does, and where fmadd falls back to a separate
// multiply and add (has_fma is false on x86), these do too. NEON has only
// c + a*b and c - a*b, so a negated c supplies the other two; negation is
// exact, so the result is still the single rounding x86 gives, signed zeros
// included.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> fmsub(Vec<T> a, Vec<T> b, Vec<T> c);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> fnmadd(Vec<T> a, Vec<T> b, Vec<T> c);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> fnmsub(Vec<T> a, Vec<T> b, Vec<T> c);

#if defined(EINSUMS_SIMD_HAVE_FMA)
#    define EINSUMS_SIMD_FMA_FORMS(T, FMSUB, FNMADD, FNMSUB)                                                                               \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> fmsub(Vec<T> a, Vec<T> b, Vec<T> c) {                                                                   \
            return FMSUB(a.reg, b.reg, c.reg);                                                                                             \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> fnmadd(Vec<T> a, Vec<T> b, Vec<T> c) {                                                                  \
            return FNMADD(a.reg, b.reg, c.reg);                                                                                            \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> fnmsub(Vec<T> a, Vec<T> b, Vec<T> c) {                                                                  \
            return FNMSUB(a.reg, b.reg, c.reg);                                                                                            \
        }
#    if defined(__AVX512F__) && defined(__AVX512VL__)
EINSUMS_SIMD_FMA_FORMS(float, _mm512_fmsub_ps, _mm512_fnmadd_ps, _mm512_fnmsub_ps)
EINSUMS_SIMD_FMA_FORMS(double, _mm512_fmsub_pd, _mm512_fnmadd_pd, _mm512_fnmsub_pd)
#    elif defined(__AVX__)
EINSUMS_SIMD_FMA_FORMS(float, _mm256_fmsub_ps, _mm256_fnmadd_ps, _mm256_fnmsub_ps)
EINSUMS_SIMD_FMA_FORMS(double, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_fnmsub_pd)
#    else
EINSUMS_SIMD_FMA_FORMS(float, _mm_fmsub_ps, _mm_fnmadd_ps, _mm_fnmsub_ps)
EINSUMS_SIMD_FMA_FORMS(double, _mm_fmsub_pd, _mm_fnmadd_pd, _mm_fnmsub_pd)
#    endif
#    undef EINSUMS_SIMD_FMA_FORMS
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> fmsub(Vec<float> a, Vec<float> b, Vec<float> c) {
    return vfmaq_f32(vnegq_f32(c.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmsub(Vec<double> a, Vec<double> b, Vec<double> c) {
    return vfmaq_f64(vnegq_f64(c.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> fnmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return vfmsq_f32(c.reg, a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fnmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return vfmsq_f64(c.reg, a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> fnmsub(Vec<float> a, Vec<float> b, Vec<float> c) {
    return vfmsq_f32(vnegq_f32(c.reg), a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fnmsub(Vec<double> a, Vec<double> b, Vec<double> c) {
    return vfmsq_f64(vnegq_f64(c.reg), a.reg, b.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> fmsub(Vec<float> a, Vec<float> b, Vec<float> c) {
    return sub(mul(a, b), c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fmsub(Vec<double> a, Vec<double> b, Vec<double> c) {
    return sub(mul(a, b), c);
}
template <>
EINSUMS_FORCEINLINE Vec<float> fnmadd(Vec<float> a, Vec<float> b, Vec<float> c) {
    return sub(c, mul(a, b));
}
template <>
EINSUMS_FORCEINLINE Vec<double> fnmadd(Vec<double> a, Vec<double> b, Vec<double> c) {
    return sub(c, mul(a, b));
}
template <>
EINSUMS_FORCEINLINE Vec<float> fnmsub(Vec<float> a, Vec<float> b, Vec<float> c) {
    return sub(neg(mul(a, b)), c);
}
template <>
EINSUMS_FORCEINLINE Vec<double> fnmsub(Vec<double> a, Vec<double> b, Vec<double> c) {
    return sub(neg(mul(a, b)), c);
}
#endif

// ===========================================================================
// Operator overloads (opt-out via EINSUMS_SIMD_NO_OPERATORS)
//
// + - * / between two Vec<T>, or a Vec<T> and a scalar on either side, and the
// compound assignments += -= *= /= with either on the right. The scalar must be
// exactly T or an integer, which converts to T: v * 2 compiles, but
// Vec<float> * 2.0 does not. That is deliberate. In a kernel written once for
// Vec<float> and for float, float * 2.0 computes in double, so the two would
// round differently; the error in the vector instantiation catches it. Write
// typed constants, T(2.0).
// ===========================================================================

#if !defined(EINSUMS_SIMD_NO_OPERATORS)
namespace detail {
/// A scalar that may stand beside a Vec<T> in an operator: T itself, or any integer but bool.
template <typename S, typename T>
concept scalar_operand_for = std::same_as<S, T> || (std::integral<S> && !std::same_as<S, bool>);
} // namespace detail

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
template <typename T>
EINSUMS_FORCEINLINE Vec<T> operator/(Vec<T> a, Vec<T> b) {
    return div(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Vec<T> operator-(Vec<T> a) {
    return neg(a);
}

#    define EINSUMS_SIMD_MIXED_OPERATOR(op, fn)                                                                                            \
        template <typename T, detail::scalar_operand_for<T> S>                                                                             \
        EINSUMS_FORCEINLINE Vec<T> operator op(Vec<T> a, S b) {                                                                            \
            return fn(a, broadcast(static_cast<T>(b)));                                                                                    \
        }                                                                                                                                  \
        template <typename T, detail::scalar_operand_for<T> S>                                                                             \
        EINSUMS_FORCEINLINE Vec<T> operator op(S a, Vec<T> b) {                                                                            \
            return fn(broadcast(static_cast<T>(a)), b);                                                                                    \
        }                                                                                                                                  \
        template <typename T>                                                                                                              \
        EINSUMS_FORCEINLINE Vec<T> &operator op## = (Vec<T> & a, Vec<T> b) {                                                               \
            return a = fn(a, b);                                                                                                           \
        }                                                                                                                                  \
        template <typename T, detail::scalar_operand_for<T> S>                                                                             \
        EINSUMS_FORCEINLINE Vec<T> &operator op## = (Vec<T> & a, S b) {                                                                    \
            return a = fn(a, broadcast(static_cast<T>(b)));                                                                                \
        }
EINSUMS_SIMD_MIXED_OPERATOR(+, add)
EINSUMS_SIMD_MIXED_OPERATOR(-, sub)
EINSUMS_SIMD_MIXED_OPERATOR(*, mul)
EINSUMS_SIMD_MIXED_OPERATOR(/, div)
#    undef EINSUMS_SIMD_MIXED_OPERATOR
#endif

// ===========================================================================
// Integer SIMD operations: Vec<int32_t>, Vec<uint32_t>, Vec<int64_t>,
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
// Intentionally not provided here:
//   stream_store on integers: no observed user; add when needed.
//   i64 multiply on SSE2: no native instruction; emulation costs more
//   than scalar fallback for typical kernels. Add later if a user needs it.
//   i8 / i16 vectors: not in scope; saturation semantics double the surface.
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
// Scalar fallback: Vec<T>::reg is a single T.
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
// Integer arithmetic: add / sub / mul.
//
// `add` and `sub` are trivial across every ISA tier and signedness because
// two's-complement wraparound is bit-identical for signed and unsigned, so
// the same instruction handles both.
//
// `mul` returns the low half of the elementwise product (matching every
// SIMD ISA's "low multiply" semantics; full 32×32→64 multiplies are a
// different operation we skip here).
//
// Coverage caveats for instructions that don't exist on the tier:
//   - SSE2 has no 32-bit integer multiply (added in SSE4.1 as PMULLD); we
//     skip Vec<int32_t>/Vec<uint32_t> mul on SSE2 unless SSE4.1 is on.
//   - SSE2/AVX/AVX2 have no native 64-bit element multiply. AVX-512DQ adds
//     VPMULLQ; without DQ we skip i64/u64 mul on those tiers.
//   - aarch64 NEON has no 64-bit integer vector multiply (vmulq_s64 only
//     exists under SVE2). We skip i64/u64 mul on NEON.
// Calls to a missing specialization produce a link error referencing the
// primary template, a clear signal to the user that the op needs a wider
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
// 256-bit integer arithmetic requires AVX2; AVX1 only had float ops.
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
// i64/u64 mul not implemented; needs AVX-512DQ.
#elif defined(__AVX__)
// AVX without AVX2 has no 256-bit integer instructions, and the integer Vecs
// are __m256i here, so no SSE2 form fits them: these operations are left
// undefined for that tier, and a call is a link error.
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
// i64/u64 mul not implemented; needs AVX-512DQ.
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
// i64/u64 mul not implemented; aarch64 NEON has no 64-bit element mul.
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
// Bitwise: and / or / xor.
//
// Single intrinsic per ISA per width, so no element-type dispatch is needed
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
#elif defined(__AVX__)
// AVX without AVX2 has no 256-bit integer instructions, and the integer Vecs
// are __m256i here, so no SSE2 form fits them: these operations are left
// undefined for that tier, and a call is a link error.
#    define EINSUMS_SIMD_INT_BITWISE(T) /* nothing; see above */
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
// NEON: vand/vorr/veor, typed by signedness and width but all alias the same hw op.
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
#    define EINSUMS_SIMD_INT_BITWISE(T) /* nothing; NEON expanded above per type */
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
// Logical shifts: shift_left<N>(v) and shift_right<N>(v).
//
// The shift count is a non-type template parameter so the intrinsics
// generated are immediate-form (best codegen). The variable-count forms,
// one count per lane, follow this section.
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
#elif defined(__AVX__)
// AVX without AVX2 has no 256-bit integer instructions, and the integer Vecs
// are __m256i here, so no SSE2 form fits them: these operations are left
// undefined for that tier, and a call is a link error.
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
// Variable logical shifts: shift_left(v, count) and shift_right(v, count)
// shift each lane of v by the matching lane of count, on the 32- and 64-bit
// integers. shift_right is logical, as the immediate form is.
//
// Each count must be in [0, bits of the element); outside that the lane is
// unspecified, because x86 gives zero and NEON reads the count's low byte as a
// signed amount.
//
// AVX2 and AVX-512 shift per lane natively, and NEON shifts by a signed
// per-lane amount, negated for a right shift. SSE has only a shift of every
// lane by one amount: 64-bit lanes take two such shifts and keep a lane of
// each, and 32-bit lanes go through memory.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count);

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_X86_VAR_SHIFTS(T, SLLV, SRLV)                                                                                     \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count) {                                                                    \
            return SLLV(v.reg, count.reg);                                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count) {                                                                   \
            return SRLV(v.reg, count.reg);                                                                                                 \
        }
EINSUMS_SIMD_X86_VAR_SHIFTS(int32_t, _mm512_sllv_epi32, _mm512_srlv_epi32)
EINSUMS_SIMD_X86_VAR_SHIFTS(uint32_t, _mm512_sllv_epi32, _mm512_srlv_epi32)
EINSUMS_SIMD_X86_VAR_SHIFTS(int64_t, _mm512_sllv_epi64, _mm512_srlv_epi64)
EINSUMS_SIMD_X86_VAR_SHIFTS(uint64_t, _mm512_sllv_epi64, _mm512_srlv_epi64)
#    undef EINSUMS_SIMD_X86_VAR_SHIFTS
#elif defined(__AVX2__)
#    define EINSUMS_SIMD_X86_VAR_SHIFTS(T, SLLV, SRLV)                                                                                     \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count) {                                                                    \
            return SLLV(v.reg, count.reg);                                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count) {                                                                   \
            return SRLV(v.reg, count.reg);                                                                                                 \
        }
EINSUMS_SIMD_X86_VAR_SHIFTS(int32_t, _mm256_sllv_epi32, _mm256_srlv_epi32)
EINSUMS_SIMD_X86_VAR_SHIFTS(uint32_t, _mm256_sllv_epi32, _mm256_srlv_epi32)
EINSUMS_SIMD_X86_VAR_SHIFTS(int64_t, _mm256_sllv_epi64, _mm256_srlv_epi64)
EINSUMS_SIMD_X86_VAR_SHIFTS(uint64_t, _mm256_sllv_epi64, _mm256_srlv_epi64)
#    undef EINSUMS_SIMD_X86_VAR_SHIFTS
#elif defined(__AVX__)
// No 256-bit integer instructions; see the comment above.
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
namespace detail {
template <typename T, typename Shift>
EINSUMS_FORCEINLINE Vec<T> shift_lanes(Vec<T> v, Vec<T> count, Shift shift) {
    using U = std::make_unsigned_t<T>;
    alignas(native_alignment) T values[Vec<T>::lanes];
    alignas(native_alignment) T counts[Vec<T>::lanes];
    storea(values, v);
    storea(counts, count);
    for (int i = 0; i < Vec<T>::lanes; ++i) {
        // A C++ shift by the width or more is undefined; give the zero PSLLD would.
        auto const n = static_cast<unsigned>(counts[i]);
        values[i]    = n < 8 * sizeof(T) ? static_cast<T>(shift(static_cast<U>(values[i]), n)) : T{0};
    }
    return loada(values);
}

/// PSLLQ and PSRLQ shift both lanes by the low 64 bits of their count, so shift twice and take lane
/// 0 of the first and lane 1 of the second.
template <typename Shift>
EINSUMS_FORCEINLINE __m128i shift_two_lanes(__m128i v, __m128i count, Shift shift) {
    __m128i const low  = shift(v, count);
    __m128i const high = shift(v, _mm_unpackhi_epi64(count, count));
    return _mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(high), _mm_castsi128_pd(low)));
}
} // namespace detail

#    define EINSUMS_SIMD_SSE_VAR_SHIFTS_32(T)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count) {                                                                    \
            return detail::shift_lanes(v, count, [](auto x, unsigned n) { return x << n; });                                               \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count) {                                                                   \
            return detail::shift_lanes(v, count, [](auto x, unsigned n) { return x >> n; });                                               \
        }
#    define EINSUMS_SIMD_SSE_VAR_SHIFTS_64(T)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count) {                                                                    \
            return detail::shift_two_lanes(v.reg, count.reg, [](__m128i x, __m128i n) { return _mm_sll_epi64(x, n); });                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count) {                                                                   \
            return detail::shift_two_lanes(v.reg, count.reg, [](__m128i x, __m128i n) { return _mm_srl_epi64(x, n); });                    \
        }
EINSUMS_SIMD_SSE_VAR_SHIFTS_32(int32_t)
EINSUMS_SIMD_SSE_VAR_SHIFTS_32(uint32_t)
EINSUMS_SIMD_SSE_VAR_SHIFTS_64(int64_t)
EINSUMS_SIMD_SSE_VAR_SHIFTS_64(uint64_t)
#    undef EINSUMS_SIMD_SSE_VAR_SHIFTS_32
#    undef EINSUMS_SIMD_SSE_VAR_SHIFTS_64
#elif defined(__aarch64__) || defined(_M_ARM64)
// VSHL shifts left by a signed per-lane amount and right by a negative one. Its amount is always a
// signed vector, so the unsigned counts are reinterpreted, and the logical right shift of a signed
// lane goes through the unsigned type.
template <>
EINSUMS_FORCEINLINE Vec<int32_t> shift_left(Vec<int32_t> v, Vec<int32_t> count) {
    return vshlq_s32(v.reg, count.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_left(Vec<uint32_t> v, Vec<uint32_t> count) {
    return vshlq_u32(v.reg, vreinterpretq_s32_u32(count.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> shift_left(Vec<int64_t> v, Vec<int64_t> count) {
    return vshlq_s64(v.reg, count.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_left(Vec<uint64_t> v, Vec<uint64_t> count) {
    return vshlq_u64(v.reg, vreinterpretq_s64_u64(count.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> shift_right(Vec<int32_t> v, Vec<int32_t> count) {
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v.reg), vnegq_s32(count.reg)));
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> shift_right(Vec<uint32_t> v, Vec<uint32_t> count) {
    return vshlq_u32(v.reg, vnegq_s32(vreinterpretq_s32_u32(count.reg)));
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> shift_right(Vec<int64_t> v, Vec<int64_t> count) {
    return vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(v.reg), vnegq_s64(count.reg)));
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> shift_right(Vec<uint64_t> v, Vec<uint64_t> count) {
    return vshlq_u64(v.reg, vnegq_s64(vreinterpretq_s64_u64(count.reg)));
}
#else
#    define EINSUMS_SIMD_SCALAR_VAR_SHIFTS(T)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_left(Vec<T> v, Vec<T> count) {                                                                    \
            auto const n = static_cast<std::make_unsigned_t<T>>(count.reg);                                                                \
            return {n < 8 * sizeof(T) ? static_cast<T>(static_cast<std::make_unsigned_t<T>>(v.reg) << n) : T{0}};                          \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> shift_right(Vec<T> v, Vec<T> count) {                                                                   \
            auto const n = static_cast<std::make_unsigned_t<T>>(count.reg);                                                                \
            return {n < 8 * sizeof(T) ? static_cast<T>(static_cast<std::make_unsigned_t<T>>(v.reg) >> n) : T{0}};                          \
        }
EINSUMS_SIMD_SCALAR_VAR_SHIFTS(int32_t)
EINSUMS_SIMD_SCALAR_VAR_SHIFTS(uint32_t)
EINSUMS_SIMD_SCALAR_VAR_SHIFTS(int64_t)
EINSUMS_SIMD_SCALAR_VAR_SHIFTS(uint64_t)
#    undef EINSUMS_SIMD_SCALAR_VAR_SHIFTS
#endif

// ===========================================================================
// Masks: Mask<T>, the comparisons, select, and mask logic.
//
// A Mask<T> holds one flag per lane of a Vec<T>, for float, double and the 32-
// and 64-bit integers. It is its own type, held the way the hardware holds it:
// a k-register on AVX-512, a vector of all-ones or zero lanes on AVX, SSE and
// NEON, and a bool in the scalar build.
//
//   cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge
//                          Mask<T> from two Vec<T>. Floating-point comparisons
//                          follow IEEE 754, so every one with a NaN is false but
//                          cmp_ne; integers compare in their own signedness.
//   select(m, a, b)        a where m is set, b elsewhere.
//   m & n, m | n, m ^ n, !m, and &=, |=, ^=; also the names bitwise_and,
//                          bitwise_or, bitwise_xor and bitwise_andnot (m & !n).
//   any, all, none, count  whether any, every or no lane is set; how many are.
//   first_n<T>(n)          lanes 0 .. n-1 set, a loop tail's mask; n of the
//                          lane count or more sets every lane.
//   mask_all<T>(), mask_none<T>()
//   to_bits(m), mask_from_bits<T>(bits)
//                          bit i of the integer is lane i.
//   mask_cast<U>(m)        the same lanes as a Mask<U>, for U of T's width.
//   to_vec(m)              a Vec<T> of all-ones or zero bits per lane, for code
//                          that uses a mask as bits. The numeric form, one or
//                          zero, is select(m, broadcast(T(1)), broadcast(T(0))).
//
// Write !m, not ~m: generic kernels also instantiate with a bool mask, and
// ~true is -2, which is still true.
//
// AVX without AVX2 has no 256-bit integer instructions, so there the integer
// comparisons are left undefined, as every integer operation is; select and
// the mask logic go through the floating-point domain and work for every type.
// ===========================================================================

namespace detail {
/// The element types that have a Mask.
template <typename T>
inline constexpr bool has_mask = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t> ||
                                 std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_MASK_IS_K 1
template <typename T>
using mask_reg_t = std::conditional_t<sizeof(T) == 4, __mmask16, __mmask8>;
#elif defined(__AVX__) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define EINSUMS_SIMD_MASK_IS_VECTOR 1
template <typename T>
using mask_reg_t = typename VecTraits<T>::reg_type;
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define EINSUMS_SIMD_MASK_IS_VECTOR 1
template <typename T>
using mask_reg_t = std::conditional_t<sizeof(T) == 4, uint32x4_t, uint64x2_t>;
#else
#    define EINSUMS_SIMD_MASK_IS_BOOL 1
template <typename T>
using mask_reg_t = bool;
#endif
} // namespace detail

/// One flag per lane of a Vec<T>.
template <typename T>
struct Mask {
    static_assert(detail::has_mask<T>, "Mask exists for float, double and the 32- and 64-bit integers");

    using reg_type   = detail::mask_reg_t<T>;
    using value_type = T;

    static constexpr int lanes = Vec<T>::lanes;

    reg_type reg;

    Mask() = default;

    EINSUMS_FORCEINLINE explicit Mask(reg_type r) : reg(r) {}
};

template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_eq(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_ne(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_lt(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_le(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_gt(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Mask<T> cmp_ge(Vec<T> a, Vec<T> b);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> select(Mask<T> mask, Vec<T> a, Vec<T> b);
/// a & ~b on the bits of two vectors (the mask form, m & !n, is below).
template <typename T>
EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b);

// ---------------------------------------------------------------------------
// Mask logic on the register, one overload set per representation.
// ---------------------------------------------------------------------------

namespace detail {
#if defined(EINSUMS_SIMD_MASK_IS_K)
template <std::unsigned_integral K>
EINSUMS_FORCEINLINE K mask_and(K a, K b) {
    return static_cast<K>(a & b);
}
template <std::unsigned_integral K>
EINSUMS_FORCEINLINE K mask_or(K a, K b) {
    return static_cast<K>(a | b);
}
template <std::unsigned_integral K>
EINSUMS_FORCEINLINE K mask_xor(K a, K b) {
    return static_cast<K>(a ^ b);
}
template <std::unsigned_integral K>
EINSUMS_FORCEINLINE K mask_andnot(K a, K b) {
    return static_cast<K>(a & ~b);
}
#elif defined(__AVX__)
// The integer forms go through the floating-point domain, so AVX without AVX2 has them too.
EINSUMS_FORCEINLINE __m256 mask_and(__m256 a, __m256 b) {
    return _mm256_and_ps(a, b);
}
EINSUMS_FORCEINLINE __m256 mask_or(__m256 a, __m256 b) {
    return _mm256_or_ps(a, b);
}
EINSUMS_FORCEINLINE __m256 mask_xor(__m256 a, __m256 b) {
    return _mm256_xor_ps(a, b);
}
EINSUMS_FORCEINLINE __m256 mask_andnot(__m256 a, __m256 b) {
    return _mm256_andnot_ps(b, a);
}
EINSUMS_FORCEINLINE __m256d mask_and(__m256d a, __m256d b) {
    return _mm256_and_pd(a, b);
}
EINSUMS_FORCEINLINE __m256d mask_or(__m256d a, __m256d b) {
    return _mm256_or_pd(a, b);
}
EINSUMS_FORCEINLINE __m256d mask_xor(__m256d a, __m256d b) {
    return _mm256_xor_pd(a, b);
}
EINSUMS_FORCEINLINE __m256d mask_andnot(__m256d a, __m256d b) {
    return _mm256_andnot_pd(b, a);
}
EINSUMS_FORCEINLINE __m256i mask_and(__m256i a, __m256i b) {
    return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
EINSUMS_FORCEINLINE __m256i mask_or(__m256i a, __m256i b) {
    return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
EINSUMS_FORCEINLINE __m256i mask_xor(__m256i a, __m256i b) {
    return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
EINSUMS_FORCEINLINE __m256i mask_andnot(__m256i a, __m256i b) {
    return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(b), _mm256_castsi256_ps(a)));
}
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR) && !(defined(__aarch64__) || defined(_M_ARM64))
EINSUMS_FORCEINLINE __m128 mask_and(__m128 a, __m128 b) {
    return _mm_and_ps(a, b);
}
EINSUMS_FORCEINLINE __m128 mask_or(__m128 a, __m128 b) {
    return _mm_or_ps(a, b);
}
EINSUMS_FORCEINLINE __m128 mask_xor(__m128 a, __m128 b) {
    return _mm_xor_ps(a, b);
}
EINSUMS_FORCEINLINE __m128 mask_andnot(__m128 a, __m128 b) {
    return _mm_andnot_ps(b, a);
}
EINSUMS_FORCEINLINE __m128d mask_and(__m128d a, __m128d b) {
    return _mm_and_pd(a, b);
}
EINSUMS_FORCEINLINE __m128d mask_or(__m128d a, __m128d b) {
    return _mm_or_pd(a, b);
}
EINSUMS_FORCEINLINE __m128d mask_xor(__m128d a, __m128d b) {
    return _mm_xor_pd(a, b);
}
EINSUMS_FORCEINLINE __m128d mask_andnot(__m128d a, __m128d b) {
    return _mm_andnot_pd(b, a);
}
EINSUMS_FORCEINLINE __m128i mask_and(__m128i a, __m128i b) {
    return _mm_and_si128(a, b);
}
EINSUMS_FORCEINLINE __m128i mask_or(__m128i a, __m128i b) {
    return _mm_or_si128(a, b);
}
EINSUMS_FORCEINLINE __m128i mask_xor(__m128i a, __m128i b) {
    return _mm_xor_si128(a, b);
}
EINSUMS_FORCEINLINE __m128i mask_andnot(__m128i a, __m128i b) {
    return _mm_andnot_si128(b, a);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
EINSUMS_FORCEINLINE uint32x4_t mask_and(uint32x4_t a, uint32x4_t b) {
    return vandq_u32(a, b);
}
EINSUMS_FORCEINLINE uint32x4_t mask_or(uint32x4_t a, uint32x4_t b) {
    return vorrq_u32(a, b);
}
EINSUMS_FORCEINLINE uint32x4_t mask_xor(uint32x4_t a, uint32x4_t b) {
    return veorq_u32(a, b);
}
EINSUMS_FORCEINLINE uint32x4_t mask_andnot(uint32x4_t a, uint32x4_t b) {
    return vbicq_u32(a, b);
}
EINSUMS_FORCEINLINE uint64x2_t mask_and(uint64x2_t a, uint64x2_t b) {
    return vandq_u64(a, b);
}
EINSUMS_FORCEINLINE uint64x2_t mask_or(uint64x2_t a, uint64x2_t b) {
    return vorrq_u64(a, b);
}
EINSUMS_FORCEINLINE uint64x2_t mask_xor(uint64x2_t a, uint64x2_t b) {
    return veorq_u64(a, b);
}
EINSUMS_FORCEINLINE uint64x2_t mask_andnot(uint64x2_t a, uint64x2_t b) {
    return vbicq_u64(a, b);
}
#else
EINSUMS_FORCEINLINE bool mask_and(bool a, bool b) {
    return a && b;
}
EINSUMS_FORCEINLINE bool mask_or(bool a, bool b) {
    return a || b;
}
EINSUMS_FORCEINLINE bool mask_xor(bool a, bool b) {
    return a != b;
}
EINSUMS_FORCEINLINE bool mask_andnot(bool a, bool b) {
    return a && !b;
}
#endif

/// Thirty-two words, sixteen all-ones then sixteen zeros. A register's worth of words starting
/// 16 - k words in is a vector mask whose first k words are set.
alignas(64) inline constexpr int32_t first_n_words[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
} // namespace detail

// ---------------------------------------------------------------------------
// Building, converting and inspecting masks; the same code for every ISA.
// ---------------------------------------------------------------------------

/// Lanes 0 .. n-1 set: the mask of a loop tail of n elements. n of the lane count or more sets every lane.
template <typename T>
EINSUMS_FORCEINLINE Mask<T> first_n(std::size_t n) {
    constexpr int     L = Vec<T>::lanes;
    std::size_t const k = n < static_cast<std::size_t>(L) ? n : static_cast<std::size_t>(L);
#if defined(EINSUMS_SIMD_MASK_IS_K)
    using K = typename Mask<T>::reg_type;
    return Mask<T>(k == static_cast<std::size_t>(L) ? static_cast<K>((1u << L) - 1u) : static_cast<K>((1u << k) - 1u));
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    using R                  = typename Mask<T>::reg_type;
    constexpr std::size_t wl = sizeof(T) / sizeof(int32_t); // words per lane
    R                     r;
    std::memcpy(&r, detail::first_n_words + 16 - k * wl, sizeof(R));
    return Mask<T>(r);
#else
    return Mask<T>(k > 0);
#endif
}

template <typename T>
EINSUMS_FORCEINLINE Mask<T> mask_all() {
    return first_n<T>(static_cast<std::size_t>(Vec<T>::lanes));
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> mask_none() {
    return first_n<T>(0);
}

/// The lanes of m as an integer, lane i in bit i.
template <typename T>
EINSUMS_FORCEINLINE uint64_t to_bits(Mask<T> m) {
#if defined(EINSUMS_SIMD_MASK_IS_K)
    return m.reg;
#elif defined(__AVX__)
    if constexpr (sizeof(T) == 4) {
        return static_cast<uint64_t>(_mm256_movemask_ps(std::bit_cast<__m256>(m.reg)));
    } else {
        return static_cast<uint64_t>(_mm256_movemask_pd(std::bit_cast<__m256d>(m.reg)));
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if constexpr (sizeof(T) == 4) {
        uint32x4_t const lane_bits = {1u, 2u, 4u, 8u};
        return vaddvq_u32(vandq_u32(m.reg, lane_bits));
    } else {
        uint64x2_t const lane_bits = {1u, 2u};
        return vaddvq_u64(vandq_u64(m.reg, lane_bits));
    }
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    if constexpr (sizeof(T) == 4) {
        return static_cast<uint64_t>(_mm_movemask_ps(std::bit_cast<__m128>(m.reg)));
    } else {
        return static_cast<uint64_t>(_mm_movemask_pd(std::bit_cast<__m128d>(m.reg)));
    }
#else
    return m.reg ? 1u : 0u;
#endif
}

/// The mask whose lane i is bit i of @p bits; bits past the lane count are ignored.
template <typename T>
EINSUMS_FORCEINLINE Mask<T> mask_from_bits(uint64_t bits) {
    [[maybe_unused]] constexpr int L = Vec<T>::lanes;
#if defined(EINSUMS_SIMD_MASK_IS_K)
    using K = typename Mask<T>::reg_type;
    return Mask<T>(static_cast<K>(bits & ((1u << L) - 1u)));
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    using W = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
    W lanes_bits[L];
    for (int i = 0; i < L; ++i) {
        lanes_bits[i] = ((bits >> i) & 1u) ? W{-1} : W{0};
    }
    typename Mask<T>::reg_type r;
    std::memcpy(&r, lanes_bits, sizeof(r));
    return Mask<T>(r);
#else
    return Mask<T>((bits & 1u) != 0);
#endif
}

/// The same lanes as a mask for U, an element type of T's width (float and int32_t, double and int64_t).
template <typename U, typename T>
    requires(sizeof(U) == sizeof(T))
EINSUMS_FORCEINLINE Mask<U> mask_cast(Mask<T> m) {
    return Mask<U>(std::bit_cast<typename Mask<U>::reg_type>(m.reg));
}

/// m as a vector whose lanes are all-ones or zero bits.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> to_vec(Mask<T> m) {
#if defined(EINSUMS_SIMD_MASK_IS_K)
    if constexpr (sizeof(T) == 4) {
        return std::bit_cast<typename Vec<T>::reg_type>(_mm512_maskz_set1_epi32(m.reg, -1));
    } else {
        return std::bit_cast<typename Vec<T>::reg_type>(_mm512_maskz_set1_epi64(m.reg, -1));
    }
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    return std::bit_cast<typename Vec<T>::reg_type>(m.reg);
#else
    using W = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    return Vec<T>(std::bit_cast<T>(m.reg ? ~W{0} : W{0}));
#endif
}

template <typename T>
EINSUMS_FORCEINLINE bool any(Mask<T> m) {
#if defined(EINSUMS_SIMD_MASK_IS_K)
    return m.reg != 0;
#elif defined(__aarch64__) || defined(_M_ARM64)
    // Every lane is all-ones or zero, so the 32-bit view answers for 64-bit lanes too.
    if constexpr (sizeof(T) == 4) {
        return vmaxvq_u32(m.reg) != 0;
    } else {
        return vmaxvq_u32(vreinterpretq_u32_u64(m.reg)) != 0;
    }
#elif defined(EINSUMS_SIMD_MASK_IS_VECTOR)
    return to_bits(m) != 0;
#else
    return m.reg;
#endif
}

template <typename T>
EINSUMS_FORCEINLINE bool all(Mask<T> m) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if constexpr (sizeof(T) == 4) {
        return vminvq_u32(m.reg) != 0;
    } else {
        return vminvq_u32(vreinterpretq_u32_u64(m.reg)) != 0;
    }
#elif defined(EINSUMS_SIMD_MASK_IS_BOOL)
    return m.reg;
#else
    return to_bits(m) == (uint64_t{1} << Vec<T>::lanes) - 1u;
#endif
}

template <typename T>
EINSUMS_FORCEINLINE bool none(Mask<T> m) {
    return !any(m);
}

/// The number of lanes set.
template <typename T>
EINSUMS_FORCEINLINE int count(Mask<T> m) {
    return std::popcount(to_bits(m));
}

template <typename T>
EINSUMS_FORCEINLINE Mask<T> bitwise_and(Mask<T> a, Mask<T> b) {
    return Mask<T>(detail::mask_and(a.reg, b.reg));
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> bitwise_or(Mask<T> a, Mask<T> b) {
    return Mask<T>(detail::mask_or(a.reg, b.reg));
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> bitwise_xor(Mask<T> a, Mask<T> b) {
    return Mask<T>(detail::mask_xor(a.reg, b.reg));
}
/// a & !b.
template <typename T>
EINSUMS_FORCEINLINE Mask<T> bitwise_andnot(Mask<T> a, Mask<T> b) {
    return Mask<T>(detail::mask_andnot(a.reg, b.reg));
}

template <typename T>
EINSUMS_FORCEINLINE Mask<T> operator&(Mask<T> a, Mask<T> b) {
    return bitwise_and(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> operator|(Mask<T> a, Mask<T> b) {
    return bitwise_or(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> operator^(Mask<T> a, Mask<T> b) {
    return bitwise_xor(a, b);
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> operator!(Mask<T> a) {
    return bitwise_andnot(mask_all<T>(), a);
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> &operator&=(Mask<T> &a, Mask<T> b) {
    return a = a & b;
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> &operator|=(Mask<T> &a, Mask<T> b) {
    return a = a | b;
}
template <typename T>
EINSUMS_FORCEINLINE Mask<T> &operator^=(Mask<T> &a, Mask<T> b) {
    return a = a ^ b;
}

// ---------------------------------------------------------------------------
// Comparisons, select and the value-level bitwise operations, per ISA.
// ---------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_AVX512_FCMP(name, pred)                                                                                           \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<float> name(Vec<float> a, Vec<float> b) {                                                                 \
            return Mask<float>(_mm512_cmp_ps_mask(a.reg, b.reg, pred));                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<double> name(Vec<double> a, Vec<double> b) {                                                              \
            return Mask<double>(_mm512_cmp_pd_mask(a.reg, b.reg, pred));                                                                   \
        }
EINSUMS_SIMD_AVX512_FCMP(cmp_eq, _CMP_EQ_OQ)
EINSUMS_SIMD_AVX512_FCMP(cmp_ne, _CMP_NEQ_UQ)
EINSUMS_SIMD_AVX512_FCMP(cmp_lt, _CMP_LT_OQ)
EINSUMS_SIMD_AVX512_FCMP(cmp_le, _CMP_LE_OQ)
EINSUMS_SIMD_AVX512_FCMP(cmp_gt, _CMP_GT_OQ)
EINSUMS_SIMD_AVX512_FCMP(cmp_ge, _CMP_GE_OQ)
#    undef EINSUMS_SIMD_AVX512_FCMP

#    define EINSUMS_SIMD_AVX512_ICMP(T, CMP)                                                                                               \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_eq(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_EQ));                                                                              \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ne(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_NE));                                                                              \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_lt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_LT));                                                                              \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_le(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_LE));                                                                              \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_gt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_NLE));                                                                             \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ge(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(CMP(a.reg, b.reg, _MM_CMPINT_NLT));                                                                             \
        }
EINSUMS_SIMD_AVX512_ICMP(int32_t, _mm512_cmp_epi32_mask)
EINSUMS_SIMD_AVX512_ICMP(uint32_t, _mm512_cmp_epu32_mask)
EINSUMS_SIMD_AVX512_ICMP(int64_t, _mm512_cmp_epi64_mask)
EINSUMS_SIMD_AVX512_ICMP(uint64_t, _mm512_cmp_epu64_mask)
#    undef EINSUMS_SIMD_AVX512_ICMP

template <>
EINSUMS_FORCEINLINE Vec<float> select(Mask<float> m, Vec<float> a, Vec<float> b) {
    return _mm512_mask_blend_ps(m.reg, b.reg, a.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> select(Mask<double> m, Vec<double> a, Vec<double> b) {
    return _mm512_mask_blend_pd(m.reg, b.reg, a.reg);
}
#    define EINSUMS_SIMD_AVX512_SELECT_INT(T, W)                                                                                           \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                                 \
            return _mm512_mask_blend_epi##W(m.reg, b.reg, a.reg);                                                                          \
        }
EINSUMS_SIMD_AVX512_SELECT_INT(int32_t, 32)
EINSUMS_SIMD_AVX512_SELECT_INT(uint32_t, 32)
EINSUMS_SIMD_AVX512_SELECT_INT(int64_t, 64)
EINSUMS_SIMD_AVX512_SELECT_INT(uint64_t, 64)
#    undef EINSUMS_SIMD_AVX512_SELECT_INT

// The floating-point logic intrinsics are AVX-512DQ; the integer ones are F.
#    define EINSUMS_SIMD_AVX512_FLOAT_LOGIC(T, cast_to, cast_from)                                                                         \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return cast_from(_mm512_and_si512(cast_to(a.reg), cast_to(b.reg)));                                                            \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return cast_from(_mm512_or_si512(cast_to(a.reg), cast_to(b.reg)));                                                             \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return cast_from(_mm512_xor_si512(cast_to(a.reg), cast_to(b.reg)));                                                            \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return cast_from(_mm512_andnot_si512(cast_to(b.reg), cast_to(a.reg)));                                                         \
        }
EINSUMS_SIMD_AVX512_FLOAT_LOGIC(float, _mm512_castps_si512, _mm512_castsi512_ps)
EINSUMS_SIMD_AVX512_FLOAT_LOGIC(double, _mm512_castpd_si512, _mm512_castsi512_pd)
#    undef EINSUMS_SIMD_AVX512_FLOAT_LOGIC

#    define EINSUMS_SIMD_AVX512_ANDNOT_INT(T)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return _mm512_andnot_si512(b.reg, a.reg);                                                                                      \
        }
EINSUMS_SIMD_AVX512_ANDNOT_INT(int32_t)
EINSUMS_SIMD_AVX512_ANDNOT_INT(uint32_t)
EINSUMS_SIMD_AVX512_ANDNOT_INT(int64_t)
EINSUMS_SIMD_AVX512_ANDNOT_INT(uint64_t)
#    undef EINSUMS_SIMD_AVX512_ANDNOT_INT

#elif defined(__AVX__) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)

// ---- floating-point comparisons ----
#    if defined(__AVX__)
#        define EINSUMS_SIMD_X86_FCMP(name, pred, sse_ps, sse_pd)                                                                          \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<float> name(Vec<float> a, Vec<float> b) {                                                             \
                return Mask<float>(_mm256_cmp_ps(a.reg, b.reg, pred));                                                                     \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<double> name(Vec<double> a, Vec<double> b) {                                                          \
                return Mask<double>(_mm256_cmp_pd(a.reg, b.reg, pred));                                                                    \
            }
#    else
// CMPNEQPS is the unordered not-equal, so a NaN lane compares true there too.
#        define EINSUMS_SIMD_X86_FCMP(name, pred, sse_ps, sse_pd)                                                                          \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<float> name(Vec<float> a, Vec<float> b) {                                                             \
                return Mask<float>(sse_ps(a.reg, b.reg));                                                                                  \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<double> name(Vec<double> a, Vec<double> b) {                                                          \
                return Mask<double>(sse_pd(a.reg, b.reg));                                                                                 \
            }
#    endif
EINSUMS_SIMD_X86_FCMP(cmp_eq, _CMP_EQ_OQ, _mm_cmpeq_ps, _mm_cmpeq_pd)
EINSUMS_SIMD_X86_FCMP(cmp_ne, _CMP_NEQ_UQ, _mm_cmpneq_ps, _mm_cmpneq_pd)
EINSUMS_SIMD_X86_FCMP(cmp_lt, _CMP_LT_OQ, _mm_cmplt_ps, _mm_cmplt_pd)
EINSUMS_SIMD_X86_FCMP(cmp_le, _CMP_LE_OQ, _mm_cmple_ps, _mm_cmple_pd)
EINSUMS_SIMD_X86_FCMP(cmp_gt, _CMP_GT_OQ, _mm_cmpgt_ps, _mm_cmpgt_pd)
EINSUMS_SIMD_X86_FCMP(cmp_ge, _CMP_GE_OQ, _mm_cmpge_ps, _mm_cmpge_pd)
#    undef EINSUMS_SIMD_X86_FCMP

// ---- integer comparisons ----
//
// AVX2 and SSE have only equality and a signed greater-than, so lt swaps the
// operands, le and ge invert, and the unsigned forms flip each operand's top
// bit first, which maps unsigned order onto signed order. SSE2 without SSE4.1
// and SSE4.2 has no 64-bit equality or greater-than and builds them from the
// 32-bit halves.
#    if defined(__AVX2__) || !defined(__AVX__)
namespace detail {
#        if defined(__AVX2__)
EINSUMS_FORCEINLINE __m256i equal32(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi32(a, b);
}
EINSUMS_FORCEINLINE __m256i equal64(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi64(a, b);
}
EINSUMS_FORCEINLINE __m256i signed_gt32(__m256i a, __m256i b) {
    return _mm256_cmpgt_epi32(a, b);
}
EINSUMS_FORCEINLINE __m256i signed_gt64(__m256i a, __m256i b) {
    return _mm256_cmpgt_epi64(a, b);
}
#        else
EINSUMS_FORCEINLINE __m128i equal32(__m128i a, __m128i b) {
    return _mm_cmpeq_epi32(a, b);
}
#            if defined(__SSE4_1__)
// PCMPEQQ is SSE4.1.
EINSUMS_FORCEINLINE __m128i equal64(__m128i a, __m128i b) {
    return _mm_cmpeq_epi64(a, b);
}
#            else
// No PCMPEQQ: compare the 32-bit halves, then AND each 64-bit lane with its half-swapped self so a
// lane is all-ones only when both halves matched.
EINSUMS_FORCEINLINE __m128i equal64(__m128i a, __m128i b) {
    __m128i const t = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(t, _mm_shuffle_epi32(t, _MM_SHUFFLE(2, 3, 0, 1)));
}
#            endif
EINSUMS_FORCEINLINE __m128i signed_gt32(__m128i a, __m128i b) {
    return _mm_cmpgt_epi32(a, b);
}
#            if defined(__SSE4_2__)
EINSUMS_FORCEINLINE __m128i signed_gt64(__m128i a, __m128i b) {
    return _mm_cmpgt_epi64(a, b);
}
#            else
// A 64-bit lane is greater when its signed high half is, or when the high halves are equal and its
// low half is greater as an unsigned number. The answer lands in the high half and is copied down.
EINSUMS_FORCEINLINE __m128i signed_gt64(__m128i a, __m128i b) {
    __m128i const bias   = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
    __m128i const hi_gt  = _mm_cmpgt_epi32(a, b);
    __m128i const hi_eq  = _mm_cmpeq_epi32(a, b);
    __m128i const lo_gt  = _mm_cmpgt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias));
    __m128i const lo_up  = _mm_shuffle_epi32(lo_gt, _MM_SHUFFLE(2, 2, 0, 0));
    __m128i const result = _mm_or_si128(hi_gt, _mm_and_si128(hi_eq, lo_up));
    return _mm_shuffle_epi32(result, _MM_SHUFFLE(3, 3, 1, 1));
}
#            endif
#        endif

template <typename T>
EINSUMS_FORCEINLINE Mask<T> ordered_gt(Vec<T> a, Vec<T> b) {
    if constexpr (std::is_unsigned_v<T>) {
        Vec<T> const top = broadcast(static_cast<T>(T{1} << (8 * sizeof(T) - 1)));
        a                = bitwise_xor(a, top);
        b                = bitwise_xor(b, top);
    }
    if constexpr (sizeof(T) == 4) {
        return Mask<T>(signed_gt32(a.reg, b.reg));
    } else {
        return Mask<T>(signed_gt64(a.reg, b.reg));
    }
}

template <typename T>
EINSUMS_FORCEINLINE Mask<T> equal(Vec<T> a, Vec<T> b) {
    if constexpr (sizeof(T) == 4) {
        return Mask<T>(equal32(a.reg, b.reg));
    } else {
        return Mask<T>(equal64(a.reg, b.reg));
    }
}
} // namespace detail

#        define EINSUMS_SIMD_X86_ICMP(T)                                                                                                   \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_eq(Vec<T> a, Vec<T> b) {                                                                       \
                return detail::equal(a, b);                                                                                                \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_ne(Vec<T> a, Vec<T> b) {                                                                       \
                return !detail::equal(a, b);                                                                                               \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_lt(Vec<T> a, Vec<T> b) {                                                                       \
                return detail::ordered_gt(b, a);                                                                                           \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_le(Vec<T> a, Vec<T> b) {                                                                       \
                return !detail::ordered_gt(a, b);                                                                                          \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_gt(Vec<T> a, Vec<T> b) {                                                                       \
                return detail::ordered_gt(a, b);                                                                                           \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Mask<T> cmp_ge(Vec<T> a, Vec<T> b) {                                                                       \
                return !detail::ordered_gt(b, a);                                                                                          \
            }
EINSUMS_SIMD_X86_ICMP(int32_t)
EINSUMS_SIMD_X86_ICMP(uint32_t)
EINSUMS_SIMD_X86_ICMP(int64_t)
EINSUMS_SIMD_X86_ICMP(uint64_t)
#        undef EINSUMS_SIMD_X86_ICMP
#    endif

// ---- select ----
#    if defined(__AVX__)
// BLENDV reads each lane's top bit; the integer forms go through the float domain so that AVX
// without AVX2 has them too.
template <>
EINSUMS_FORCEINLINE Vec<float> select(Mask<float> m, Vec<float> a, Vec<float> b) {
    return _mm256_blendv_ps(b.reg, a.reg, m.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> select(Mask<double> m, Vec<double> a, Vec<double> b) {
    return _mm256_blendv_pd(b.reg, a.reg, m.reg);
}
#        define EINSUMS_SIMD_AVX_SELECT_INT(T, sfx)                                                                                        \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                             \
                return _mm256_cast##sfx##_si256(                                                                                           \
                    _mm256_blendv_##sfx(_mm256_castsi256_##sfx(b.reg), _mm256_castsi256_##sfx(a.reg), _mm256_castsi256_##sfx(m.reg)));     \
            }
EINSUMS_SIMD_AVX_SELECT_INT(int32_t, ps)
EINSUMS_SIMD_AVX_SELECT_INT(uint32_t, ps)
EINSUMS_SIMD_AVX_SELECT_INT(int64_t, pd)
EINSUMS_SIMD_AVX_SELECT_INT(uint64_t, pd)
#        undef EINSUMS_SIMD_AVX_SELECT_INT
#    elif defined(__SSE4_1__)
template <>
EINSUMS_FORCEINLINE Vec<float> select(Mask<float> m, Vec<float> a, Vec<float> b) {
    return _mm_blendv_ps(b.reg, a.reg, m.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<double> select(Mask<double> m, Vec<double> a, Vec<double> b) {
    return _mm_blendv_pd(b.reg, a.reg, m.reg);
}
#        define EINSUMS_SIMD_SSE_SELECT_INT(T)                                                                                             \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                             \
                return _mm_blendv_epi8(b.reg, a.reg, m.reg);                                                                               \
            }
EINSUMS_SIMD_SSE_SELECT_INT(int32_t)
EINSUMS_SIMD_SSE_SELECT_INT(uint32_t)
EINSUMS_SIMD_SSE_SELECT_INT(int64_t)
EINSUMS_SIMD_SSE_SELECT_INT(uint64_t)
#        undef EINSUMS_SIMD_SSE_SELECT_INT
#    else
// SSE2 has no BLENDV: (mask & a) | (~mask & b).
template <>
EINSUMS_FORCEINLINE Vec<float> select(Mask<float> m, Vec<float> a, Vec<float> b) {
    return _mm_or_ps(_mm_and_ps(m.reg, a.reg), _mm_andnot_ps(m.reg, b.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> select(Mask<double> m, Vec<double> a, Vec<double> b) {
    return _mm_or_pd(_mm_and_pd(m.reg, a.reg), _mm_andnot_pd(m.reg, b.reg));
}
#        define EINSUMS_SIMD_SSE_SELECT_INT(T)                                                                                             \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                             \
                return _mm_or_si128(_mm_and_si128(m.reg, a.reg), _mm_andnot_si128(m.reg, b.reg));                                          \
            }
EINSUMS_SIMD_SSE_SELECT_INT(int32_t)
EINSUMS_SIMD_SSE_SELECT_INT(uint32_t)
EINSUMS_SIMD_SSE_SELECT_INT(int64_t)
EINSUMS_SIMD_SSE_SELECT_INT(uint64_t)
#        undef EINSUMS_SIMD_SSE_SELECT_INT
#    endif

// ---- bitwise operations on the values ----
#    if defined(__AVX__)
#        define EINSUMS_SIMD_X86_FLOAT_LOGIC(T, sfx)                                                                                       \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                   \
                return _mm256_and_##sfx(a.reg, b.reg);                                                                                     \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                    \
                return _mm256_or_##sfx(a.reg, b.reg);                                                                                      \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                   \
                return _mm256_xor_##sfx(a.reg, b.reg);                                                                                     \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                \
                return _mm256_andnot_##sfx(b.reg, a.reg);                                                                                  \
            }
#    else
#        define EINSUMS_SIMD_X86_FLOAT_LOGIC(T, sfx)                                                                                       \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                   \
                return _mm_and_##sfx(a.reg, b.reg);                                                                                        \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                    \
                return _mm_or_##sfx(a.reg, b.reg);                                                                                         \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                   \
                return _mm_xor_##sfx(a.reg, b.reg);                                                                                        \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                \
                return _mm_andnot_##sfx(b.reg, a.reg);                                                                                     \
            }
#    endif
EINSUMS_SIMD_X86_FLOAT_LOGIC(float, ps)
EINSUMS_SIMD_X86_FLOAT_LOGIC(double, pd)
#    undef EINSUMS_SIMD_X86_FLOAT_LOGIC
#    define EINSUMS_SIMD_X86_ANDNOT_INT(T)                                                                                                 \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return detail::mask_andnot(a.reg, b.reg);                                                                                      \
        }
EINSUMS_SIMD_X86_ANDNOT_INT(int32_t)
EINSUMS_SIMD_X86_ANDNOT_INT(uint32_t)
EINSUMS_SIMD_X86_ANDNOT_INT(int64_t)
EINSUMS_SIMD_X86_ANDNOT_INT(uint64_t)
#    undef EINSUMS_SIMD_X86_ANDNOT_INT

#elif defined(__aarch64__) || defined(_M_ARM64)
// NEON comparisons give an unsigned lane mask, which is what Mask holds.
template <>
EINSUMS_FORCEINLINE Mask<float> cmp_eq(Vec<float> a, Vec<float> b) {
    return Mask<float>(vceqq_f32(a.reg, b.reg));
}
template <>
EINSUMS_FORCEINLINE Mask<double> cmp_eq(Vec<double> a, Vec<double> b) {
    return Mask<double>(vceqq_f64(a.reg, b.reg));
}
// NEON has no not-equal compare; inverting the ordered equal makes a NaN lane true.
template <>
EINSUMS_FORCEINLINE Mask<float> cmp_ne(Vec<float> a, Vec<float> b) {
    return !cmp_eq(a, b);
}
template <>
EINSUMS_FORCEINLINE Mask<double> cmp_ne(Vec<double> a, Vec<double> b) {
    return !cmp_eq(a, b);
}
#    define EINSUMS_SIMD_NEON_ORDER(T, sfx)                                                                                                \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_lt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(vcltq_##sfx(a.reg, b.reg));                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_le(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(vcleq_##sfx(a.reg, b.reg));                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_gt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(vcgtq_##sfx(a.reg, b.reg));                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ge(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(vcgeq_##sfx(a.reg, b.reg));                                                                                     \
        }
EINSUMS_SIMD_NEON_ORDER(float, f32)
EINSUMS_SIMD_NEON_ORDER(double, f64)
EINSUMS_SIMD_NEON_ORDER(int32_t, s32)
EINSUMS_SIMD_NEON_ORDER(uint32_t, u32)
EINSUMS_SIMD_NEON_ORDER(int64_t, s64)
EINSUMS_SIMD_NEON_ORDER(uint64_t, u64)
#    undef EINSUMS_SIMD_NEON_ORDER
#    define EINSUMS_SIMD_NEON_INT_EQ(T, sfx)                                                                                               \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_eq(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(vceqq_##sfx(a.reg, b.reg));                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ne(Vec<T> a, Vec<T> b) {                                                                           \
            return !cmp_eq(a, b);                                                                                                          \
        }
EINSUMS_SIMD_NEON_INT_EQ(int32_t, s32)
EINSUMS_SIMD_NEON_INT_EQ(uint32_t, u32)
EINSUMS_SIMD_NEON_INT_EQ(int64_t, s64)
EINSUMS_SIMD_NEON_INT_EQ(uint64_t, u64)
#    undef EINSUMS_SIMD_NEON_INT_EQ

#    define EINSUMS_SIMD_NEON_SELECT(T, sfx)                                                                                               \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                                 \
            return vbslq_##sfx(m.reg, a.reg, b.reg);                                                                                       \
        }
EINSUMS_SIMD_NEON_SELECT(float, f32)
EINSUMS_SIMD_NEON_SELECT(double, f64)
EINSUMS_SIMD_NEON_SELECT(int32_t, s32)
EINSUMS_SIMD_NEON_SELECT(uint32_t, u32)
EINSUMS_SIMD_NEON_SELECT(int64_t, s64)
EINSUMS_SIMD_NEON_SELECT(uint64_t, u64)
#    undef EINSUMS_SIMD_NEON_SELECT

#    define EINSUMS_SIMD_NEON_FLOAT_LOGIC(T, sfx)                                                                                          \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return vreinterpretq_##sfx##_u32(vandq_u32(vreinterpretq_u32_##sfx(a.reg), vreinterpretq_u32_##sfx(b.reg)));                   \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return vreinterpretq_##sfx##_u32(vorrq_u32(vreinterpretq_u32_##sfx(a.reg), vreinterpretq_u32_##sfx(b.reg)));                   \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return vreinterpretq_##sfx##_u32(veorq_u32(vreinterpretq_u32_##sfx(a.reg), vreinterpretq_u32_##sfx(b.reg)));                   \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return vreinterpretq_##sfx##_u32(vbicq_u32(vreinterpretq_u32_##sfx(a.reg), vreinterpretq_u32_##sfx(b.reg)));                   \
        }
EINSUMS_SIMD_NEON_FLOAT_LOGIC(float, f32)
EINSUMS_SIMD_NEON_FLOAT_LOGIC(double, f64)
#    undef EINSUMS_SIMD_NEON_FLOAT_LOGIC
template <>
EINSUMS_FORCEINLINE Vec<int32_t> bitwise_andnot(Vec<int32_t> a, Vec<int32_t> b) {
    return vbicq_s32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint32_t> bitwise_andnot(Vec<uint32_t> a, Vec<uint32_t> b) {
    return vbicq_u32(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int64_t> bitwise_andnot(Vec<int64_t> a, Vec<int64_t> b) {
    return vbicq_s64(a.reg, b.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<uint64_t> bitwise_andnot(Vec<uint64_t> a, Vec<uint64_t> b) {
    return vbicq_u64(a.reg, b.reg);
}

#else
// Scalar build: a Vec is one value and a Mask one bool.
namespace detail {
template <typename T>
using mask_bits_t = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
template <typename T, typename Op>
EINSUMS_FORCEINLINE T bits_op(T a, T b, Op op) {
    return std::bit_cast<T>(static_cast<mask_bits_t<T>>(op(std::bit_cast<mask_bits_t<T>>(a), std::bit_cast<mask_bits_t<T>>(b))));
}
} // namespace detail

#    define EINSUMS_SIMD_SCALAR_CMP(T)                                                                                                     \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_eq(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg == b.reg);                                                                                                \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ne(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg != b.reg);                                                                                                \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_lt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg < b.reg);                                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_le(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg <= b.reg);                                                                                                \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_gt(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg > b.reg);                                                                                                 \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Mask<T> cmp_ge(Vec<T> a, Vec<T> b) {                                                                           \
            return Mask<T>(a.reg >= b.reg);                                                                                                \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> select(Mask<T> m, Vec<T> a, Vec<T> b) {                                                                 \
            return m.reg ? a : b;                                                                                                          \
        }
EINSUMS_SIMD_SCALAR_CMP(float)
EINSUMS_SIMD_SCALAR_CMP(double)
EINSUMS_SIMD_SCALAR_CMP(int32_t)
EINSUMS_SIMD_SCALAR_CMP(uint32_t)
EINSUMS_SIMD_SCALAR_CMP(int64_t)
EINSUMS_SIMD_SCALAR_CMP(uint64_t)
#    undef EINSUMS_SIMD_SCALAR_CMP

#    define EINSUMS_SIMD_SCALAR_FLOAT_LOGIC(T)                                                                                             \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_and(Vec<T> a, Vec<T> b) {                                                                       \
            return {detail::bits_op(a.reg, b.reg, [](auto x, auto y) { return x & y; })};                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_or(Vec<T> a, Vec<T> b) {                                                                        \
            return {detail::bits_op(a.reg, b.reg, [](auto x, auto y) { return x | y; })};                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_xor(Vec<T> a, Vec<T> b) {                                                                       \
            return {detail::bits_op(a.reg, b.reg, [](auto x, auto y) { return x ^ y; })};                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return {detail::bits_op(a.reg, b.reg, [](auto x, auto y) { return x & ~y; })};                                                 \
        }
EINSUMS_SIMD_SCALAR_FLOAT_LOGIC(float)
EINSUMS_SIMD_SCALAR_FLOAT_LOGIC(double)
#    undef EINSUMS_SIMD_SCALAR_FLOAT_LOGIC
#    define EINSUMS_SIMD_SCALAR_ANDNOT_INT(T)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> bitwise_andnot(Vec<T> a, Vec<T> b) {                                                                    \
            return {static_cast<T>(a.reg & ~b.reg)};                                                                                       \
        }
EINSUMS_SIMD_SCALAR_ANDNOT_INT(int32_t)
EINSUMS_SIMD_SCALAR_ANDNOT_INT(uint32_t)
EINSUMS_SIMD_SCALAR_ANDNOT_INT(int64_t)
EINSUMS_SIMD_SCALAR_ANDNOT_INT(uint64_t)
#    undef EINSUMS_SIMD_SCALAR_ANDNOT_INT
#endif

// ===========================================================================
// Rounding to an integral value, float and double:
//
//   floor(x)       toward -infinity           std::floor
//   ceil(x)        toward +infinity           std::ceil
//   trunc(x)       toward zero                std::trunc
//   round(x)       nearest, ties away from 0  std::round
//   round_even(x)  nearest, ties to even      std::nearbyint in the default mode
//
// The result stays floating point; convert() turns it into an integer. Each
// matches its std:: function lane for lane, signed zeros included (floor of
// -0.0 is -0.0, ceil of -0.5 is -0.0), and NaN and infinities pass through.
// round_even ignores the dynamic rounding mode, as the instructions below do.
//
// AVX-512, AVX and SSE4.1 round with one instruction in any mode but away from
// zero, which is built from trunc. NEON has an instruction for all five. SSE2
// has none: adding and subtracting 2^23 (float) or 2^52 (double) rounds a
// smaller magnitude to an integer, ties to even, because the sum has no
// fraction bits left, and floor, ceil and trunc correct that by one where it
// went the wrong way. Larger magnitudes are integers already and are returned
// unchanged. That emulation relies on the add and subtract not being folded
// together, so it is wrong under -ffast-math, -fassociative-math or icx's
// default -fp-model=fast.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE Vec<T> floor(Vec<T> x);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> ceil(Vec<T> x);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> trunc(Vec<T> x);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> round(Vec<T> x);
template <typename T>
EINSUMS_FORCEINLINE Vec<T> round_even(Vec<T> x);

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
namespace detail {
/// round(x) from trunc(x): step one further from zero where the dropped fraction is at least a half.
/// x - trunc(x) is exact, so the comparison sees the true fraction.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> round_half_away(Vec<T> x) {
    Vec<T> const t       = trunc(x);
    Vec<T> const away    = bitwise_or(broadcast(T{1}), bitwise_and(x, broadcast(T{-0.0})));
    Vec<T> const dropped = abs(sub(x, t));
    Vec<T> const stepped = add(t, away);
    return select(cmp_ge(dropped, broadcast(T{0.5})), stepped, t);
}
} // namespace detail
#endif

#define EINSUMS_SIMD_X86_ROUNDING(T, ROUND)                                                                                                \
    template <>                                                                                                                            \
    EINSUMS_FORCEINLINE Vec<T> floor(Vec<T> x) {                                                                                           \
        return ROUND(x.reg, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);                                                                    \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_FORCEINLINE Vec<T> ceil(Vec<T> x) {                                                                                            \
        return ROUND(x.reg, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);                                                                    \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_FORCEINLINE Vec<T> trunc(Vec<T> x) {                                                                                           \
        return ROUND(x.reg, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);                                                                       \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_FORCEINLINE Vec<T> round_even(Vec<T> x) {                                                                                      \
        return ROUND(x.reg, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                                                                \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_FORCEINLINE Vec<T> round(Vec<T> x) {                                                                                           \
        return detail::round_half_away(x);                                                                                                 \
    }

#if defined(__AVX512F__) && defined(__AVX512VL__)
// VRNDSCALE with a scale of zero is a plain round in the given mode.
EINSUMS_SIMD_X86_ROUNDING(float, _mm512_roundscale_ps)
EINSUMS_SIMD_X86_ROUNDING(double, _mm512_roundscale_pd)
#elif defined(__AVX__)
EINSUMS_SIMD_X86_ROUNDING(float, _mm256_round_ps)
EINSUMS_SIMD_X86_ROUNDING(double, _mm256_round_pd)
#elif defined(__SSE4_1__)
EINSUMS_SIMD_X86_ROUNDING(float, _mm_round_ps)
EINSUMS_SIMD_X86_ROUNDING(double, _mm_round_pd)
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
namespace detail {
template <typename T>
inline constexpr T no_fraction_bits = sizeof(T) == 4 ? T{8388608.0} : T{4503599627370496.0}; // 2^23, 2^52

template <typename T>
EINSUMS_FORCEINLINE Vec<T> sse2_round_even(Vec<T> x) {
    Vec<T> const big     = broadcast(no_fraction_bits<T>);
    Vec<T> const ax      = abs(x);
    Vec<T> const rounded = sub(add(ax, big), big);
    Vec<T> const signed_ = bitwise_or(rounded, bitwise_and(x, broadcast(T{-0.0})));
    return select(cmp_lt(ax, big), signed_, x);
}

/// floor(x): round to nearest, then step down where that went up. OR-ing in x's sign gives -0.0 for
/// floor(-0.0); every other negative x already floors to a negative value.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> sse2_floor(Vec<T> x) {
    Vec<T> const r = sse2_round_even(x);
    return sub(r, select(cmp_gt(r, x), broadcast(T{1}), broadcast(T{0})));
}

/// ceil(x): round to nearest, then step up where that went down. A negative x above -1 steps up to
/// +0.0, so x's sign is OR-ed back in to make it -0.0.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> sse2_ceil(Vec<T> x) {
    Vec<T> const r  = sse2_round_even(x);
    Vec<T> const up = add(r, select(cmp_lt(r, x), broadcast(T{1}), broadcast(T{0})));
    return bitwise_or(up, bitwise_and(x, broadcast(T{-0.0})));
}

/// trunc(x) = floor(|x|) with x's sign.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> sse2_trunc(Vec<T> x) {
    return bitwise_or(sse2_floor(abs(x)), bitwise_and(x, broadcast(T{-0.0})));
}
} // namespace detail

#    define EINSUMS_SIMD_SSE2_ROUNDING(T)                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> floor(Vec<T> x) {                                                                                       \
            return detail::sse2_floor(x);                                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> ceil(Vec<T> x) {                                                                                        \
            return detail::sse2_ceil(x);                                                                                                   \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> trunc(Vec<T> x) {                                                                                       \
            return detail::sse2_trunc(x);                                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round_even(Vec<T> x) {                                                                                  \
            return detail::sse2_round_even(x);                                                                                             \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round(Vec<T> x) {                                                                                       \
            return detail::round_half_away(x);                                                                                             \
        }
EINSUMS_SIMD_SSE2_ROUNDING(float)
EINSUMS_SIMD_SSE2_ROUNDING(double)
#    undef EINSUMS_SIMD_SSE2_ROUNDING
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define EINSUMS_SIMD_NEON_ROUNDING(T, sfx)                                                                                             \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> floor(Vec<T> x) {                                                                                       \
            return vrndmq_##sfx(x.reg);                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> ceil(Vec<T> x) {                                                                                        \
            return vrndpq_##sfx(x.reg);                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> trunc(Vec<T> x) {                                                                                       \
            return vrndq_##sfx(x.reg);                                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round(Vec<T> x) {                                                                                       \
            return vrndaq_##sfx(x.reg);                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round_even(Vec<T> x) {                                                                                  \
            return vrndnq_##sfx(x.reg);                                                                                                    \
        }
EINSUMS_SIMD_NEON_ROUNDING(float, f32)
EINSUMS_SIMD_NEON_ROUNDING(double, f64)
#    undef EINSUMS_SIMD_NEON_ROUNDING
#else
#    define EINSUMS_SIMD_SCALAR_ROUNDING(T)                                                                                                \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> floor(Vec<T> x) {                                                                                       \
            return {std::floor(x.reg)};                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> ceil(Vec<T> x) {                                                                                        \
            return {std::ceil(x.reg)};                                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> trunc(Vec<T> x) {                                                                                       \
            return {std::trunc(x.reg)};                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round(Vec<T> x) {                                                                                       \
            return {std::round(x.reg)};                                                                                                    \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> round_even(Vec<T> x) {                                                                                  \
            return {std::nearbyint(x.reg)};                                                                                                \
        }
EINSUMS_SIMD_SCALAR_ROUNDING(float)
EINSUMS_SIMD_SCALAR_ROUNDING(double)
#    undef EINSUMS_SIMD_SCALAR_ROUNDING
#endif
#undef EINSUMS_SIMD_X86_ROUNDING

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
#elif defined(__AVX__)
// AVX without AVX2 has no 256-bit integer instructions, and the integer Vecs
// are __m256i here, so no SSE2 form fits them: these operations are left
// undefined for that tier, and a call is a link error.
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
// Dot-product helpers: int8 × int8 → int32 accumulating.
//
// Three signedness combos:
//   - dot_product_ss(acc, a, b): signed × signed
//   - dot_product_uu(acc, a, b): unsigned × unsigned
//   - dot_product_us(acc, a, b): unsigned × signed (the standard ML-quant
//     shape; activations are unsigned, weights are signed)
//
// All three accumulate `acc + sum_of_4_byte_products` per int32 lane:
// 16 byte pairs → 4 int32 outputs per 128-bit register, scaling up to 64
// pairs → 16 outputs at AVX-512 width.
//
// Hardware coverage is uneven: ARM FEAT_DOTPROD gives ss + uu directly,
// FEAT_I8MM (M3+, requires `-mcpu=apple-m4` in this build) adds us.
// Intel AVX-VNNI / AVX-512 VNNI gives only the us shape natively. For the
// missing combos on each platform a kernel needs to either fall back to
// scalar or emulate via shuffle + sign-extend; we don't ship emulation
// here. Calling a missing specialization fails to link, which is the
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
#elif defined(__AVX512VNNI__) && defined(__AVX512F__) && defined(__AVX512VL__)
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b) {
    return _mm512_dpbusd_epi32(acc.reg, a.reg, b.reg);
}
#elif defined(__AVXVNNI__) && defined(__AVX2__)
EINSUMS_FORCEINLINE Vec<int32_t> dot_product_us(Vec<int32_t> acc, Vec<uint8_t> a, Vec<int8_t> b) {
    return _mm256_dpbusd_epi32(acc.reg, a.reg, b.reg);
}
#endif

// ===========================================================================
// IEEE half-precision (Vec<half_t>): broadcast / load / store / arithmetic.
//
// Available when the build sees:
//   - `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` on aarch64 (M1+ default), or
//   - `__AVX512FP16__` on x86 (Sapphire Rapids+, requires -mavx512fp16).
//
// Operations covered: broadcast, loadu/loada, storeu/storea, add, sub, mul,
// fma. Other ops such as gather, transpose, and complex are deferred; they
// need dedicated specializations in Shuffle.hpp / Gather.hpp / ComplexVec.hpp.
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
// Brain-float bf16 (Vec<bfloat16_t>): load / store only.
//
// Both ARM BF16 and AVX-512BF16 expose conversion to/from FP32 plus dot
// product, but no native lane-wise add/sub/mul on bf16. The convention
// in BF16 kernels is "convert to FP32, do math, convert back", so we
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
    // __m512bh and __m512i alias each other on Intel ICC/Clang, so the load
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

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
