//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(simd)

// ===========================================================================
// Horizontal reductions: the sum, minimum or maximum of a Vec's lanes.
//
// For float, double and int32_t. The lanes are combined pairwise in a tree
// whose shape depends on the register width, so a floating-point sum can
// differ in its last bits from one ISA to another and from a sequential
// loop. A NaN lane gives an unspecified result from reduce_min and
// reduce_max. The integer sum wraps.
// ===========================================================================

template <typename T>
EINSUMS_FORCEINLINE T reduce_add(Vec<T> v);
template <typename T>
EINSUMS_FORCEINLINE T reduce_min(Vec<T> v);
template <typename T>
EINSUMS_FORCEINLINE T reduce_max(Vec<T> v);

#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX512F__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
namespace detail {
// Fold a 128-bit register to lane 0 with op, then read lane 0.
template <typename Op>
EINSUMS_FORCEINLINE float fold128_ps(__m128 v, Op op) {
    __m128 const swapped = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 const pairs   = op(v, swapped);
    return _mm_cvtss_f32(op(pairs, _mm_movehl_ps(pairs, pairs)));
}
template <typename Op>
EINSUMS_FORCEINLINE double fold128_pd(__m128d v, Op op) {
    return _mm_cvtsd_f64(op(v, _mm_unpackhi_pd(v, v)));
}
template <typename Op>
EINSUMS_FORCEINLINE int32_t fold128_epi32(__m128i v, Op op) {
    __m128i const pairs = op(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(op(pairs, _mm_shuffle_epi32(pairs, _MM_SHUFFLE(1, 0, 3, 2))));
}

// The three combining steps, one object each so the folds inline them.
struct add_op {
    EINSUMS_FORCEINLINE __m128  operator()(__m128 a, __m128 b) const { return _mm_add_ps(a, b); }
    EINSUMS_FORCEINLINE __m128d operator()(__m128d a, __m128d b) const { return _mm_add_pd(a, b); }
    EINSUMS_FORCEINLINE __m128i operator()(__m128i a, __m128i b) const { return _mm_add_epi32(a, b); }
};
struct min_op {
    EINSUMS_FORCEINLINE __m128  operator()(__m128 a, __m128 b) const { return _mm_min_ps(a, b); }
    EINSUMS_FORCEINLINE __m128d operator()(__m128d a, __m128d b) const { return _mm_min_pd(a, b); }
#    if defined(__SSE4_1__)
    EINSUMS_FORCEINLINE __m128i operator()(__m128i a, __m128i b) const { return _mm_min_epi32(a, b); }
#    else
    // SSE2 has no signed 32-bit min: pick through a compare.
    EINSUMS_FORCEINLINE __m128i operator()(__m128i a, __m128i b) const {
        __m128i const a_greater = _mm_cmpgt_epi32(a, b);
        return _mm_or_si128(_mm_and_si128(a_greater, b), _mm_andnot_si128(a_greater, a));
    }
#    endif
};
struct max_op {
    EINSUMS_FORCEINLINE __m128  operator()(__m128 a, __m128 b) const { return _mm_max_ps(a, b); }
    EINSUMS_FORCEINLINE __m128d operator()(__m128d a, __m128d b) const { return _mm_max_pd(a, b); }
#    if defined(__SSE4_1__)
    EINSUMS_FORCEINLINE __m128i operator()(__m128i a, __m128i b) const { return _mm_max_epi32(a, b); }
#    else
    EINSUMS_FORCEINLINE __m128i operator()(__m128i a, __m128i b) const {
        __m128i const a_greater = _mm_cmpgt_epi32(a, b);
        return _mm_or_si128(_mm_and_si128(a_greater, a), _mm_andnot_si128(a_greater, b));
    }
#    endif
};
} // namespace detail
#endif

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE float reduce_add(Vec<float> v) {
    return _mm512_reduce_add_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_add(Vec<double> v) {
    return _mm512_reduce_add_pd(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_add(Vec<int32_t> v) {
    return _mm512_reduce_add_epi32(v.reg);
}
template <>
EINSUMS_FORCEINLINE float reduce_min(Vec<float> v) {
    return _mm512_reduce_min_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_min(Vec<double> v) {
    return _mm512_reduce_min_pd(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_min(Vec<int32_t> v) {
    return _mm512_reduce_min_epi32(v.reg);
}
template <>
EINSUMS_FORCEINLINE float reduce_max(Vec<float> v) {
    return _mm512_reduce_max_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_max(Vec<double> v) {
    return _mm512_reduce_max_pd(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_max(Vec<int32_t> v) {
    return _mm512_reduce_max_epi32(v.reg);
}
#elif defined(__AVX__)
// Fold the two 128-bit halves together, then finish in SSE. Only AVX
// instructions touch the 256-bit register, so this needs no AVX2.
#    define EINSUMS_SIMD_AVX_REDUCE(name, op)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE float name(Vec<float> v) {                                                                                     \
            return detail::fold128_ps(detail::op##_op{}(_mm256_castps256_ps128(v.reg), _mm256_extractf128_ps(v.reg, 1)),                   \
                                      detail::op##_op{});                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE double name(Vec<double> v) {                                                                                   \
            return detail::fold128_pd(detail::op##_op{}(_mm256_castpd256_pd128(v.reg), _mm256_extractf128_pd(v.reg, 1)),                   \
                                      detail::op##_op{});                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE int32_t name(Vec<int32_t> v) {                                                                                 \
            return detail::fold128_epi32(detail::op##_op{}(_mm256_castsi256_si128(v.reg), _mm256_extractf128_si256(v.reg, 1)),             \
                                         detail::op##_op{});                                                                               \
        }
EINSUMS_SIMD_AVX_REDUCE(reduce_add, add)
EINSUMS_SIMD_AVX_REDUCE(reduce_min, min)
EINSUMS_SIMD_AVX_REDUCE(reduce_max, max)
#    undef EINSUMS_SIMD_AVX_REDUCE
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define EINSUMS_SIMD_SSE_REDUCE(name, op)                                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE float name(Vec<float> v) {                                                                                     \
            return detail::fold128_ps(v.reg, detail::op##_op{});                                                                           \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE double name(Vec<double> v) {                                                                                   \
            return detail::fold128_pd(v.reg, detail::op##_op{});                                                                           \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE int32_t name(Vec<int32_t> v) {                                                                                 \
            return detail::fold128_epi32(v.reg, detail::op##_op{});                                                                        \
        }
EINSUMS_SIMD_SSE_REDUCE(reduce_add, add)
EINSUMS_SIMD_SSE_REDUCE(reduce_min, min)
EINSUMS_SIMD_SSE_REDUCE(reduce_max, max)
#    undef EINSUMS_SIMD_SSE_REDUCE
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE float reduce_add(Vec<float> v) {
    return vaddvq_f32(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_add(Vec<double> v) {
    return vaddvq_f64(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_add(Vec<int32_t> v) {
    return vaddvq_s32(v.reg);
}
template <>
EINSUMS_FORCEINLINE float reduce_min(Vec<float> v) {
    return vminvq_f32(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_min(Vec<double> v) {
    return vminvq_f64(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_min(Vec<int32_t> v) {
    return vminvq_s32(v.reg);
}
template <>
EINSUMS_FORCEINLINE float reduce_max(Vec<float> v) {
    return vmaxvq_f32(v.reg);
}
template <>
EINSUMS_FORCEINLINE double reduce_max(Vec<double> v) {
    return vmaxvq_f64(v.reg);
}
template <>
EINSUMS_FORCEINLINE int32_t reduce_max(Vec<int32_t> v) {
    return vmaxvq_s32(v.reg);
}
#else
// Scalar fallback: one lane is its own reduction.
#    define EINSUMS_SIMD_SCALAR_REDUCE(T)                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE T reduce_add(Vec<T> v) {                                                                                       \
            return v.reg;                                                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE T reduce_min(Vec<T> v) {                                                                                       \
            return v.reg;                                                                                                                  \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE T reduce_max(Vec<T> v) {                                                                                       \
            return v.reg;                                                                                                                  \
        }
EINSUMS_SIMD_SCALAR_REDUCE(float)
EINSUMS_SIMD_SCALAR_REDUCE(double)
EINSUMS_SIMD_SCALAR_REDUCE(int32_t)
#    undef EINSUMS_SIMD_SCALAR_REDUCE
#endif

EINSUMS_NAMESPACE_END(simd)
