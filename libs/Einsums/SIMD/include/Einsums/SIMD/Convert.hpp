//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstdint>
#include <limits>

EINSUMS_NAMESPACE_BEGIN(simd)

// ===========================================================================
// Lane-wise conversion between element types of the same width, so the lane
// count carries over: convert<float>(Vec<int32_t>) and convert<int32_t>(Vec<float>).
//
// int32 to float rounds to nearest (ties to even), which is exact below 2^24.
// float to int32 truncates toward zero, as a C++ cast does. A NaN, or a value
// outside the int32 range, gives an unspecified lane: x86 returns INT32_MIN
// and NEON saturates. The scalar fallback returns INT32_MIN rather than
// performing the cast, which C++ leaves undefined.
// ===========================================================================

template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert(Vec<From> v);

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm512_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm512_cvttps_epi32(v.reg);
}
#elif defined(__AVX__)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm256_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm256_cvttps_epi32(v.reg);
}
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return _mm_cvtepi32_ps(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return _mm_cvttps_epi32(v.reg);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return vcvtq_f32_s32(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    return vcvtq_s32_f32(v.reg);
}
#else
template <>
EINSUMS_FORCEINLINE Vec<float> convert<float, int32_t>(Vec<int32_t> v) {
    return {static_cast<float>(v.reg)};
}
template <>
EINSUMS_FORCEINLINE Vec<int32_t> convert<int32_t, float>(Vec<float> v) {
    // 2^31 is exactly representable; every float below it and at or above -2^31 truncates into range.
    constexpr float limit = 2147483648.0f;
    if (!(v.reg >= -limit && v.reg < limit)) {
        return {std::numeric_limits<int32_t>::min()};
    }
    return {static_cast<int32_t>(v.reg)};
}
#endif

// ===========================================================================
// Widening and narrowing between float and the 16-bit float types half_t and
// bfloat16_t, where the build has them (see Vec.hpp). A 16-bit vector holds
// exactly twice Vec<float>'s lanes, so it widens to two float vectors:
//
//   convert_low<float>(v)    lanes 0 .. L/2 - 1 of v, as float
//   convert_high<float>(v)   lanes L/2 .. L - 1 of v, as float
//   convert<H>(low, high)    low's lanes then high's, rounded to H
//
// Widening is exact. Narrowing rounds to nearest, ties to even, as a C++
// conversion does; a NaN stays a NaN, though its payload need not survive.
// NEON uses its conversion instructions. Every other build that has the types
// converts lane by lane through memory, which is correct everywhere and waits
// for hardware to measure a native version against.
// ===========================================================================

template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert_low(Vec<From> v);
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert_high(Vec<From> v);
template <typename To, typename From>
EINSUMS_FORCEINLINE Vec<To> convert(Vec<From> low, Vec<From> high);

namespace detail {
/// Half of a 16-bit float vector, lanes [offset, offset + lanes of float), widened lane by lane.
template <typename H>
EINSUMS_FORCEINLINE Vec<float> widen_lanes(Vec<H> v, int offset) {
    constexpr int L = Vec<float>::lanes;
    static_assert(Vec<H>::lanes == 2 * L, "a 16-bit float vector holds twice float's lanes");
    alignas(native_alignment) H     narrow[2 * L];
    alignas(native_alignment) float wide[L];
    storeu(narrow, v);
    for (int i = 0; i < L; ++i) {
        wide[i] = static_cast<float>(narrow[offset + i]);
    }
    return loadu(wide);
}

/// Two float vectors rounded lane by lane into one 16-bit float vector.
template <typename H>
EINSUMS_FORCEINLINE Vec<H> narrow_lanes(Vec<float> low, Vec<float> high) {
    constexpr int L = Vec<float>::lanes;
    static_assert(Vec<H>::lanes == 2 * L, "a 16-bit float vector holds twice float's lanes");
    alignas(native_alignment) float wide[2 * L];
    alignas(native_alignment) H     narrow[2 * L];
    storeu(wide, low);
    storeu(wide + L, high);
    for (int i = 0; i < 2 * L; ++i) {
        narrow[i] = static_cast<H>(wide[i]);
    }
    return loadu(narrow);
}
} // namespace detail

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
#    if defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, half_t>(Vec<half_t> v) {
    return vcvt_f32_f16(vget_low_f16(v.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, half_t>(Vec<half_t> v) {
    return vcvt_high_f32_f16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> convert<half_t, float>(Vec<float> low, Vec<float> high) {
    return vcvt_high_f16_f32(vcvt_f16_f32(low.reg), high.reg);
}
#    else
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, half_t>(Vec<half_t> v) {
    return detail::widen_lanes(v, 0);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, half_t>(Vec<half_t> v) {
    return detail::widen_lanes(v, Vec<float>::lanes);
}
template <>
EINSUMS_FORCEINLINE Vec<half_t> convert<half_t, float>(Vec<float> low, Vec<float> high) {
    return detail::narrow_lanes<half_t>(low, high);
}
#    endif
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
#    if defined(__aarch64__) || defined(_M_ARM64)
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return vcvtq_low_f32_bf16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return vcvtq_high_f32_bf16(v.reg);
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> convert<bfloat16_t, float>(Vec<float> low, Vec<float> high) {
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(low.reg), high.reg);
}
#    else
template <>
EINSUMS_FORCEINLINE Vec<float> convert_low<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return detail::widen_lanes(v, 0);
}
template <>
EINSUMS_FORCEINLINE Vec<float> convert_high<float, bfloat16_t>(Vec<bfloat16_t> v) {
    return detail::widen_lanes(v, Vec<float>::lanes);
}
template <>
EINSUMS_FORCEINLINE Vec<bfloat16_t> convert<bfloat16_t, float>(Vec<float> low, Vec<float> high) {
    return detail::narrow_lanes<bfloat16_t>(low, high);
}
#    endif
#endif

EINSUMS_NAMESPACE_END(simd)
