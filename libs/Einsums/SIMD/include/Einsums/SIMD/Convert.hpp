//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
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

EINSUMS_NAMESPACE_END(simd)
