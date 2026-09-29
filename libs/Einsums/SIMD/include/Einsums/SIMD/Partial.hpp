//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

EINSUMS_NAMESPACE_BEGIN(simd)

// ===========================================================================
// Partial load / store: the first n lanes of a Vec, for the tail of a loop
// whose length is not a multiple of the lane count.
//
// loadu_partial reads the first min(n, lanes) elements at p and zeroes the
// rest of the Vec; storeu_partial writes the first min(n, lanes) lanes to p.
// Neither touches memory past those elements, so a tail that ends at the end
// of an allocation is safe, which a full-width loadu there is not.
//
// AVX-512 uses masked loads and stores, which do not fault on masked-off
// lanes, and AVX uses VMASKMOV for float and double. Every other combination
// goes through a zeroed stack buffer, which is correct everywhere and costs a
// store-to-load round trip once per loop.
// ===========================================================================

/// True where loadu_partial and storeu_partial are single masked instructions: AVX-512 for float,
/// double and the 32- and 64-bit integers, AVX for float and double. Elsewhere they go through a
/// stack buffer, and a kernel whose loop tail is a few elements may do better with a scalar
/// remainder there; `if constexpr (native_partial<T>)` chooses at compile time, per rung.
template <typename T>
inline constexpr bool native_partial = false;

template <typename T>
EINSUMS_FORCEINLINE Vec<T> loadu_partial(T const *p, std::size_t n) {
    constexpr int L = Vec<T>::lanes;
    if (n >= static_cast<std::size_t>(L)) {
        return loadu<T>(p);
    }
    alignas(native_alignment) T buf[L] = {};
    std::memcpy(buf, p, n * sizeof(T));
    return loadu<T>(buf);
}

template <typename T>
EINSUMS_FORCEINLINE void storeu_partial(T *p, Vec<T> v, std::size_t n) {
    constexpr int L = Vec<T>::lanes;
    if (n >= static_cast<std::size_t>(L)) {
        storeu<T>(p, v);
        return;
    }
    alignas(native_alignment) T buf[L];
    storeu<T>(buf, v);
    std::memcpy(p, buf, n * sizeof(T));
}

#if defined(__AVX512F__) && defined(__AVX512VL__)
template <>
inline constexpr bool native_partial<float> = true;
template <>
inline constexpr bool native_partial<double> = true;
template <>
inline constexpr bool native_partial<int32_t> = true;
template <>
inline constexpr bool native_partial<uint32_t> = true;
template <>
inline constexpr bool native_partial<int64_t> = true;
template <>
inline constexpr bool native_partial<uint64_t> = true;

namespace detail {
/// The lane mask selecting the first min(n, lanes) of @p lanes lanes.
template <typename Mask>
EINSUMS_FORCEINLINE Mask first_lanes(std::size_t n, int lanes) {
    return n >= static_cast<std::size_t>(lanes) ? static_cast<Mask>(~Mask{0}) : static_cast<Mask>((Mask{1} << n) - 1);
}
} // namespace detail

template <>
EINSUMS_FORCEINLINE Vec<float> loadu_partial(float const *p, std::size_t n) {
    return _mm512_maskz_loadu_ps(detail::first_lanes<__mmask16>(n, 16), p);
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu_partial(double const *p, std::size_t n) {
    return _mm512_maskz_loadu_pd(detail::first_lanes<__mmask8>(n, 8), p);
}
template <>
EINSUMS_FORCEINLINE void storeu_partial(float *p, Vec<float> v, std::size_t n) {
    _mm512_mask_storeu_ps(p, detail::first_lanes<__mmask16>(n, 16), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu_partial(double *p, Vec<double> v, std::size_t n) {
    _mm512_mask_storeu_pd(p, detail::first_lanes<__mmask8>(n, 8), v.reg);
}
#    define EINSUMS_SIMD_AVX512_PARTIAL_INT(T, Mask, lanes, W)                                                                             \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> loadu_partial(T const *p, std::size_t n) {                                                              \
            return _mm512_maskz_loadu_epi##W(detail::first_lanes<Mask>(n, lanes), p);                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu_partial(T *p, Vec<T> v, std::size_t n) {                                                           \
            _mm512_mask_storeu_epi##W(p, detail::first_lanes<Mask>(n, lanes), v.reg);                                                      \
        }
EINSUMS_SIMD_AVX512_PARTIAL_INT(int32_t, __mmask16, 16, 32)
EINSUMS_SIMD_AVX512_PARTIAL_INT(uint32_t, __mmask16, 16, 32)
EINSUMS_SIMD_AVX512_PARTIAL_INT(int64_t, __mmask8, 8, 64)
EINSUMS_SIMD_AVX512_PARTIAL_INT(uint64_t, __mmask8, 8, 64)
#    undef EINSUMS_SIMD_AVX512_PARTIAL_INT
#elif defined(__AVX__)
template <>
inline constexpr bool native_partial<float> = true;
template <>
inline constexpr bool native_partial<double> = true;

namespace detail {
// Eight all-ones words followed by eight zeros: loading eight words starting at
// 8 - n gives a VMASKMOV mask whose first n 32-bit lanes are set.
alignas(64) inline constexpr int32_t first_lanes_table[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

EINSUMS_FORCEINLINE __m256i first_lanes_32(std::size_t n) {
    return _mm256_loadu_si256(reinterpret_cast<__m256i const *>(first_lanes_table + (8 - n)));
}
// A 64-bit lane is two 32-bit words, both of which the mask must set.
EINSUMS_FORCEINLINE __m256i first_lanes_64(std::size_t n) {
    return first_lanes_32(2 * n);
}
} // namespace detail

template <>
EINSUMS_FORCEINLINE Vec<float> loadu_partial(float const *p, std::size_t n) {
    if (n >= 8) {
        return _mm256_loadu_ps(p);
    }
    return _mm256_maskload_ps(p, detail::first_lanes_32(n));
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu_partial(double const *p, std::size_t n) {
    if (n >= 4) {
        return _mm256_loadu_pd(p);
    }
    return _mm256_maskload_pd(p, detail::first_lanes_64(n));
}
template <>
EINSUMS_FORCEINLINE void storeu_partial(float *p, Vec<float> v, std::size_t n) {
    if (n >= 8) {
        _mm256_storeu_ps(p, v.reg);
        return;
    }
    _mm256_maskstore_ps(p, detail::first_lanes_32(n), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu_partial(double *p, Vec<double> v, std::size_t n) {
    if (n >= 4) {
        _mm256_storeu_pd(p, v.reg);
        return;
    }
    _mm256_maskstore_pd(p, detail::first_lanes_64(n), v.reg);
}
#endif

EINSUMS_NAMESPACE_END(simd)
