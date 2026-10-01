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
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ===========================================================================
// Masked and partial loads and stores.
//
//   loadu(p, m)              p[i] in each lane m sets, zero elsewhere
//   storeu(p, v, m)          writes p[i] for each lane m sets, nothing elsewhere
//   loadu_partial(p, n)      loadu(p, first_n<T>(n)): the first min(n, lanes)
//   storeu_partial(p, v, n)  storeu(p, v, first_n<T>(n))
//
// None of them reads or writes an inactive lane's memory, so a tail that ends
// at the end of an allocation is safe, which a full-width loadu there is not.
//
// The masked forms are single instructions on AVX-512 (k-masks), on AVX2
// (VMASKMOV for float and double, VPMASKMOV for the 32- and 64-bit integers)
// and on AVX (VMASKMOV, the integers through its float form). SSE and NEON
// have no masked load, and only SSE's slow non-temporal masked store, so there
// they copy the active lanes through a stack buffer. The partial forms exist
// for every Vec type; those without a Mask (the 16-bit floats, the 8-bit
// integers) always copy through a buffer.
// ===========================================================================

/// True where the masked and partial loads and stores of T are single masked instructions. Where
/// they are not, a kernel whose loop tail is a few elements may do better with a scalar remainder;
/// `if constexpr (native_masked_memory<T>)` chooses at compile time, per rung.
template <typename T>
inline constexpr bool native_masked_memory = false;
/// The earlier name of native_masked_memory.
template <typename T>
inline constexpr bool native_partial = native_masked_memory<T>;

template <typename T>
EINSUMS_FORCEINLINE Vec<T> loadu(T const *p, Mask<T> m);
template <typename T>
EINSUMS_FORCEINLINE void storeu(T *p, Vec<T> v, Mask<T> m);

namespace detail {
/// The emulated masked load: p[i] for each lane set in bits, zero elsewhere, through a stack buffer.
/// SSE and NEON overload it with forms that assemble the lanes in registers (below); the scalar
/// build, and rungs with a native masked load, which never call it, keep this one.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> load_lanes(T const *p, uint64_t bits) {
    constexpr int               L      = Vec<T>::lanes;
    alignas(native_alignment) T buf[L] = {};
    for (int i = 0; i < L; ++i) {
        if ((bits >> i) & 1u) {
            buf[i] = p[i];
        }
    }
    return loada(buf);
}
/// The masked store through a stack buffer, touching only the active lanes. The buffer costs a
/// store load round trip here too, but a vector store forwards to the narrower loads inside it.
template <typename T>
EINSUMS_FORCEINLINE void storeu_masked_lanes(T *p, Vec<T> v, Mask<T> m) {
    constexpr int               L = Vec<T>::lanes;
    alignas(native_alignment) T buf[L];
    storea(buf, v);
    uint64_t const set_lanes = to_bits(m);
    for (int i = 0; i < L; ++i) {
        if ((set_lanes >> i) & 1u) {
            p[i] = buf[i];
        }
    }
}
} // namespace detail

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define EINSUMS_SIMD_AVX512_MASKED(T, sfx)                                                                                             \
        template <>                                                                                                                        \
        inline constexpr bool native_masked_memory<T> = true;                                                                              \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> loadu(T const *p, Mask<T> m) {                                                                          \
            return _mm512_maskz_loadu_##sfx(m.reg, p);                                                                                     \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(T *p, Vec<T> v, Mask<T> m) {                                                                       \
            _mm512_mask_storeu_##sfx(p, m.reg, v.reg);                                                                                     \
        }
EINSUMS_SIMD_AVX512_MASKED(float, ps)
EINSUMS_SIMD_AVX512_MASKED(double, pd)
EINSUMS_SIMD_AVX512_MASKED(int32_t, epi32)
EINSUMS_SIMD_AVX512_MASKED(uint32_t, epi32)
EINSUMS_SIMD_AVX512_MASKED(int64_t, epi64)
EINSUMS_SIMD_AVX512_MASKED(uint64_t, epi64)
#    undef EINSUMS_SIMD_AVX512_MASKED
#elif defined(__AVX__)
template <>
inline constexpr bool native_masked_memory<float> = true;
template <>
inline constexpr bool native_masked_memory<double> = true;
template <>
EINSUMS_FORCEINLINE Vec<float> loadu(float const *p, Mask<float> m) {
    return _mm256_maskload_ps(p, _mm256_castps_si256(m.reg));
}
template <>
EINSUMS_FORCEINLINE Vec<double> loadu(double const *p, Mask<double> m) {
    return _mm256_maskload_pd(p, _mm256_castpd_si256(m.reg));
}
template <>
EINSUMS_FORCEINLINE void storeu(float *p, Vec<float> v, Mask<float> m) {
    _mm256_maskstore_ps(p, _mm256_castps_si256(m.reg), v.reg);
}
template <>
EINSUMS_FORCEINLINE void storeu(double *p, Vec<double> v, Mask<double> m) {
    _mm256_maskstore_pd(p, _mm256_castpd_si256(m.reg), v.reg);
}
#    if defined(__AVX2__)
#        define EINSUMS_SIMD_AVX_MASKED_INT(T, W, P)                                                                                       \
            template <>                                                                                                                    \
            inline constexpr bool native_masked_memory<T> = true;                                                                          \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> loadu(T const *p, Mask<T> m) {                                                                      \
                return _mm256_maskload_epi##W(reinterpret_cast<P const *>(p), m.reg);                                                      \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE void storeu(T *p, Vec<T> v, Mask<T> m) {                                                                   \
                _mm256_maskstore_epi##W(reinterpret_cast<P *>(p), m.reg, v.reg);                                                           \
            }
EINSUMS_SIMD_AVX_MASKED_INT(int32_t, 32, int)
EINSUMS_SIMD_AVX_MASKED_INT(uint32_t, 32, int)
EINSUMS_SIMD_AVX_MASKED_INT(int64_t, 64, long long)
EINSUMS_SIMD_AVX_MASKED_INT(uint64_t, 64, long long)
#        undef EINSUMS_SIMD_AVX_MASKED_INT
#    else
// AVX without AVX2: the integers through VMASKMOV's float form, which moves the bits unchanged.
#        define EINSUMS_SIMD_AVX_MASKED_INT(T, sfx, F)                                                                                     \
            template <>                                                                                                                    \
            inline constexpr bool native_masked_memory<T> = true;                                                                          \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE Vec<T> loadu(T const *p, Mask<T> m) {                                                                      \
                return _mm256_cast##sfx##_si256(_mm256_maskload_##sfx(reinterpret_cast<F const *>(p), m.reg));                             \
            }                                                                                                                              \
            template <>                                                                                                                    \
            EINSUMS_FORCEINLINE void storeu(T *p, Vec<T> v, Mask<T> m) {                                                                   \
                _mm256_maskstore_##sfx(reinterpret_cast<F *>(p), m.reg, _mm256_castsi256_##sfx(v.reg));                                    \
            }
EINSUMS_SIMD_AVX_MASKED_INT(int32_t, ps, float)
EINSUMS_SIMD_AVX_MASKED_INT(uint32_t, ps, float)
EINSUMS_SIMD_AVX_MASKED_INT(int64_t, pd, double)
EINSUMS_SIMD_AVX_MASKED_INT(uint64_t, pd, double)
#        undef EINSUMS_SIMD_AVX_MASKED_INT
#    endif
#else
// No masked load: the active lanes are read one at a time and assembled in registers, never through
// a stack buffer. Writing the lanes to memory and reloading them as a vector stalls on store
// forwarding (the stall the SSE gather had), about 9 ns a call on Zen+, which a kernel with a short
// tail per element group pays on every group.
namespace detail {
#    if defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
/// Four 32-bit lanes; the prefixes a partial load asks for are one or two loads each. The integer
/// loads carry no type, so float data is read without aliasing it as an integer.
EINSUMS_FORCEINLINE __m128i load_lanes_32x4(void const *p, uint64_t bits) {
    auto const *q = static_cast<char const *>(p);
    switch (bits & 15u) {
    case 0:
        return _mm_setzero_si128();
    case 1:
        return _mm_loadu_si32(q);
    case 3:
        return _mm_loadu_si64(q);
    case 7:
        return _mm_unpacklo_epi64(_mm_loadu_si64(q), _mm_loadu_si32(q + 8));
    case 15:
        return _mm_loadu_si128(reinterpret_cast<__m128i const *>(q));
    default: {
        __m128i const l0 = (bits & 1u) ? _mm_loadu_si32(q) : _mm_setzero_si128();
        __m128i const l1 = (bits & 2u) ? _mm_loadu_si32(q + 4) : _mm_setzero_si128();
        __m128i const l2 = (bits & 4u) ? _mm_loadu_si32(q + 8) : _mm_setzero_si128();
        __m128i const l3 = (bits & 8u) ? _mm_loadu_si32(q + 12) : _mm_setzero_si128();
        return _mm_unpacklo_epi64(_mm_unpacklo_epi32(l0, l1), _mm_unpacklo_epi32(l2, l3));
    }
    }
}
/// Two 64-bit lanes.
EINSUMS_FORCEINLINE __m128i load_lanes_64x2(void const *p, uint64_t bits) {
    auto const *q = static_cast<char const *>(p);
    switch (bits & 3u) {
    case 0:
        return _mm_setzero_si128();
    case 1:
        return _mm_loadu_si64(q);
    case 2:
        return _mm_unpacklo_epi64(_mm_setzero_si128(), _mm_loadu_si64(q + 8));
    default:
        return _mm_loadu_si128(reinterpret_cast<__m128i const *>(q));
    }
}
EINSUMS_FORCEINLINE Vec<float> load_lanes(float const *p, uint64_t bits) {
    return _mm_castsi128_ps(load_lanes_32x4(p, bits));
}
EINSUMS_FORCEINLINE Vec<double> load_lanes(double const *p, uint64_t bits) {
    return _mm_castsi128_pd(load_lanes_64x2(p, bits));
}
EINSUMS_FORCEINLINE Vec<int32_t> load_lanes(int32_t const *p, uint64_t bits) {
    return load_lanes_32x4(p, bits);
}
EINSUMS_FORCEINLINE Vec<uint32_t> load_lanes(uint32_t const *p, uint64_t bits) {
    return load_lanes_32x4(p, bits);
}
EINSUMS_FORCEINLINE Vec<int64_t> load_lanes(int64_t const *p, uint64_t bits) {
    return load_lanes_64x2(p, bits);
}
EINSUMS_FORCEINLINE Vec<uint64_t> load_lanes(uint64_t const *p, uint64_t bits) {
    return load_lanes_64x2(p, bits);
}
#    elif defined(__aarch64__) || defined(_M_ARM64)
/// Each active lane loaded into a zeroed register with LD1 (single structure, one lane).
#        define EINSUMS_SIMD_NEON_LOAD_LANES(T, sfx)                                                                                       \
            EINSUMS_FORCEINLINE Vec<T> load_lanes(T const *p, uint64_t bits) {                                                             \
                auto r = vdupq_n_##sfx(0);                                                                                                 \
                if (bits & 1u) {                                                                                                           \
                    r = vld1q_lane_##sfx(p, r, 0);                                                                                         \
                }                                                                                                                          \
                if (bits & 2u) {                                                                                                           \
                    r = vld1q_lane_##sfx(p + 1, r, 1);                                                                                     \
                }                                                                                                                          \
                if constexpr (Vec<T>::lanes == 4) {                                                                                        \
                    if (bits & 4u) {                                                                                                       \
                        r = vld1q_lane_##sfx(p + 2, r, 2 % Vec<T>::lanes);                                                                 \
                    }                                                                                                                      \
                    if (bits & 8u) {                                                                                                       \
                        r = vld1q_lane_##sfx(p + 3, r, 3 % Vec<T>::lanes);                                                                 \
                    }                                                                                                                      \
                }                                                                                                                          \
                return r;                                                                                                                  \
            }
EINSUMS_SIMD_NEON_LOAD_LANES(float, f32)
EINSUMS_SIMD_NEON_LOAD_LANES(double, f64)
EINSUMS_SIMD_NEON_LOAD_LANES(int32_t, s32)
EINSUMS_SIMD_NEON_LOAD_LANES(uint32_t, u32)
EINSUMS_SIMD_NEON_LOAD_LANES(int64_t, s64)
EINSUMS_SIMD_NEON_LOAD_LANES(uint64_t, u64)
#        undef EINSUMS_SIMD_NEON_LOAD_LANES
#    endif
} // namespace detail

#    define EINSUMS_SIMD_EMULATED_MASKED(T)                                                                                                \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE Vec<T> loadu(T const *p, Mask<T> m) {                                                                          \
            return detail::load_lanes(p, to_bits(m));                                                                                      \
        }                                                                                                                                  \
        template <>                                                                                                                        \
        EINSUMS_FORCEINLINE void storeu(T *p, Vec<T> v, Mask<T> m) {                                                                       \
            detail::storeu_masked_lanes(p, v, m);                                                                                          \
        }
EINSUMS_SIMD_EMULATED_MASKED(float)
EINSUMS_SIMD_EMULATED_MASKED(double)
EINSUMS_SIMD_EMULATED_MASKED(int32_t)
EINSUMS_SIMD_EMULATED_MASKED(uint32_t)
EINSUMS_SIMD_EMULATED_MASKED(int64_t)
EINSUMS_SIMD_EMULATED_MASKED(uint64_t)
#    undef EINSUMS_SIMD_EMULATED_MASKED
#endif

/// The first min(n, lanes) elements at p, zero in the other lanes.
template <typename T>
EINSUMS_FORCEINLINE Vec<T> loadu_partial(T const *p, std::size_t n) {
    constexpr int L = Vec<T>::lanes;
    if (n >= static_cast<std::size_t>(L)) {
        return loadu<T>(p);
    }
    if constexpr (native_masked_memory<T>) {
        return loadu(p, first_n<T>(n));
    } else if constexpr (detail::has_mask<T>) {
        // The prefix's bits directly, without building the mask only to read its bits back.
        return detail::load_lanes(p, (uint64_t{1} << n) - 1u);
    } else {
        alignas(native_alignment) T buf[L] = {};
        std::memcpy(buf, p, n * sizeof(T));
        return loadu<T>(buf);
    }
}

/// Write the first min(n, lanes) lanes of v to p.
template <typename T>
EINSUMS_FORCEINLINE void storeu_partial(T *p, Vec<T> v, std::size_t n) {
    constexpr int L = Vec<T>::lanes;
    if (n >= static_cast<std::size_t>(L)) {
        storeu<T>(p, v);
        return;
    }
    if constexpr (detail::has_mask<T>) {
        storeu(p, v, first_n<T>(n));
    } else {
        alignas(native_alignment) T buf[L];
        storeu<T>(buf, v);
        std::memcpy(p, buf, n * sizeof(T));
    }
}

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
