//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/Platform.hpp>

#include <array>
#include <cstring>
#include <type_traits>

// Include platform intrinsic headers
#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX512F__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <immintrin.h>
#    endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#    include <arm_neon.h>
#endif

EINSUMS_NAMESPACE_BEGIN(simd)
EINSUMS_SIMD_ISA_NAMESPACE_BEGIN()

// ---------------------------------------------------------------------------
// Half-precision type aliases.
//
// `half_t` is the IEEE-754 binary16 type (5-bit exponent, 10-bit mantissa),
// the same numeric range as float but roughly half the precision. ARM uses `__fp16`
// (also exposed as `_Float16` on newer Clang) under the FP16 vector
// extension; x86 uses `_Float16` under AVX-512FP16.
//
// `bfloat16_t` is the brain-float format (8-bit exponent, 7-bit mantissa),
// the same range as float with much less precision. Useful for ML inference.
// Both ARM and x86 spell it `__bf16`.
//
// These aliases only exist when the compiler accepts the underlying type;
// builds without the appropriate -mcpu / -mavx512fp16 flags simply won't
// see them. Vec<half_t> / Vec<bfloat16_t> are correspondingly only defined
// on those builds.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
using half_t = __fp16;
#elif defined(__AVX512FP16__)
using half_t = _Float16;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
using bfloat16_t = __bf16;
#endif

// ---------------------------------------------------------------------------
// VecTraits<T>: maps scalar type T to native SIMD register type.
// Specializations are guarded by ISA macros. Widest available wins.
// ---------------------------------------------------------------------------

template <typename T>
struct VecTraits;

// ===========================================================================
// x86 AVX-512: 512-bit registers
// ===========================================================================
#if defined(__AVX512F__) && defined(__AVX512VL__)

template <>
struct VecTraits<float> {
    using reg_type             = __m512;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m512d;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};

// All x86 integer types share __m512i, so the element type is encoded in the
// `lanes` count and the per-op intrinsic the user calls (epi32 / epi64 / …).
template <>
struct VecTraits<int32_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 512;
};

// 8-bit integer types: 64 lanes per 512-bit register. The general
// arithmetic API is intentionally NOT specialized for these because
// per-lane mul/shift on bytes has saturation surprises and many ISAs
// only expose dot-product accumulators rather than direct ops. Use
// the dot_product_* helpers in Operations.hpp.
template <>
struct VecTraits<int8_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 64;
    static constexpr int bits  = 512;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m512i;
    using int_type             = __m512i;
    static constexpr int lanes = 64;
    static constexpr int bits  = 512;
};

// AVX-512FP16: 32 IEEE half-precision lanes per 512-bit register.
#    if defined(__AVX512FP16__)
template <>
struct VecTraits<half_t> {
    using reg_type             = __m512h;
    using int_type             = __m512i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 512;
};
#    endif
// AVX-512BF16: 32 bfloat16 lanes; native arithmetic intrinsics are limited
// to convert + dot product (see Operations.hpp). The register type alias
// follows Intel's `__m512bh`.
#    if defined(__AVX512BF16__)
template <>
struct VecTraits<bfloat16_t> {
    using reg_type             = __m512bh;
    using int_type             = __m512i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 512;
};
#    endif

// ===========================================================================
// x86 AVX/AVX2: 256-bit registers
// ===========================================================================
#elif defined(__AVX__)

template <>
struct VecTraits<float> {
    using reg_type             = __m256;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m256d;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 8;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 256;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 256;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m256i;
    using int_type             = __m256i;
    static constexpr int lanes = 32;
    static constexpr int bits  = 256;
};

// ===========================================================================
// x86 SSE2: 128-bit registers
// ===========================================================================
#elif defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))

template <>
struct VecTraits<float> {
    using reg_type             = __m128;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<double> {
    using reg_type             = __m128d;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = __m128i;
    using int_type             = __m128i;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};

// ===========================================================================
// ARM NEON (aarch64, including Apple Silicon): 128-bit registers
// ===========================================================================
#elif defined(__aarch64__) || defined(_M_ARM64)

template <>
struct VecTraits<float> {
    using reg_type             = float32x4_t;
    using int_type             = int32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<double> {
    using reg_type             = float64x2_t;
    using int_type             = int64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

// NEON has distinct register types per signedness/width, such as vaddq_s32 vs
// vaddq_u32 vs vaddq_s64, so each integer Vec specialization picks
// the matching `*x*_t` typedef.
template <>
struct VecTraits<int32_t> {
    using reg_type             = int32x4_t;
    using int_type             = int32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = uint32x4_t;
    using int_type             = uint32x4_t;
    static constexpr int lanes = 4;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = int64x2_t;
    using int_type             = int64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = uint64x2_t;
    using int_type             = uint64x2_t;
    static constexpr int lanes = 2;
    static constexpr int bits  = 128;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = int8x16_t;
    using int_type             = int8x16_t;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = uint8x16_t;
    using int_type             = uint8x16_t;
    static constexpr int lanes = 16;
    static constexpr int bits  = 128;
};

// NEON FP16 (M1+, gated on -march=armv8.2-a+fp16 or -mcpu=apple-m1+).
// 8 half-precision lanes per 128-bit register.
#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
struct VecTraits<half_t> {
    using reg_type             = float16x8_t;
    using int_type             = int16x8_t;
    static constexpr int lanes = 8;
    static constexpr int bits  = 128;
};
#    endif
// NEON BF16 (M3+ / Cortex-A715+, gated on -mcpu=apple-m3+ / -march=…+bf16).
#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <>
struct VecTraits<bfloat16_t> {
    using reg_type             = bfloat16x8_t;
    using int_type             = int16x8_t;
    static constexpr int lanes = 8;
    static constexpr int bits  = 128;
};
#    endif

// ===========================================================================
// Scalar fallback: no SIMD
// ===========================================================================
#else

template <>
struct VecTraits<float> {
    using reg_type             = float;
    using int_type             = int32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};

template <>
struct VecTraits<double> {
    using reg_type             = double;
    using int_type             = int64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};

template <>
struct VecTraits<int32_t> {
    using reg_type             = int32_t;
    using int_type             = int32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};
template <>
struct VecTraits<uint32_t> {
    using reg_type             = uint32_t;
    using int_type             = uint32_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 32;
};
template <>
struct VecTraits<int64_t> {
    using reg_type             = int64_t;
    using int_type             = int64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};
template <>
struct VecTraits<uint64_t> {
    using reg_type             = uint64_t;
    using int_type             = uint64_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 64;
};

template <>
struct VecTraits<int8_t> {
    using reg_type             = int8_t;
    using int_type             = int8_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 8;
};
template <>
struct VecTraits<uint8_t> {
    using reg_type             = uint8_t;
    using int_type             = uint8_t;
    static constexpr int lanes = 1;
    static constexpr int bits  = 8;
};

#endif

// ---------------------------------------------------------------------------
// Vec<T, N>: N lanes of T.
//
// N defaults to the native lane count, VecTraits<T>::lanes, and then Vec<T> is
// a thin wrapper over one platform register: no virtual dispatch, implicit
// conversion to and from the raw register for interop, and operator[] for
// debugging. Every operation in these headers is written for that native
// Vec<T>.
//
// A larger N, a whole multiple K of the native count, holds K native vectors,
// part[0] .. part[K-1], in lane order: lane i is lane i % L of part[i / L], and
// in memory the parts follow one another. Wide.hpp applies each operation to
// the parts. The use is mixed precision at equal width: Vec<double,
// lanes<float>> holds as many doubles as a Vec<float> holds floats.
//
// A partial specialization may not name VecTraits<T>::lanes as an argument, so
// the two layouts are chosen by a bool in the storage base class.
// ---------------------------------------------------------------------------

/// The native lane count of T: what Vec<T> holds in one register.
template <typename T>
inline constexpr int native_lanes_v = VecTraits<T>::lanes;

template <typename T, int N = VecTraits<T>::lanes>
    requires(N > 0 && N % VecTraits<T>::lanes == 0)
struct Vec;

namespace detail {
template <typename T, int N, bool Native = (N == VecTraits<T>::lanes)>
struct vec_storage;

/// One register.
template <typename T, int N>
struct vec_storage<T, N, true> {
    // VecTraits<T> is only specialized for the types we support on the
    // active ISA tier. Instantiating Vec<T> for an unsupported type
    // (e.g. int8_t, char, std::complex) gives an "incomplete type"
    // error pointing here.
    using traits   = VecTraits<T>;
    using reg_type = typename traits::reg_type;

    reg_type reg;

    vec_storage() = default;

    EINSUMS_FORCEINLINE vec_storage(reg_type r) : reg(r) {}

    EINSUMS_FORCEINLINE operator reg_type() const { return reg; }

    /// Element access for debugging only. Stores to a temporary buffer, then indexes it.
    EINSUMS_FORCEINLINE T operator[](int i) const {
        alignas(native_alignment) T buf[N];
        std::memcpy(buf, &reg, sizeof(reg));
        return buf[i];
    }
};

/// K native vectors.
template <typename T, int N>
struct vec_storage<T, N, false> {
    using part_type = Vec<T>;

    static constexpr int parts = N / VecTraits<T>::lanes;

    std::array<Vec<T>, parts> part;

    /// Element access for debugging only.
    EINSUMS_FORCEINLINE T operator[](int i) const { return part[i / VecTraits<T>::lanes][i % VecTraits<T>::lanes]; }
};
} // namespace detail

template <typename T, int N>
    requires(N > 0 && N % VecTraits<T>::lanes == 0)
struct Vec : detail::vec_storage<T, N> {
    using detail::vec_storage<T, N>::vec_storage;
    using value_type = T; ///< The element type, as einsums::simd::scalar_t<Vec<T, N>> reads it.

    static constexpr int  lanes  = N;
    static constexpr int  bits   = N * static_cast<int>(sizeof(T)) * 8;
    static constexpr bool native = N == VecTraits<T>::lanes; ///< Whether this is one register.

    Vec() = default;
};

/// Convenience alias for the number of lanes in a native Vec<T>.
template <typename T>
inline constexpr int lanes = Vec<T>::lanes;

EINSUMS_SIMD_ISA_NAMESPACE_END()
EINSUMS_NAMESPACE_END(simd)
