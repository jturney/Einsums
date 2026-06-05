//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>

namespace einsums::simd {

// ---------------------------------------------------------------------------
// Feature detection — constexpr booleans for each ISA extension.
// These are independent: e.g. has_avx2 and has_sse42 are both true on AVX2 hw.
// ---------------------------------------------------------------------------

// ---- x86 features ----

inline constexpr bool has_sse2 =
#if defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
    true;
#else
    false;
#endif

inline constexpr bool has_ssse3 =
#if defined(__SSSE3__)
    true;
#else
    false;
#endif

inline constexpr bool has_sse41 =
#if defined(__SSE4_1__)
    true;
#else
    false;
#endif

inline constexpr bool has_sse42 =
#if defined(__SSE4_2__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx =
#if defined(__AVX__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx2 =
#if defined(__AVX2__)
    true;
#else
    false;
#endif

inline constexpr bool has_fma =
#if defined(__FMA__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512 =
#if defined(__AVX512F__) && defined(__AVX512VL__)
    true;
#else
    false;
#endif

// AVX-10 (Intel's consolidation of AVX2 + AVX-512). Two width tiers:
//   - AVX-10/256: all AVX-512 *instructions* at 256-bit max width. Ships
//     on consumer chips that won't carry full 512-bit AVX-512 silicon
//     (Granite Rapids client, Panther Lake). These chips define
//     `__AVX__`/`__AVX2__` AND `__AVX10_1_256__`, but NOT `__AVX512F__`,
//     so they naturally route through the AVX/AVX2 tier in Operations
//     and Shuffle.
//   - AVX-10/512: full 512-bit width, conceptually a rebrand of the
//     AVX-512 family. These chips define `__AVX512F__` *and* the
//     AVX-10/512 macros, so they already hit the existing AVX-512 tier.
//
// Future enhancement (not in this round): on AVX-10/256-only chips, we
// could opt into the EVEX-encoded AVX-512 instruction set at 256-bit
// width (masked ops, embedded broadcast). The existing AVX2 path is
// correct as-is — these flags expose the additional capability for
// callers who want to hand-write specialized kernels.
inline constexpr bool has_avx10_1 =
#if defined(__AVX10_1__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_2 =
#if defined(__AVX10_2__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_256 =
#if defined(__AVX10_1_256__) || defined(__AVX10_2_256__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_512 =
#if defined(__AVX10_1_512__) || defined(__AVX10_2_512__)
    true;
#else
    false;
#endif

// VNNI (Vector Neural Network Instructions) — int8 dot-product accelerators.
//   - `has_avx_vnni`     ⇒ Alder Lake+ consumer chips, 256-bit dpbusd_epi32.
//   - `has_avx512_vnni`  ⇒ Cascade Lake-X+ server, 512-bit dpbusd at full
//     AVX-512 width. Both add the fused unsigned×signed → int32
//     accumulator used by quantized ML kernels.
inline constexpr bool has_avx_vnni =
#if defined(__AVXVNNI__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512_vnni =
#if defined(__AVX512VNNI__)
    true;
#else
    false;
#endif

// ---- ARM features ----

inline constexpr bool has_neon =
#if defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_fp16 =
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_bf16 =
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_i8mm =
#if defined(__ARM_FEATURE_MATMUL_INT8)
    true;
#else
    false;
#endif

// FEAT_DOTPROD — vdotq_s32/_u32. Default-on for M1+ and most modern
// aarch64 server chips; gates the same-sign int8 dot-product helpers.
inline constexpr bool has_neon_dotprod =
#if defined(__ARM_FEATURE_DOTPROD)
    true;
#else
    false;
#endif

// ---- x86 half-precision features (AVX-512 sub-extensions) ----
//
// `has_avx512_fp16` ⇒ Sapphire Rapids+ (consumer: not yet shipping).
// `has_avx512_bf16` ⇒ Cooper Lake+ (server) and Granite Rapids client.
// Both gate native arithmetic on the respective half-format; without
// them, kernels must round-trip through FP32.

inline constexpr bool has_avx512_fp16 =
#if defined(__AVX512FP16__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512_bf16 =
#if defined(__AVX512BF16__)
    true;
#else
    false;
#endif

inline constexpr bool is_apple_silicon =
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__APPLE__)
    true;
#else
    false;
#endif

// ---------------------------------------------------------------------------
// Native register width — widest available ISA determines this.
// Used to set Vec<T>::lanes and blocking parameters.
// ---------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512VL__)
inline constexpr int native_bits = 512;
#elif defined(__AVX__)
inline constexpr int native_bits = 256;
#elif defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
                inline constexpr int native_bits = 128;
#elif defined(__aarch64__) || defined(_M_ARM64)
                inline constexpr int native_bits = 128;
#else
                inline constexpr int native_bits = 0; // scalar fallback
#endif

inline constexpr int    native_bytes     = (native_bits > 0) ? (native_bits / 8) : 0;
inline constexpr size_t native_alignment = (native_bytes > 0) ? static_cast<size_t>(native_bytes) : alignof(double);

/// Number of elements of type T that fit in one native SIMD register.
/// Returns 1 for scalar fallback so loop counts remain valid.
template <typename T>
inline constexpr int native_lanes = (native_bits > 0) ? (native_bits / (8 * static_cast<int>(sizeof(T)))) : 1;

} // namespace einsums::simd
