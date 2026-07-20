//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * \file
 * Compute the tensor transposition
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Assert.hpp>
#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Files.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Macros.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Prefetch.hpp>
#include <Einsums/SIMD/Shuffle.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include "TransposeImpl.hpp"

namespace hptt {
namespace EINSUMS_SIMD_ARCH_NS {

// std::abs has no overload for __fp16 / __bf16, so promote those to float
// before calling. Real and complex types pass straight through.
namespace detail_hptt {
template <typename T>
EINSUMS_FORCEINLINE auto abs_promoted(T x) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
    if constexpr (std::is_same_v<T, einsums::simd::half_t>) {
        return std::abs(static_cast<float>(x));
    } else
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
        if constexpr (std::is_same_v<T, einsums::simd::bfloat16_t>) {
        return std::abs(static_cast<float>(x));
    } else
#endif
    {
        return std::abs(x);
    }
}
} // namespace detail_hptt

// ---------------------------------------------------------------------------
// Generic scalar micro_kernel: used for complex types and as fallback.
// ---------------------------------------------------------------------------
template <typename floatType, bool betaIsZero, bool conjA>
struct MicroKernel {
    static void execute(floatType const *A, size_t const lda, size_t const innerStrideA, floatType *B, size_t const ldb,
                        size_t const innerStrideB, floatType const alpha, floatType const beta) {
        constexpr size_t n = einsums::simd::native_lanes<floatType>;

        if constexpr (betaIsZero) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    if constexpr (conjA)
                        B[(i * innerStrideB) + (j * ldb)] = alpha * conj(A[(j * innerStrideA) + (lda * i)]);
                    else
                        B[(i * innerStrideB) + (j * ldb)] = alpha * A[(j * innerStrideA) + (lda * i)];
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    if constexpr (conjA) {
                        B[(i * innerStrideB) + (j * ldb)] =
                            alpha * conj(A[(j * innerStrideA) + (lda * i)]) + beta * B[(i * innerStrideB) + (j * ldb)];
                    } else {
                        B[(i * innerStrideB) + (j * ldb)] =
                            alpha * A[(j * innerStrideA) + (lda * i)] + beta * B[(i * innerStrideB) + (j * ldb)];
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// SIMD micro_kernel implementation shared by float and double.
// Factored as a helper to avoid duplicating the gather → transpose → scale →
// fmadd → scatter pipeline for each type.
//
// When innerStride is 1 (the common case for contiguous data), we use
// gather_fixed<1> / scatter_fixed<1> which compile down to a single
// loadu/storeu with no branch overhead. For other strides, we fall back
// to the runtime gather/scatter which selects the best available
// instruction (AVX2 hardware gather, NEON structured loads, or scalar).
// ---------------------------------------------------------------------------
namespace detail_hptt {

template <typename T, bool betaIsZero>
static EINSUMS_FORCEINLINE void micro_kernel_simd(T const *A, size_t lda, size_t innerStrideA, T *B, size_t ldb, size_t innerStrideB,
                                                  T alpha, T beta) {
    using namespace einsums::simd;
    constexpr int N = Vec<T>::lanes;

    auto va = broadcast(alpha);

    Vec<T> rows[N]; // NOLINT

    // Load A rows: fast path for stride==1
    if (innerStrideA == 1) {
        for (int i = 0; i < N; ++i)
            rows[i] = gather_fixed<1>(A + i * lda);
    } else {
        for (int i = 0; i < N; ++i)
            rows[i] = gather(A + i * lda, static_cast<std::ptrdiff_t>(innerStrideA));
    }

    transpose_inplace(rows);

    for (int i = 0; i < N; ++i)
        rows[i] = rows[i] * va;

    if constexpr (!betaIsZero) {
        auto vb = broadcast(beta);
        if (innerStrideB == 1) {
            for (int i = 0; i < N; ++i) {
                auto rowB = gather_fixed<1>(B + i * ldb);
                rows[i]   = fmadd(rowB, vb, rows[i]);
            }
        } else {
            for (int i = 0; i < N; ++i) {
                auto rowB = gather(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB));
                rows[i]   = fmadd(rowB, vb, rows[i]);
            }
        }
    }

    if (innerStrideB == 1) {
        for (int i = 0; i < N; ++i)
            scatter_fixed<1>(B + i * ldb, rows[i]);
    } else {
        for (int i = 0; i < N; ++i)
            scatter(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB), rows[i]);
    }
}

} // namespace detail_hptt

// ---------------------------------------------------------------------------
// SIMD-accelerated micro_kernel for float and double (non-complex).
// Uses einsums::simd for portable SIMD across x86 (SSE2/AVX/AVX2/AVX-512)
// and ARM NEON (including Apple Silicon).
// ---------------------------------------------------------------------------
template <bool betaIsZero, bool conjA>
struct MicroKernel<float, betaIsZero, conjA> {
    static void execute(float const *A, size_t const lda, size_t const innerStrideA, float *B, size_t const ldb, size_t const innerStrideB,
                        float const alpha, float const beta) {
        detail_hptt::micro_kernel_simd<float, betaIsZero>(A, lda, innerStrideA, B, ldb, innerStrideB, alpha, beta);
    }
};

template <bool betaIsZero, bool conjA>
struct MicroKernel<double, betaIsZero, conjA> {
    static void execute(double const *A, size_t const lda, size_t const innerStrideA, double *B, size_t const ldb,
                        size_t const innerStrideB, double const alpha, double const beta) {
        detail_hptt::micro_kernel_simd<double, betaIsZero>(A, lda, innerStrideA, B, ldb, innerStrideB, alpha, beta);
    }
};

// ---------------------------------------------------------------------------
// SIMD-accelerated micro_kernel for half_t (FP16). Reuses the float/double
// pipeline since Vec<half_t> has full broadcast/gather/transpose/multiply
// support on NEON FP16 and AVX-512FP16.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template <bool betaIsZero, bool conjA>
struct MicroKernel<einsums::simd::half_t, betaIsZero, conjA> {
    using half_t = einsums::simd::half_t;
    static void execute(half_t const *A, size_t const lda, size_t const innerStrideA, half_t *B, size_t const ldb,
                        size_t const innerStrideB, half_t const alpha, half_t const beta) {
        detail_hptt::micro_kernel_simd<half_t, betaIsZero>(A, lda, innerStrideA, B, ldb, innerStrideB, alpha, beta);
    }
};
#endif

// ---------------------------------------------------------------------------
// micro_kernel for bfloat16_t. BF16 has SIMD load/store but no Vec<bf16>×Vec<bf16>
// multiply that returns BF16; every native arithmetic instruction
// (vbfmla*, vbfdot, vmulq_f32 of converted halves) lands in FP32. So we
// load BF16 vectors, transpose 8×8 in-register, then for each output row
// convert BF16→FP32 (two halves), do alpha·A (+ beta·B) in FP32, and pack
// FP32→BF16 back into one 8-lane vector.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <bool betaIsZero, bool conjA>
struct MicroKernel<einsums::simd::bfloat16_t, betaIsZero, conjA> {
    using bf16_t = einsums::simd::bfloat16_t;
    static void execute(bf16_t const *A, size_t const lda, size_t const innerStrideA, bf16_t *B, size_t const ldb,
                        size_t const innerStrideB, bf16_t const alpha, bf16_t const beta) {
        using namespace einsums::simd;
        constexpr int N = Vec<bf16_t>::lanes; // 8

        Vec<bf16_t> rows[N]; // NOLINT
        if (innerStrideA == 1) {
            for (int i = 0; i < N; ++i)
                rows[i] = loadu(A + i * lda);
        } else {
            for (int i = 0; i < N; ++i)
                rows[i] = gather(A + i * lda, static_cast<std::ptrdiff_t>(innerStrideA));
        }

        transpose_inplace(rows);

        float32x4_t const va = vdupq_n_f32(static_cast<float>(alpha));
        float32x4_t       vb{};
        if constexpr (!betaIsZero) {
            vb = vdupq_n_f32(static_cast<float>(beta));
        }

        for (int i = 0; i < N; ++i) {
            float32x4_t lo = vmulq_f32(vcvtq_low_f32_bf16(rows[i].reg), va);
            float32x4_t hi = vmulq_f32(vcvtq_high_f32_bf16(rows[i].reg), va);

            if constexpr (!betaIsZero) {
                bfloat16x8_t b_row;
                if (innerStrideB == 1) {
                    b_row = vld1q_bf16(reinterpret_cast<__bf16 const *>(B + i * ldb));
                } else {
                    b_row = gather(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB)).reg;
                }
                lo = vfmaq_f32(lo, vcvtq_low_f32_bf16(b_row), vb);
                hi = vfmaq_f32(hi, vcvtq_high_f32_bf16(b_row), vb);
            }

            // Pack two float32x4 → one bfloat16x8: low half from `lo`, high half from `hi`.
            bfloat16x8_t result = vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(lo), hi);

            if (innerStrideB == 1) {
                vst1q_bf16(reinterpret_cast<__bf16 *>(B + i * ldb), result);
            } else {
                Vec<bf16_t> out;
                out.reg = result;
                scatter(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB), out);
            }
        }
    }
};
#elif defined(__AVX512BF16__)
// AVX-512BF16 platforms keep the FP32-promoted scalar path; equivalent
// SIMD work would use _mm512_cvtne2ps_pbh / _mm512_cvtpbh_ps but the
// hardware isn't available in this dev env to validate.
template <bool betaIsZero, bool conjA>
struct MicroKernel<einsums::simd::bfloat16_t, betaIsZero, conjA> {
    using bf16_t = einsums::simd::bfloat16_t;
    static void execute(bf16_t const *A, size_t const lda, size_t const innerStrideA, bf16_t *B, size_t const ldb,
                        size_t const innerStrideB, bf16_t const alpha, bf16_t const beta) {
        constexpr size_t n     = einsums::simd::native_lanes<bf16_t>;
        float const      a_f32 = static_cast<float>(alpha);
        float const      b_f32 = betaIsZero ? 0.0f : static_cast<float>(beta);

        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {
                float const a_val = static_cast<float>(A[(j * innerStrideA) + (lda * i)]);
                float       out   = a_f32 * a_val;
                if constexpr (!betaIsZero) {
                    float const b_val = static_cast<float>(B[(i * innerStrideB) + (j * ldb)]);
                    out += b_f32 * b_val;
                }
                B[(i * innerStrideB) + (j * ldb)] = static_cast<bf16_t>(out);
            }
        }
    }
};
#endif

// ---------------------------------------------------------------------------
// SIMD-accelerated micro_kernel for complex<float>
// ---------------------------------------------------------------------------
template <bool betaIsZero, bool conjA>
struct MicroKernel<std::complex<float>, betaIsZero, conjA> {
    static void execute(std::complex<float> const *A, size_t const lda, size_t const innerStrideA, std::complex<float> *B, size_t const ldb,
                        size_t const innerStrideB, std::complex<float> const alpha, std::complex<float> const beta) {
        using namespace einsums::simd;
        constexpr int N = CVec<float>::complex_lanes;

        auto va = complex_broadcast(alpha);

        CVec<float> rows[N]; // NOLINT
        for (int i = 0; i < N; ++i)
            rows[i] = complex_gather(A + i * lda, static_cast<std::ptrdiff_t>(innerStrideA));

        complex_transpose_inplace(rows);

        // Optionally conjugate A
        if constexpr (conjA) {
            for (int i = 0; i < N; ++i)
                rows[i] = conjugate(rows[i]);
        }

        // Scale by alpha (complex multiply)
        for (int i = 0; i < N; ++i)
            rows[i] = complex_mul(va, rows[i]);

        if constexpr (!betaIsZero) {
            auto vb = complex_broadcast(beta);
            for (int i = 0; i < N; ++i) {
                auto rowB = complex_gather(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB));
                rows[i]   = complex_add(complex_mul(vb, rowB), rows[i]);
            }
        }

        for (int i = 0; i < N; ++i)
            complex_scatter(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB), rows[i]);
    }
};

// ---------------------------------------------------------------------------
// SIMD-accelerated micro_kernel for complex<double>
// ---------------------------------------------------------------------------
template <bool betaIsZero, bool conjA>
struct MicroKernel<std::complex<double>, betaIsZero, conjA> {
    static void execute(std::complex<double> const *A, size_t const lda, size_t const innerStrideA, std::complex<double> *B,
                        size_t const ldb, size_t const innerStrideB, std::complex<double> const alpha, std::complex<double> const beta) {
        using namespace einsums::simd;
        constexpr int N = CVec<double>::complex_lanes;

        auto va = complex_broadcast(alpha);

        CVec<double> rows[N]; // NOLINT
        for (int i = 0; i < N; ++i)
            rows[i] = complex_gather(A + i * lda, static_cast<std::ptrdiff_t>(innerStrideA));

        complex_transpose_inplace(rows);

        if constexpr (conjA) {
            for (int i = 0; i < N; ++i)
                rows[i] = conjugate(rows[i]);
        }

        for (int i = 0; i < N; ++i)
            rows[i] = complex_mul(va, rows[i]);

        if constexpr (!betaIsZero) {
            auto vb = complex_broadcast(beta);
            for (int i = 0; i < N; ++i) {
                auto rowB = complex_gather(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB));
                rows[i]   = complex_add(complex_mul(vb, rowB), rows[i]);
            }
        }

        for (int i = 0; i < N; ++i)
            complex_scatter(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB), rows[i]);
    }
};

// ---------------------------------------------------------------------------
// streamingStore and prefetch: now use einsums::simd
// ---------------------------------------------------------------------------
template <typename floatType>
static void streamingStore(floatType *out, floatType const *in) {
    using namespace einsums::simd;
    if constexpr (std::is_floating_point_v<floatType>) {
        // Real f32/f64: native non-temporal store on x86, STNP on aarch64.
        stream_store(out, loadu(in));
    } else if constexpr (std::is_same_v<floatType, std::complex<float>> || std::is_same_v<floatType, std::complex<double>>) {
        // Complex: SIMD load+store. (No non-temporal complex variant yet.)
        complex_storeu(out, complex_loadu(in));
    } else {
        // half_t / bfloat16_t: non-temporal store on aarch64, regular SIMD store elsewhere.
        stream_store(out, loadu(in));
    }
}

template <typename floatType, einsums::simd::PrefetchHint Hint = einsums::simd::PrefetchHint::T2>
static EINSUMS_FORCEINLINE void prefetch_block(floatType const *A, size_t const lda) {
    constexpr int n = einsums::simd::native_bits / 8 / sizeof(floatType);
    for (int i = 0; i < n; ++i)
        einsums::simd::prefetch<Hint>(A + i * lda);
}
template <bool betaIsZero, typename floatType, bool conjA>
static EINSUMS_FORCEINLINE void macro_kernel_scalar(floatType const *A, size_t const lda, int blockingA, size_t innerStrideA, floatType *B,
                                                    size_t const ldb, int blockingB, size_t innerStrideB, floatType const alpha,
                                                    floatType const beta) {
    EINSUMS_ASSERT(blockingA > 0 && blockingB > 0);

    if constexpr (betaIsZero) {
        for (int j = 0; j < blockingA; ++j) {
            for (int i = 0; i < blockingB; ++i) {
                if (conjA)
                    B[(i * innerStrideB) + (j * ldb)] = alpha * conj(A[(i * lda) + (j * innerStrideA)]);
                else
                    B[(i * innerStrideB) + (j * ldb)] = alpha * A[(i * lda) + (j * innerStrideA)];
            }
        }
    } else {
        for (int j = 0; j < blockingA; ++j) {
            for (int i = 0; i < blockingB; ++i) {
                if (conjA)
                    B[(i * innerStrideB) + (j * ldb)] =
                        alpha * conj(A[(i * lda) + (j * innerStrideA)]) + beta * B[(i * innerStrideB) + (j * ldb)];
                else
                    B[(i * innerStrideB) + (j * ldb)] =
                        alpha * A[(i * lda) + (j * innerStrideA)] + beta * B[(i * innerStrideB) + (j * ldb)];
            }
        }
    }
}

template <int blockingA, int blockingB, bool betaIsZero, typename floatType, bool useStreamingStores_, bool conjA>
static EINSUMS_FORCEINLINE void macro_kernel(floatType const *A, floatType const *Anext, size_t const lda, size_t innerStrideA,
                                             floatType *B, floatType const *Bnext, size_t const ldb, size_t innerStrideB,
                                             floatType const alpha, floatType const beta) {
    constexpr int blocking_micro_ = einsums::simd::native_bits / 8 / sizeof(floatType);
    constexpr int blocking_       = blocking_micro_ * 4;

    // Non-temporal SIMD stores (stream_store -> _mm{,256,512}_stream_p{s,d}) require the
    // destination to be aligned to the vector width of the compiled rung: 16 B (SSE),
    // 32 B (AVX/AVX2), 64 B (AVX-512). A hardcoded 32-byte check silently under-aligns the
    // AVX-512 rung, which then #GP-faults (SEGV) or corrupts the result. Gate on the actual
    // native width so each rung only streams to a sufficiently aligned B / row stride.
    constexpr size_t stream_align       = einsums::simd::native_bits / 8;
    bool const       useStreamingStores = useStreamingStores_ && betaIsZero && (blockingB * sizeof(floatType)) % stream_align == 0 &&
                                    ((uint64_t)B) % stream_align == 0 && (ldb * sizeof(floatType)) % stream_align == 0;

    floatType *Btmp    = B;
    size_t     ldb_tmp = ldb;
    floatType  buffer[blockingA * blockingB]; // __attribute__((aligned(64)));
    if ((useStreamingStores_ && useStreamingStores && innerStrideB == 1)) {
        Btmp    = buffer;
        ldb_tmp = blockingB;
    }

    if constexpr (blockingA == blocking_ && blockingB == blocking_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (0 * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (2 * blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (2 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (3 * blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
                                                                            ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (0 * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (blocking_micro_ * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
                                                                            ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (2 * blocking_micro_ * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (3 * blocking_micro_ * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
                                                                            ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == 2 * blocking_micro_ && blockingB == blocking_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (0 * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (2 * blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (2 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
                                                                            ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == blocking_ && blockingB == 2 * blocking_micro_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (0 * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                           Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (0 * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch_block<floatType>(Anext + (blocking_micro_ * lda + 2 * blocking_micro_), lda);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, einsums::simd::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else {
        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 0)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A, lda, innerStrideA, Btmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + blocking_micro_ * lda, lda, innerStrideA,
                                                               Btmp + (innerStrideB * blocking_micro_), ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 2 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + 2 * blocking_micro_ * lda, lda, innerStrideA,
                                                               Btmp + (innerStrideB * 2 * blocking_micro_), ldb_tmp, innerStrideB, alpha,
                                                               beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 3 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + 3 * blocking_micro_ * lda, lda, innerStrideA,
                                                               Btmp + (innerStrideB * 3 * blocking_micro_), ldb_tmp, innerStrideB, alpha,
                                                               beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 0)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * blocking_micro_), lda, innerStrideA,
                                                               Btmp + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 2 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 3 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 0)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * 2 * blocking_micro_), lda, innerStrideA,
                                                               Btmp + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 2 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 3 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 0)
            MicroKernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * 3 * blocking_micro_), lda, innerStrideA,
                                                               Btmp + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 2 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 3 * blocking_micro_)
            MicroKernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);
    }

    // write buffer to main-memory via non-temporal stores
    if ((useStreamingStores_ && useStreamingStores && innerStrideB == 1)) {
        for (int i = 0; i < blockingA; i++) {
            for (int j = 0; j < blockingB; j += blocking_micro_)
                streamingStore<floatType>(B + i * ldb + j, buffer + i * ldb_tmp + j);
        }
    }
}

template <bool betaIsZero, typename floatType, bool conjA>
void transpose_int_scalar(floatType const *A, size_t sizeStride1A, size_t innerStrideA, floatType *B, size_t sizeStride1B, // NOLINT
                          size_t innerStrideB, floatType const alpha, floatType const beta, ComputeNode const *plan) {
    ptrdiff_t const end       = plan->end;
    size_t const    lda       = plan->lda;
    size_t const    ldb       = plan->ldb;
    ptrdiff_t const offDiffAB = plan->offDiffAB;
    if (plan->next->next != nullptr) {
        // recurse
        ptrdiff_t i = plan->start;
        if (plan->indexA)
            transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], end - plan->start, innerStrideA, &B[i * ldb],
                                                               sizeStride1B, innerStrideB, alpha, beta, plan->next.get());
        else if (plan->indexB)
            transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], sizeStride1A, innerStrideA, &B[i * ldb],
                                                               end - plan->start, innerStrideB, alpha, beta, plan->next.get());
        else
            for (; i < end; i++)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], sizeStride1A, innerStrideA, &B[i * ldb],
                                                                   sizeStride1B, innerStrideB, alpha, beta, plan->next.get());
    } else {
        // macro-kernel
        size_t const    lda_macro       = plan->next->lda;
        size_t const    ldb_macro       = plan->next->ldb;
        ptrdiff_t       i               = plan->start;
        ptrdiff_t const scalarRemainder = plan->end - plan->start;
        if (scalarRemainder > 0) {
            if (lda == 1)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, scalarRemainder, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, sizeStride1B, innerStrideB, alpha, beta);
            else if (ldb == 1)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, sizeStride1A, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, scalarRemainder, innerStrideB, alpha, beta);
            else
                for (; i < end; i++)
                    macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, sizeStride1A, innerStrideA,
                                                                      &B[i * ldb], ldb_macro, sizeStride1B, innerStrideB, alpha, beta);
        }
    }
}
template <int blockingA, int blockingB, bool betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int(floatType const *A, floatType const *Anext, size_t innerStrideA, floatType *B, floatType const *Bnext, // NOLINT
                   size_t innerStrideB, floatType const alpha, floatType const beta, ComputeNode const *plan) {
    ptrdiff_t const end       = plan->end - (plan->inc - 1);
    ptrdiff_t const inc       = plan->inc;
    size_t const    lda       = plan->lda;
    size_t const    ldb       = plan->ldb;
    int32_t const   offDiffAB = plan->offDiffAB;

    constexpr int blocking_micro_ = einsums::simd::native_bits / 8 / sizeof(floatType);
    constexpr int blocking_       = blocking_micro_ * 4;

    if (plan->next->next != nullptr) {
        // recurse
        ptrdiff_t i;
        for (i = plan->start; i < end; i += inc) {
            if (i + inc < end)
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + 1 + offDiffAB) * lda], innerStrideA, &B[i * ldb], &B[(i + 1) * ldb], innerStrideB,
                    alpha, beta, plan->next.get());
            else if (i == plan->start || i + inc >= end)
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + offDiffAB) * lda], innerStrideA, &B[i * ldb], &B[i * ldb], innerStrideB, alpha, beta,
                    plan->next.get());
            else
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next.get());
        }
        // remainder
        if (blocking_ / 2 >= blocking_micro_ && (i + blocking_ / 2) <= plan->end) {
            if (plan->indexA)
                transpose_int<blocking_ / 2, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next.get());
            else if (plan->indexB)
                transpose_int<blockingA, blocking_ / 2, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next.get());
            i += blocking_ / 2;
        }
        if (blocking_ / 4 >= blocking_micro_ && (i + blocking_ / 4) <= plan->end) {
            if (plan->indexA)
                transpose_int<blocking_ / 4, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next.get());
            else if (plan->indexB)
                transpose_int<blockingA, blocking_ / 4, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next.get());
            i += blocking_ / 4;
        }
        ptrdiff_t const scalarRemainder = plan->end - i;
        if (scalarRemainder > 0) {
            if (plan->indexA)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], scalarRemainder, innerStrideA, &B[i * ldb],
                                                                   blockingB, innerStrideB, alpha, beta, plan->next.get());
            else if (plan->indexB)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], blockingA, innerStrideA, &B[i * ldb],
                                                                   scalarRemainder, innerStrideB, alpha, beta, plan->next.get());
            else
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], blockingA, innerStrideA, &B[i * ldb],
                                                                   blockingB, innerStrideB, alpha, beta, plan->next.get());
        }
    } else {
        size_t const lda_macro = plan->next->lda;
        size_t const ldb_macro = plan->next->ldb;
        // invoke macro-kernel

        ptrdiff_t i;
        for (i = plan->start; i < end; i += inc)
            if (i + inc < end)
                macro_kernel<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + 1) * lda], lda_macro, innerStrideA, &B[i * ldb], &B[(i + 1) * ldb], ldb_macro,
                    innerStrideB, alpha, beta);
            else
                macro_kernel<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
        // remainder
        if (blocking_ / 2 >= blocking_micro_ && (i + blocking_ / 2) <= plan->end) {
            if (plan->indexA)
                macro_kernel<blocking_ / 2, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel<blockingA, blocking_ / 2, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            i += blocking_ / 2;
        }
        if (blocking_ / 4 >= blocking_micro_ && (i + blocking_ / 4) <= plan->end) {
            if (plan->indexA)
                macro_kernel<blocking_ / 4, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel<blockingA, blocking_ / 4, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            i += blocking_ / 4;
        }
        ptrdiff_t const scalarRemainder = plan->end - i;
        if (scalarRemainder > 0) {
            if (plan->indexA)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, scalarRemainder, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, blockingB, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, blockingA, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, scalarRemainder, innerStrideB, alpha, beta);
            else
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, blockingA, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, blockingB, innerStrideB, alpha, beta);
        }
    }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int_constStride1(floatType const *A, floatType *B, floatType const alpha, floatType const beta, // NOLINT
                                ComputeNode const *plan) {
    ptrdiff_t const end = plan->end - (plan->inc - 1);
    /// @todo Fix code.
    constexpr ptrdiff_t inc       = 1;
    size_t const        lda       = plan->lda;
    size_t const        ldb       = plan->ldb;
    ptrdiff_t const     offDiffAB = plan->offDiffAB;

    if (plan->next != nullptr) {
        for (ptrdiff_t i = plan->start; i < end; i += inc) {
            // recurse
            transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, conjA>(&A[(i + offDiffAB) * lda], &B[i * ldb], alpha,
                                                                                         beta, plan->next.get());
        }
    } else if constexpr (!betaIsZero) {
        for (ptrdiff_t i = plan->start; i < end; i += inc) {
            if constexpr (conjA)
                B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]) + beta * B[i * ldb];
            else
                B[i * ldb] = alpha * A[(i + offDiffAB) * lda] + beta * B[i * ldb];
        }
    } else {
        if constexpr (useStreamingStores) {
            if constexpr (conjA) {
#pragma vector nontemporal
                for (ptrdiff_t i = plan->start; i < end; i += inc) {
                    B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]);
                }
            } else {
#pragma vector nontemporal
                for (ptrdiff_t i = plan->start; i < end; i += inc) {
                    B[i * ldb] = alpha * A[(i + offDiffAB) * lda];
                }
            }
        } else if constexpr (conjA) {
            for (ptrdiff_t i = plan->start; i < end; i += inc) {
                B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]);
            }
        } else {
            for (ptrdiff_t i = plan->start; i < end; i += inc) {
                B[i * ldb] = alpha * A[(i + offDiffAB) * lda];
            }
        }
    }
}

template <typename floatType>
TransposeImpl<floatType>::TransposeImpl(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                        size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB,
                                        int const dim, floatType const *A, floatType const alpha, floatType *B, floatType const beta,
                                        SelectionMethod const selectionMethod, int const numThreads, int const *threadIds,
                                        bool const useRowMajor)
    : _A(A), _B(B), _alpha(alpha), _beta(beta), _dim(-1), _innerStrideA(0), _innerStrideB(0), _numThreads(numThreads), _masterPlan(nullptr),
      _selectionMethod(selectionMethod), _maxAutotuningCandidates(-1), _selectedParallelStrategyId(-1), _selectedLoopOrderId(-1),
      _conjA(false) {
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif

    std::vector<int>    tmpPerm(dim);
    std::vector<size_t> tmpSizeA(dim), tmpOuterSizeA(dim), tmpOuterSizeB(dim), tmpOffsetA(dim), tmpOffsetB(dim);

    account_for_row_major(sizeA, outerSizeA, outerSizeB, offsetA, offsetB, perm, tmpSizeA.data(), tmpOuterSizeA.data(),
                          tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(), tmpPerm.data(), dim, useRowMajor);

    _sizeA.resize(dim);
    _perm.resize(dim);
    _outerSizeA.resize(dim);
    _outerSizeB.resize(dim);
    _offsetA.resize(dim);
    _offsetB.resize(dim);
    _lda.resize(dim);
    _ldb.resize(dim);
    _threadIds.reserve(dim);
    if (threadIds) {
        // compact threadIds. E.g., 1, 7, 5 -> local_id(1) = 0, local_id(7) = 2,
        // local_id(5) = 1
        for (int i = 0; i < numThreads; ++i)
            _threadIds.push_back(threadIds[i]);
        std::sort(_threadIds.begin(), _threadIds.end());
    } else {
        for (int i = 0; i < numThreads; ++i)
            _threadIds.push_back(i);
    }

    verify_parameter(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(),
                     innerStrideA, innerStrideB, dim);

    _innerStrideA = innerStrideA;
    _innerStrideB = innerStrideB;

    // initializes dim_, outerSizeA, outerSizeB, sizeA and perm
    skip_indices(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(), dim);
    fuse_indices();

    // initializes lda_ and ldb_
    compute_leading_dimensions();

    // create plan
    this->create_plan();
}

template <typename floatType>
TransposeImpl<floatType>::TransposeImpl(TransposeImpl<floatType> const &other)
    : _A(other._A), _B(other._B), _alpha(other._alpha), _beta(other._beta), _dim(other._dim), _numThreads(other._numThreads),
      _masterPlan(other._masterPlan), _selectionMethod(other._selectionMethod),
      _selectedParallelStrategyId(other._selectedParallelStrategyId), _selectedLoopOrderId(other._selectedLoopOrderId),
      _maxAutotuningCandidates(other._maxAutotuningCandidates), _sizeA(other._sizeA), _perm(other._perm), _outerSizeA(other._outerSizeA),
      _outerSizeB(other._outerSizeB), _offsetA(other._offsetA), _offsetB(other._offsetB), _innerStrideA(other._innerStrideA),
      _innerStrideB(other._innerStrideB), _lda(other._lda), _ldb(other._ldb), _threadIds(other._threadIds), _conjA(other._conjA) {
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif
}

template <typename floatType>
TransposeImpl<floatType>::~TransposeImpl() {
#ifdef _OPENMP
    omp_destroy_lock(&_writelock);
#endif
}

template <typename floatType>
void TransposeImpl<floatType>::execute_estimate(Plan const *plan) noexcept {
    if (plan == nullptr) {
        EINSUMS_LOG_ERROR("HPTT: plan has not yet been created.");
        exit(-1);
    }

    constexpr bool useStreamingStores = false;

    int const numTasks = plan->get_num_tasks();
#ifdef _OPENMP
#    pragma omp parallel for num_threads(_numThreads) if (_numThreads > 1)
#endif
    for (int taskId = 0; taskId < numTasks; taskId++)
        if (_perm[0] != 0) {
            auto rootNode = plan->get_root_node(taskId);
            if (detail_hptt::abs_promoted(_beta) < get_zero_threshold<floatType>()) {
                if (_conjA)
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, true>(_A, _A, _innerStrideA, _B, _B,
                                                                                                _innerStrideB, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, false>(_A, _A, _innerStrideA, _B, _B,
                                                                                                 _innerStrideB, 0.0, 1.0, rootNode);
            } else {
                if (_conjA)
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, true>(_A, _A, _innerStrideA, _B, _B,
                                                                                                _innerStrideB, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, false>(_A, _A, _innerStrideA, _B, _B,
                                                                                                 _innerStrideB, 0.0, 1.0, rootNode);
            }
        } else {
            auto rootNode = plan->get_root_node(taskId);
            if (detail_hptt::abs_promoted(_beta) < get_zero_threshold<floatType>()) {
                if (_conjA)
                    transpose_int_constStride1<1, floatType, useStreamingStores, true>(_A, _B, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<1, floatType, useStreamingStores, false>(_A, _B, 0.0, 1.0, rootNode);
            } else {
                if (_conjA)
                    transpose_int_constStride1<0, floatType, useStreamingStores, true>(_A, _B, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<0, floatType, useStreamingStores, false>(_A, _B, 0.0, 1.0, rootNode);
            }
        }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_1D(floatType const *A, floatType *B, size_t const myStart, size_t const myEnd, ptrdiff_t const offDiffAB_,
                    size_t const lda, size_t const ldb, floatType const alpha, floatType const beta, int numThreads) {
    if constexpr (!betaIsZero) {
        HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if (conjA) B[i * ldb] =
                                         alpha * conj(A[(i + offDiffAB_) * lda]) + beta * B[i * ldb];
                       else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda] + beta * B[i * ldb];)
    } else {
        if constexpr (useStreamingStores)
            HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if constexpr (conjA) B[i * ldb] =
                                             alpha * conj(A[(i + offDiffAB_) * lda]);
                           else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda];)
        else
            HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if constexpr (conjA) B[i * ldb] =
                                             alpha * conj(A[(i + offDiffAB_) * lda]);
                           else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda];)
    }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_2D(floatType const *A, size_t const (&lda)[2], floatType *B, size_t const (&ldb)[2], size_t const n0, size_t const myStart,
                    size_t const myEnd, ptrdiff_t const offDiffAB_[2], size_t const offsetB_, floatType const alpha, floatType const beta,
                    int numThreads) {
    if constexpr (!betaIsZero) {
        HPTT_DUPLICATE(spawnThreads,
                       for (size_t j = myStart; j < myEnd; j++) for (size_t i = offsetB_; i < n0 + offsetB_; i++) if constexpr (conjA)
                           B[(i * ldb[0]) + j * ldb[1]] = alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]) +
                                                          beta * B[(i * ldb[0]) + j * ldb[1]];
                       else B[(i * ldb[0]) + j * ldb[1]] =
                           alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]] + beta * B[(i * ldb[0]) + j * ldb[1]];)
    } else {
        if constexpr (useStreamingStores)
            HPTT_DUPLICATE(spawnThreads, for (size_t j = myStart; j < myEnd; j++)
                                             _Pragma("vector nontemporal") for (size_t i = offsetB_; i < n0 + offsetB_;
                                                                                i++) if constexpr (conjA) B[(i * ldb[0]) + j * ldb[1]] =
                                                 alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]);
                           else B[(i * ldb[0]) + j * ldb[1]] = alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]];)
        else
            HPTT_DUPLICATE(spawnThreads, for (size_t j = myStart; j < myEnd; j++) for (size_t i = offsetB_; i < n0 + offsetB_;
                                                                                       i++) if (conjA) B[(i * ldb[0]) + j * ldb[1]] =
                                             alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]);
                           else B[(i * ldb[0]) + j * ldb[1]] = alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]];)
    }
}

template <typename floatType>
template <bool spawnThreads>
void TransposeImpl<floatType>::get_start_end(size_t n, size_t &myStart, size_t &myEnd) const {
#ifdef _OPENMP
    int myLocalThreadId = get_local_thread_id(omp_get_thread_num());
#else
    int myLocalThreadId = 0;
#endif

    if (myLocalThreadId == -1) // skip those threads which do not participate in this plan
    {
        myStart = n;
        myEnd   = n;
        return;
    }
    if constexpr (spawnThreads) { // worksharing will be handled by the OpenMP runtime
        myStart = 0;
        myEnd   = n;
        return;
    }

    size_t const workPerThread = (n + _numThreads - 1) / _numThreads;
    myStart                    = std::min(n, myLocalThreadId * workPerThread);
    myEnd                      = std::min(n, (myLocalThreadId + 1) * workPerThread);
}

template <typename floatType>
int TransposeImpl<floatType>::get_local_thread_id(int myThreadId) const {
    int myLocalId = -1;
    for (int i = 0; i < _numThreads; ++i)
        if (myThreadId == _threadIds[i])
            myLocalId = i;
    return myLocalId;
}

template <typename floatType>
template <bool useStreamingStores, bool spawnThreads, bool betaIsZero>
void TransposeImpl<floatType>::execute_expert() noexcept {
    if (_masterPlan == nullptr) {
        EINSUMS_LOG_ERROR("HPTT: master plan has not yet been created.");
        exit(-1);
    }

    size_t myStart = 0;
    size_t myEnd   = 0;

    if (_dim == 1) {
        get_start_end<spawnThreads>(_sizeA[0], myStart, myEnd);
        ptrdiff_t const offDiffAB_ = (ptrdiff_t)_offsetA[0] - (ptrdiff_t)_offsetB[0];
        if (_conjA)
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(
                _A, _B, myStart + _offsetB[0], myEnd + _offsetB[0], offDiffAB_, _lda[0], _ldb[0], _alpha, _beta, _numThreads);
        else
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(
                _A, _B, myStart + _offsetB[0], myEnd + _offsetB[0], offDiffAB_, _lda[0], _ldb[0], _alpha, _beta, _numThreads);
        return;
    } else if (_dim == 2 && _perm[0] == 0) {
        get_start_end<spawnThreads>(_sizeA[1], myStart, myEnd);
        ptrdiff_t const offDiffAB_[2] = {((ptrdiff_t)_offsetA[0] - (ptrdiff_t)_offsetB[0]),
                                         ((ptrdiff_t)_offsetA[1] - (ptrdiff_t)_offsetB[1])};
        if (_conjA)
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(_A, {_lda[0], _lda[1]}, _B, {_ldb[0], _ldb[1]},
                                                                                   _sizeA[0], myStart + _offsetB[1], myEnd + _offsetB[1],
                                                                                   offDiffAB_, _offsetB[0], _alpha, _beta, _numThreads);
        else
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(_A, {_lda[0], _lda[1]}, _B, {_ldb[0], _ldb[1]},
                                                                                    _sizeA[0], myStart + _offsetB[1], myEnd + _offsetB[1],
                                                                                    offDiffAB_, _offsetB[0], _alpha, _beta, _numThreads);
        return;
    }

    int const numTasks   = _masterPlan->get_num_tasks();
    int const numThreads = _numThreads;
    get_start_end<spawnThreads>(numTasks, myStart, myEnd);

    HPTT_DUPLICATE(
        spawnThreads,
        for (int taskId = myStart; taskId < myEnd; taskId++) if (_perm[0] != 0) {
            auto rootNode = _masterPlan->get_root_node(taskId);
            if (_conjA)
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, true>(
                    _A, _A, _innerStrideA, _B, _B, _innerStrideB, _alpha, _beta, rootNode);
            else
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, false>(
                    _A, _A, _innerStrideA, _B, _B, _innerStrideB, _alpha, _beta, rootNode);
        } else {
            auto rootNode = _masterPlan->get_root_node(taskId);
            if (_conjA)
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, true>(_A, _B, _alpha, _beta, rootNode);
            else
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, false>(_A, _B, _alpha, _beta, rootNode);
        })
}
template <typename floatType>
void TransposeImpl<floatType>::execute() noexcept {
    if (_masterPlan == nullptr) {
        EINSUMS_LOG_ERROR("HPTT: master plan has not yet been created.");
        exit(-1);
    }

    bool           spawnThreads       = _numThreads > 1;
    bool           betaIsZero         = (_beta == (floatType)0.0);
    constexpr bool useStreamingStores = true;
    if (spawnThreads) {
        if (betaIsZero) {
            this->execute_expert<useStreamingStores, true, true>();
        } else {
            this->execute_expert<useStreamingStores, true, false>();
        }
    } else {
        if (betaIsZero) {
            this->execute_expert<useStreamingStores, false, true>();
        } else {
            this->execute_expert<useStreamingStores, false, false>();
        }
    }
}

template <typename floatType>
void TransposeImpl<floatType>::print() noexcept {
    _masterPlan->print();
}

template <typename floatType>
size_t TransposeImpl<floatType>::get_increment(int loopIdx) const {
    size_t inc = 1;
    if (_perm[0] != 0) {
        if (loopIdx == 0 || loopIdx == _perm[0])
            inc = blocking_;
    }
    return inc;
}

template <typename floatType>
void TransposeImpl<floatType>::get_available_parallelism(std::vector<int> &numTasksPerLoop) const {
    numTasksPerLoop.resize(_dim);
    for (int loopIdx = 0; loopIdx < _dim; ++loopIdx) {
        size_t inc               = this->get_increment(loopIdx);
        numTasksPerLoop[loopIdx] = (_sizeA[loopIdx] + inc - 1) / inc;
    }
}

template <typename floatType>
void TransposeImpl<floatType>::get_all_parallelism_strategies(std::list<int>                &primeFactorsToMatch,
                                                              std::vector<int>              &availableParallelismAtLoop, // NOLINT
                                                              std::vector<int>              &achievedParallelismAtLoop,
                                                              std::vector<std::vector<int>> &parallelismStrategies) const {
    if (primeFactorsToMatch.size() > 0) {
        // match every primefactor ...
        for (auto p : primeFactorsToMatch) {
            // ... with every loop
            for (int i = 0; i < _dim; i++) {
                std::list<int>   primeFactorsToMatch_(primeFactorsToMatch);
                std::vector<int> availableParallelismAtLoop_(availableParallelismAtLoop);
                std::vector<int> achievedParallelismAtLoop_(achievedParallelismAtLoop);

                primeFactorsToMatch_.erase(std::find(primeFactorsToMatch_.begin(), primeFactorsToMatch_.end(), p));
                availableParallelismAtLoop_[i] = (availableParallelismAtLoop_[i] + p - 1) / p;
                achievedParallelismAtLoop_[i] *= p;

                this->get_all_parallelism_strategies(primeFactorsToMatch_, availableParallelismAtLoop_, achievedParallelismAtLoop_,
                                                     parallelismStrategies);
            }
        }
    } else {
        // avoid duplicates
        if (parallelismStrategies.end() == std::find(parallelismStrategies.begin(), parallelismStrategies.end(), achievedParallelismAtLoop))
            parallelismStrategies.push_back(achievedParallelismAtLoop);
    }
}

// balancing if one tries to parallelize avail many tasks with req many threads
// e.g., balancing(3,4) = 0.75
static float getBalancing(int avail, int req) {
    return ((float)(avail)) / (float)((int)((avail + req - 1) / req) * req);
}

template <typename floatType>
float TransposeImpl<floatType>::get_load_balance(std::vector<int> const &parallelismStrategy) const {
    float load_balance = 1.0;
    int   totalTasks   = 1;
    for (int i = 0; i < _dim; ++i) {

        size_t inc = this->get_increment(i);
        while (_sizeA[i] < inc)
            inc /= 2;
        size_t availableParallelism = (_sizeA[i] + inc - 1) / inc;

        if (i == 0 || _perm[i] == 0)
            // account for the load-imbalancing due to blocking
            load_balance *= getBalancing(_sizeA[i], inc);
        load_balance *= getBalancing(availableParallelism, parallelismStrategy[i]);
        totalTasks *= parallelismStrategy[i];
    }

    // how well can these tasks be distributed among _numThreads?
    //  e.g., totalTasks = 3, numThreads = 8 => 3./8
    //  e.g., totalTasks = 5, numThreads = 8 => 5./8
    //  e.g., totalTasks = 15, numThreads = 8 => 15./16
    //  e.g., totalTasks = 17, numThreads = 8 => 17./24
    float workDistribution = ((float)totalTasks) / (((totalTasks + _numThreads - 1) / _numThreads) * _numThreads);

    load_balance *= workDistribution;
    return load_balance;
}

template <typename floatType>
void TransposeImpl<floatType>::get_best_parallelism_strategy(std::vector<int> &bestParallelismStrategy) const {
    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);
    int totalAvailableParallelism =
        std::accumulate(availableParallelismAtLoop.begin(), availableParallelismAtLoop.end(), 1, std::multiplies<int>());

    // reduce the probability of parallelizing the stride-1 index
    // if this loops would be parallelized, these two statements ensure that each
    // thread would have at least two macro-kernels of work at this loop-level
    //
    // However, if the total available parallelism is too small, then we do not
    // artificially limit the available parallelism further
    int reduceParallelismB = 4; // avoid parallelization in stride-1 B more strongly
    int reduceParallelismA = 2;
    if (totalAvailableParallelism < 2 * _numThreads)
        reduceParallelismB = 1;
    else if (totalAvailableParallelism < 4 * _numThreads)
        reduceParallelismB = 2;
    totalAvailableParallelism =
        (totalAvailableParallelism / availableParallelismAtLoop[_perm[0]]) * (availableParallelismAtLoop[_perm[0]] / reduceParallelismB);
    if (totalAvailableParallelism < 2 * _numThreads)
        reduceParallelismA = 1;
    availableParallelismAtLoop[_perm[0]] = std::max(1, availableParallelismAtLoop[_perm[0]] / reduceParallelismB);
    availableParallelismAtLoop[0]        = std::max(1, availableParallelismAtLoop[0] / reduceParallelismA);

    // Objectives: 1) load-balancing
    //             2) avoid parallelizing stride-1 loops (rational: less
    //             consecutive memory accesses) 3) avoid false sharing

    std::vector<int> loopsAllowed;
    for (int i = _dim - 1; i >= 1; i--)
        if (_perm[i] != 0)
            loopsAllowed.push_back(_perm[i]);
    std::vector<int> loopsAllowedStride1{0, _perm[0]};

    int            totalTasks = 1; // goal: totalTasks should be a close multiple of numTasks_
    std::list<int> primeFactors;
    get_prime_factors(_numThreads, primeFactors);

    // 1. parallelize using 100% load balancing
    parallelize(bestParallelismStrategy, availableParallelismAtLoop, totalTasks, primeFactors, 1.0, loopsAllowed);

    if (totalTasks != _numThreads) { // no perfect match has been found

        // Option 1: keep parallelizing non-stride-1 loops only, but allowing
        // load-imbalance
        std::vector<int> strat1(bestParallelismStrategy);
        std::vector<int> avail1(availableParallelismAtLoop);
        std::list<int>   primes1(primeFactors);
        int              totalTasks1 = totalTasks;
        parallelize(strat1, avail1, totalTasks1, primes1, 0.92, loopsAllowed);
        if (get_load_balance(strat1) > 0.90) {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
        }

        if (_perm[0] != 0) {
            // Option 2: also parallelize stride-1 loops, enforcing perfect loop
            // balancing
            std::vector<int> strat2(bestParallelismStrategy);
            std::vector<int> avail2(availableParallelismAtLoop);
            std::list<int>   primes2(primeFactors);
            int              totalTasks2 = totalTasks;
            parallelize(strat2, avail2, totalTasks2, primes2, 1.0, loopsAllowedStride1);
            if (get_load_balance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat1
            parallelize(strat1, avail1, totalTasks1, primes1, 1.0, loopsAllowedStride1);
            if (get_load_balance(strat1) > 0.90) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat2
            parallelize(strat2, avail2, totalTasks2, primes2, 0.92, loopsAllowed);
            if (get_load_balance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            if (get_load_balance(strat1) > 0.80) // reduced threshold
            {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (get_load_balance(strat2) > 0.82) // reduced threshold
            {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            parallelize(strat1, avail1, totalTasks1, primes1, 0.9, loopsAllowedStride1);
            parallelize(strat2, avail2, totalTasks2, primes2, 0.8, loopsAllowed);
            float lb1 = get_load_balance(strat1);
            float lb2 = get_load_balance(strat2);
            //         printVector(strat2,"strat2");
            //         printf("strat2: %f\n",getLoadBalance(strat2));
            if ((lb1 > 0.8 && lb2 < 0.85) || (lb1 > lb2 && lb1 > 0.75)) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (lb2 >= 0.85) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // fallback
            std::vector<int> allLoops;
            for (int i = _dim - 1; i >= 1; i--)
                allLoops.push_back(_perm[i]);
            allLoops.push_back(0);
            allLoops.push_back(_perm[0]);
            parallelize(strat1, avail1, totalTasks1, primes1, 0., allLoops);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());

        } else {
            parallelize(strat1, avail1, totalTasks1, primes1, 0.0, loopsAllowed);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
        }
    }
}

template <typename floatType>
void TransposeImpl<floatType>::parallelize(std::vector<int> &parallelismStrategy, std::vector<int> &availableParallelismAtLoop,
                                           int &totalTasks, std::list<int> &primeFactors, float const minBalancing,
                                           std::vector<int> const &loopsAllowed) const

{
    bool suboptimalParallelizationUsed = false;
    // find loop which minimizes load imbalance for the given prime factor
    for (auto it = primeFactors.begin(); it != primeFactors.end(); it++) {
        int   suitedLoop    = -1;
        float bestBalancing = 0;

        for (auto idx : loopsAllowed) {
            float balancing = getBalancing(availableParallelismAtLoop[idx], *it);
            if (balancing > bestBalancing) {
                bestBalancing = balancing;
                suitedLoop    = idx;
            }
        }
        // allow up to one slightly less optimal splitting to prefer parallelizing
        // idx=0 over idx=perm[0]
        if (suboptimalParallelizationUsed == false && suitedLoop == _perm[0] && getBalancing(availableParallelismAtLoop[0], *it) >= 0.949) {
            suitedLoop                    = 0;
            suboptimalParallelizationUsed = true;
        }
        if (suitedLoop != -1 && bestBalancing >= minBalancing) {
            availableParallelismAtLoop[suitedLoop] /= *it;
            parallelismStrategy[suitedLoop] *= *it;
            totalTasks *= *it;
            it = primeFactors.erase(it);
            it--;
        }
    }
}

template <typename floatType>
double TransposeImpl<floatType>::parallelism_cost_heuristic(std::vector<int> const &achievedParallelismAtLoop) const {
    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);

    double cost = 1;
    // penalize load-imbalance
    for (int loopIdx = 0; loopIdx < _dim; ++loopIdx) {
        if (achievedParallelismAtLoop[loopIdx] <= 1)
            continue;

        int const blocksPerThread =
            (availableParallelismAtLoop[loopIdx] + achievedParallelismAtLoop[loopIdx] - 1) / achievedParallelismAtLoop[loopIdx];
        int       inc           = this->get_increment(loopIdx);
        int const effectiveSize = blocksPerThread * inc * achievedParallelismAtLoop[loopIdx];
        cost *= ((double)(effectiveSize) / _sizeA[loopIdx]);
    }

    // penalize parallelization of stride-1 loops
    if (_perm[0] == 0)
        cost *= std::pow(1.01, achievedParallelismAtLoop[0] - 1); // strongly penalize this case

    cost *= std::pow(1.00010, std::min(16, achievedParallelismAtLoop[0] - 1));        // if at all, prefer ...
    cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[_perm[0]] - 1)); // parallelization in stride-1 of A

    int const workPerThread =
        (availableParallelismAtLoop[_perm[0]] + achievedParallelismAtLoop[_perm[0]] - 1) / achievedParallelismAtLoop[_perm[0]];
    if (workPerThread * sizeof(floatType) % 64 != 0 && achievedParallelismAtLoop[_perm[0]] > 1) { // avoid false-sharing
        cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[_perm[0]] - 1));         // penalize this parallelization again
    }
    return cost;
}

template <typename floatType>
void TransposeImpl<floatType>::get_parallelism_strategies(std::vector<std::vector<int>> &parallelismStrategies) const {
    parallelismStrategies.clear();
    if (_numThreads == 1) {
        parallelismStrategies.emplace_back(std::vector<int>(_dim, 1));
        return;
    }
    std::vector<int> bestParallelismStrategy(_dim, 1);
    get_best_parallelism_strategy(bestParallelismStrategy);
    if (this->infoLevel_ > 0)
        EINSUMS_LOG_INFO("HPTT: loadbalancing: {}", get_load_balance(bestParallelismStrategy));

    if (_selectionMethod == ESTIMATE) {
        parallelismStrategies.push_back(bestParallelismStrategy);
        return;
    }

    // ATTENTION: we don't care about the case where _numThreads is a large prime
    // number... (sorry, KNC)
    //
    // we factorize numThreads into its prime factors because we have to match
    // every one to a certain loop. In principle every loop could be used to
    // match every primefactor, but some choices are preferable over others.
    // E.g., we want to achieve good load-balancing _and_ try to avoid the
    // stride-1 index of B (due to false sharing)
    std::list<int> primeFactors;
    get_prime_factors(_numThreads, primeFactors);
    if (this->infoLevel_ > 0)
        print_vector(primeFactors, "primes");

    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);
    if (this->infoLevel_ > 0)
        print_vector(availableParallelismAtLoop, "available Parallelism");

    std::vector<int> achievedParallelismAtLoop(_dim, 1);

    this->get_all_parallelism_strategies(primeFactors, availableParallelismAtLoop, achievedParallelismAtLoop, parallelismStrategies);

    // sort according to loop heuristic
    std::sort(parallelismStrategies.begin(), parallelismStrategies.end(),
              [this](std::vector<int> const &loopOrder1, std::vector<int> const &loopOrder2) {
                  return this->parallelism_cost_heuristic(loopOrder1) < this->parallelism_cost_heuristic(loopOrder2);
              });

    parallelismStrategies.insert(parallelismStrategies.begin(), bestParallelismStrategy);

    if (this->infoLevel_ > 1)
        for (auto const &strat : parallelismStrategies) {
            print_vector(strat, "parallelization");
            EINSUMS_LOG_INFO("HPTT: cost: {}", this->parallelism_cost_heuristic(strat));
        }
}

template <typename floatType>
void TransposeImpl<floatType>::verify_parameter(size_t const *size, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                                size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA,
                                                size_t const innerStrideB, int const dim) const {
    if (dim < 1) {
        EINSUMS_LOG_ERROR("HPTT: dimensionality too low.");
        exit(-1);
    }

    std::vector<int> found(dim, 0);

    for (int i = 0; i < dim; ++i) {
        if (size[i] <= 0) {
            EINSUMS_LOG_ERROR("HPTT: size at position {} is invalid", i);
            exit(-1);
        }
        found[perm[i]] = 1;
    }

    for (int i = 0; i < dim; ++i)
        if (found[i] <= 0) {
            EINSUMS_LOG_ERROR("HPTT: permutation invalid");
            exit(-1);
        }

    if (outerSizeA != nullptr)
        for (int i = 0; i < dim; ++i)
            if (outerSizeA[i] < size[i]) {
                EINSUMS_LOG_ERROR("HPTT: outerSizeA invalid");
                exit(-1);
            }

    if (outerSizeB != nullptr)
        for (int i = 0; i < dim; ++i)
            if (outerSizeB[i] < size[perm[i]]) {
                EINSUMS_LOG_ERROR("HPTT: outerSizeB invalid");
                exit(-1);
            }

    if (offsetA != nullptr)
        for (int i = 0; i < dim; ++i)
            if (offsetA[i] + size[i] > outerSizeA[i]) {
                EINSUMS_LOG_ERROR("HPTT: offsetA invalid");
                exit(-1);
            }

    if (offsetB != nullptr)
        for (int i = 0; i < dim; ++i)
            if (offsetB[i] + size[perm[i]] > outerSizeB[i]) {
                EINSUMS_LOG_ERROR("HPTT: offsetB invalid");
                exit(-1);
            }

    if (innerStrideA < 0) {
        EINSUMS_LOG_ERROR("HPTT: innerStrideA invalid");
        exit(-1);
    }

    if (innerStrideB < 0) {
        EINSUMS_LOG_ERROR("HPTT: innerStrideB invalid");
        exit(-1);
    }
}

template <typename floatType>
void TransposeImpl<floatType>::compute_leading_dimensions() {
    _lda[0] = _innerStrideA;
    if (_outerSizeA[0] == -1)
        for (int i = 1; i < _dim; ++i)
            _lda[i] = _lda[i - 1] * _sizeA[i - 1];
    else
        for (int i = 1; i < _dim; ++i)
            _lda[i] = _outerSizeA[i - 1] * _lda[i - 1];

    _ldb[0] = _innerStrideB;
    if (_outerSizeB[0] == -1)
        for (int i = 1; i < _dim; ++i)
            _ldb[i] = _ldb[i - 1] * _sizeA[_perm[i - 1]];
    else
        for (int i = 1; i < _dim; ++i)
            _ldb[i] = _outerSizeB[i - 1] * _ldb[i - 1];
}

template <typename floatType>
void TransposeImpl<floatType>::skip_indices(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                            size_t const *offsetA, size_t const *offsetB, int const dim) {
    for (int i = 0; i < dim; ++i) {
        _perm[i]  = perm[i];
        _sizeA[i] = sizeA[i];
        if (outerSizeA)
            _outerSizeA[i] = outerSizeA[i];
        else
            _outerSizeA[i] = sizeA[i];
        if (outerSizeB)
            _outerSizeB[i] = outerSizeB[i];
        else
            _outerSizeB[i] = sizeA[perm[i]];
        if (offsetA)
            _offsetA[i] = offsetA[i];
        else
            _offsetA[i] = 0;
        if (offsetB)
            _offsetB[i] = offsetB[i];
        else
            _offsetB[i] = 0;
    }

    size_t skipped = 0;
    for (int i = 0; i < dim; ++i) {
        int idxB = 0;
        for (; idxB < dim; ++idxB)
            if (perm[idxB] == i)
                break;
        if (sizeA[i] == 1 && (!outerSizeA || outerSizeA[i] == 1) && (!outerSizeB || outerSizeB[idxB] == 1)) {
            _sizeA[i]         = -1;
            _outerSizeA[i]    = -1;
            _outerSizeB[idxB] = -1;
            _offsetA[i]       = -1;
            _offsetB[idxB]    = -1;
            _perm[idxB]       = -1;
            skipped++;
        }
    }
    // compact arrays (remove -1)
    for (int i = 0; i < dim; ++i)
        if (_sizeA[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_sizeA[j] != -1)
                    break;
            if (j < dim)
                std::swap(_sizeA[i], _sizeA[j]);
        }
    for (int i = 0; i < dim; ++i)
        if (_outerSizeA[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_outerSizeA[j] != -1)
                    break;
            if (j < dim) {
                std::swap(_outerSizeA[i], _outerSizeA[j]);
                std::swap(_offsetA[i], _offsetA[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (_outerSizeB[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_outerSizeB[j] != -1)
                    break;
            if (j < dim) {
                std::swap(_outerSizeB[i], _outerSizeB[j]);
                std::swap(_offsetB[i], _offsetB[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (_perm[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_perm[j] != -1)
                    break;
            if (j < dim)
                std::swap(_perm[i], _perm[j]);
        }

    _dim = dim - skipped;
    if (_dim == 0) {
        _dim = 1;
        _perm.resize(_dim);
        _sizeA.resize(_dim);
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _perm[0]       = 0;
        _sizeA[0]      = 1;
        _outerSizeA[0] = 1;
        _outerSizeB[0] = 1;
        _offsetA[0]    = 0;
        _offsetB[0]    = 0;
    } else {
        _perm.resize(_dim);
        _sizeA.resize(_dim);
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _offsetA.resize(_dim);
        _offsetB.resize(_dim);

        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < _dim; ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < _dim; ++pos) {
                if (_perm[pos] >= currentValue && _perm[pos] < minValue) {
                    minValue = _perm[pos];
                    minPos   = pos;
                }
            }
            _perm[minPos] = currentValue; // minValue renamed to currentValue
            currentValue++;
        }
    }

    EINSUMS_LOG_DEBUG("HPTT: dim={}, innerStrideA={}, innerStrideB={}", _dim, _innerStrideA, _innerStrideB);
}

/**
 * \brief fuses indices whenever possible
 * \detailed For instance:
 *           perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 &
 * size=10,11*12,13 \return This function will initialize sizeA_, perm_,
 * outerSizeA_, outersize_ and dim_
 */
template <typename floatType>
void TransposeImpl<floatType>::fuse_indices() {
    std::list<std::tuple<int, int>> fusedIndices;

    std::vector<int> perm;
    // correct perm
    for (int i = 0; i < _dim; ++i) {
        // merge indices if the two consecutive entries are identical
        int toMerge = i;
        perm.push_back(_perm[i]);
        /* By definition if size == outerSize, then no offsets are present. However,
         *  by merging with the subsequent dimension the stride the offset depends upon
         *  is lost. Therefore, the offset of the next offset must be zero too! */
        while (i + 1 < _dim && _perm[i] + 1 == _perm[i + 1] && (_sizeA[_perm[i]] == _outerSizeA[_perm[i]]) &&
               (_sizeA[_perm[i]] == _outerSizeB[i]) && (_offsetA[_perm[i + 1]] == 0) && (_offsetB[i + 1] == 0)) {
            EINSUMS_LOG_DEBUG("HPTT: merging indices {} and {}", _perm[i], _perm[i + 1]);
            fusedIndices.emplace_back(std::make_tuple(_perm[toMerge], _perm[i + 1]));
            i++;
        }
    }

    // correct sizes and outer-sizes
    for (auto tup : fusedIndices) {
        _sizeA[std::get<0>(tup)] *= _sizeA[std::get<1>(tup)];
        _outerSizeA[std::get<0>(tup)] *= _outerSizeA[std::get<1>(tup)];
        _outerSizeA[std::get<1>(tup)] = -1;
        _offsetA[std::get<1>(tup)]    = -1;

        auto pos1 = std::find(_perm.begin(), _perm.end(), std::get<0>(tup)) - _perm.begin();
        auto pos2 = std::find(_perm.begin(), _perm.end(), std::get<1>(tup)) - _perm.begin();
        _outerSizeB[pos1] *= _outerSizeB[pos2];
        _outerSizeB[pos2] = -1;
        _offsetB[pos2]    = -1;
    }

    if (fusedIndices.size() > 0) {
        _perm = perm;
        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < _perm.size(); ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < _perm.size(); ++pos) {
                if (_perm[pos] >= currentValue && _perm[pos] < minValue) {
                    minValue = _perm[pos];
                    minPos   = pos;
                }
            }
            EINSUMS_LOG_DEBUG("HPTT: perm[{}]: {} -> {}", minPos, _perm[minPos], currentValue);
            _perm[minPos]        = currentValue; // minValue renamed to currentValue
            _sizeA[currentValue] = _sizeA[minValue];
            currentValue++;
        }

        // compact outer size (e.g.: outerSizeA_[] = {24,-1,5,-1,13} ->
        // {24,5,13,-1,-1} -> {24,5,13}
        for (int i = 0; i < _dim; ++i)
            if (_outerSizeA[i] == -1) {
                int j = i + 1;
                for (; j < _dim; ++j)
                    if (_outerSizeA[j] != -1)
                        break;
                if (j < _dim) {
                    std::swap(_outerSizeA[i], _outerSizeA[j]);
                    std::swap(_offsetA[i], _offsetA[j]);
                }
            }
        for (int i = 0; i < _dim; ++i)
            if (_outerSizeB[i] == -1) {
                int j = i + 1;
                for (; j < _dim; ++j)
                    if (_outerSizeB[j] != -1)
                        break;
                if (j < _dim) {
                    std::swap(_outerSizeB[i], _outerSizeB[j]);
                    std::swap(_offsetB[i], _offsetB[j]);
                }
            }
        _dim -= fusedIndices.size();
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _offsetA.resize(_dim);
        _offsetB.resize(_dim);
        _sizeA.resize(_dim);
        _perm.resize(_dim);

        EINSUMS_LOG_DEBUG("HPTT: after index fusion: dim={}", _dim);
    }
}

// returns the best loop order (same as the best one with exhaustive search)
template <typename floatType>
void TransposeImpl<floatType>::get_best_loop_order(std::vector<int> &loopOrder) const {
    auto totalOuterSizeA = std::accumulate(_outerSizeA.begin(), _outerSizeA.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    auto totalOuterSizeB = std::accumulate(_outerSizeB.begin(), _outerSizeB.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    if (totalOuterSizeA > totalOuterSizeB && totalOuterSizeB <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to A over those to B (Rationale: reduce TLB misses)
        for (int i = 0; i < _dim; ++i)
            loopOrder[_dim - 1 - i] = i; // innermost loop idx is stored at dim_-1
        return;
    } else if (totalOuterSizeB > totalOuterSizeA && totalOuterSizeA <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to B over those to A (Rationale: reduce TLB misses)
        for (int i = 0; i < _dim; ++i)
            loopOrder[_dim - 1 - i] = _dim - 1 - i; // innermost loop idx is stored at dim_-1
        return;
    }

    // create cost matrix; cost[i,idx] === cost for idx being at loop-level i
    std::vector<double> costs(_dim * _dim);
    for (int i = 0; i < _dim; ++i) {
        for (int idx = 0; idx < _dim; ++idx) { // idx is at loop i
            double cost = 0;
            if (i != 0) {
                int const        posB        = find_pos(idx, _perm);
                int const        importanceA = (1 << (_dim - idx));  // stride-1 has the most importance ...
                int const        importanceB = (1 << (_dim - posB)); // subsequent indices are half as important
                int const        penalty     = 10 * (1 << (i - 1));
                constexpr double bias        = 1.01;
                cost                         = (importanceA + importanceB * bias) * penalty;
            }
            costs[i + idx * _dim] = cost;
        }
    }
    std::list<int> availLoopLevels; // available rows
    std::list<int> availIndices;
    for (int i = 0; i < _dim; ++i) {
        availLoopLevels.push_back(i);
        availIndices.push_back(i);
    }

    // create best loop order constructively without generating all
    for (int i = 0; i < _dim; ++i) {
        // find column with maximum cost
        int    selectedIdx = 0;
        double maxValueAll = 0;
        for (auto c : availIndices) {
            double maxValue = 0;
            for (auto r : availLoopLevels) {
                double const val = costs[c * _dim + r];
                maxValue         = (val > maxValue) ? val : maxValue;
            }

            if (maxValue > maxValueAll) {
                maxValueAll = maxValue;
                selectedIdx = c;
            }
        }
        // find minimum in that column
        int    selectedLoopLevel = 0;
        double minValue          = 1e100;
        for (auto r : availLoopLevels) {
            double const val = costs[selectedIdx * _dim + r];
            if (val < minValue) {
                minValue          = val;
                selectedLoopLevel = r;
            }
        }
        // update loop order
        loopOrder[_dim - 1 - i] = selectedIdx; // innermost loop idx is stored at dim_-1
        // remove selected row
        for (auto it = availLoopLevels.begin(); it != availLoopLevels.end(); it++)
            if (*it == selectedLoopLevel) {
                availLoopLevels.erase(it);
                break;
            }
        // remove selected col
        for (auto it = availIndices.begin(); it != availIndices.end(); it++)
            if (*it == selectedIdx) {
                availIndices.erase(it);
                break;
            }
    }
}

template <typename floatType>
double TransposeImpl<floatType>::loop_cost_heuristic(std::vector<int> const &loopOrder) const {
    double loopCost = 0.0;
    for (int i = 1; i < _dim; ++i) {
        int const idx         = loopOrder[_dim - 1 - i];
        int const posB        = find_pos(idx, _perm);
        int const importanceA = (1 << (_dim - idx));  // stride-1 has the most importance ...
        int const importanceB = (1 << (_dim - posB)); // subsequent indices are half as important
        int const penalty     = 10 * (1 << (i - 1));
        double    bias        = 1.01;
        loopCost += (importanceA + importanceB * bias) * penalty;
    }

    return loopCost;
}

template <typename floatType>
void TransposeImpl<floatType>::get_loop_orders(std::vector<std::vector<int>> &loopOrders) const {
    loopOrders.clear();
    if (_selectionMethod == ESTIMATE) {
        loopOrders.emplace_back(std::vector<int>(_dim));
        get_best_loop_order(loopOrders[0]);
        return;
    }

    std::vector<int> loopOrder;
    for (int i = 0; i < _dim; i++)
        loopOrder.push_back(i); // NOLINT

    // create all loopOrders
    do {
        if (_perm[0] == 0 && loopOrder[_dim - 1] != 0)
            continue; // ATTENTION: we skip all loop-orders where the stride-1 index
                      // is not the inner-most loop iff perm[0] == 0 (both for perf &
                      // correctness)

        loopOrders.push_back(loopOrder);
    } while (std::next_permutation(loopOrder.begin(), loopOrder.end()));

    // sort according to loop heuristic
    std::sort(loopOrders.begin(), loopOrders.end(), [this](std::vector<int> const &loopOrder1, std::vector<int> const &loopOrder2) {
        return this->loop_cost_heuristic(loopOrder1) < this->loop_cost_heuristic(loopOrder2);
    });

    if (this->infoLevel_ > 1)
        for (auto const &loopOrder : loopOrders) {
            print_vector(loopOrder, "loop");
            EINSUMS_LOG_INFO("HPTT: penalty: {}", loop_cost_heuristic(loopOrder));
        }
}

template <typename floatType>
void TransposeImpl<floatType>::create_plan() {
//   printf("entering createPlan()\n");
#ifdef HPTT_TIMERS
    double timeStart = omp_get_wtime();
#endif

    std::vector<std::shared_ptr<Plan>> allPlans;
    create_plans(allPlans);

#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: createPlans() took {} ms", (omp_get_wtime() - timeStart) * 1000);
    timeStart = omp_get_wtime();
#endif
    _masterPlan = select_plan(allPlans);
    if (this->infoLevel_ > 0) {
        EINSUMS_LOG_INFO("HPTT: configuration of best plan:");
        _masterPlan->print();
    }
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: SelectPlan() took {} ms", (omp_get_wtime() - timeStart) * 1000);
#endif
}

template <typename floatType>
void TransposeImpl<floatType>::create_plans(std::vector<std::shared_ptr<Plan>> &plans) const {
    if (_dim == 1 || (_dim == 2 && _perm[0] == 0)) {
        plans.emplace_back(new Plan); // create dummy plan
        return;                       // handled within execute()
    }
#ifdef HPTT_TIMERS
    double parallelStrategiesTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> parallelismStrategies;
    this->get_parallelism_strategies(parallelismStrategies);
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: {} parallel strategies. Time: {} ms", parallelismStrategies.size(),
                     (omp_get_wtime() - parallelStrategiesTime) * 1000);

    double loopOrdersTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> loopOrders;
    this->get_loop_orders(loopOrders);
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: {} loop orders. Time: {} ms", loopOrders.size(), (omp_get_wtime() - loopOrdersTime) * 1000);
#endif

    if (_selectedParallelStrategyId != -1) {
        int              selectedParallelStrategyId = std::min((int)parallelismStrategies.size() - 1, _selectedParallelStrategyId);
        std::vector<int> parStrategy(parallelismStrategies[selectedParallelStrategyId]);
        print_vector(parStrategy, "selected parallel: ");
        parallelismStrategies.clear();
        parallelismStrategies.push_back(parStrategy);
    }
    if (_selectedLoopOrderId != -1) {
        int              selectedLoopOrderId = std::min((int)loopOrders.size() - 1, _selectedLoopOrderId);
        std::vector<int> loopOrder(loopOrders[selectedLoopOrderId]);
        print_vector(loopOrder, "selected loopOrder: ");
        loopOrders.clear();
        loopOrders.push_back(loopOrder);
    }

    int const posStride1A_inB = find_pos(0, _perm);
    int const posStride1B_inA = _perm[0];

    // combine the loopOrder and parallelismStrategies according to their
    // heuristics, search the space with a growing rectangle (from best to worst,
    // see line marked with ***)
    bool done = false;
    for (int start = 0; start < std::max(parallelismStrategies.size(), loopOrders.size()) && !done; start++)
        for (int i = 0; i < parallelismStrategies.size() && !done; i++) {
            for (int j = 0; j < loopOrders.size() && !done; j++) {
                if (i > start || j > start || (i != start && j != start))
                    continue; // these are already done ***

                auto      numThreadsAtLoop = parallelismStrategies[i];
                auto      loopOrder        = loopOrders[j];
                auto      plan             = std::make_shared<Plan>(loopOrder, numThreadsAtLoop);
                int const numTasks         = plan->get_num_tasks();

#ifdef _OPENMP
#    pragma omp parallel for num_threads(_numThreads) if (_numThreads > 1)
#endif
                for (int taskId = 0; taskId < numTasks; taskId++) {
                    ComputeNode *currentNode = plan->get_root_node(taskId);

                    int numThreadsPerComm = numTasks; // global communicator // e.g., 6
                    int taskIdComm        = taskId;   // e.g., 0,1,2,3,4,5
                    // divide each loop-level l, corresponding to index loopOrder[l], into
                    // numThreadsAtLoop[index] chunks
                    for (int l = 0; l < _dim; ++l) {
                        int const index  = loopOrder[l];
                        currentNode->inc = this->get_increment(index);

                        int const numTasksAtLevel         = numThreadsAtLoop[index];                                   //  e.g., 3
                        int const numParallelismAvailable = (_sizeA[index] + currentNode->inc - 1) / currentNode->inc; // e.g., 5
                        int const workPerThread = (numParallelismAvailable + numTasksAtLevel - 1) / numTasksAtLevel;   // ceil(5/3) = 2

                        numThreadsPerComm /= numTasksAtLevel;                // numThreads in next communicator // 6/3 = 2
                        int const commId = (taskIdComm / numThreadsPerComm); //  = 0,0,1,1,2,2
                        taskIdComm       = taskIdComm % numThreadsPerComm;   // local taskId in next
                                                                             // communicator // 0,1,0,1,0,1

                        if (index == 0)
                            currentNode->indexA = true;
                        if (find_pos(index, _perm) == 0)
                            currentNode->indexB = true;
                        currentNode->start = std::min(_sizeA[index] + _offsetB[find_pos(index, _perm)],
                                                      commId * workPerThread * currentNode->inc + _offsetB[find_pos(index, _perm)]);
                        currentNode->end   = std::min(_sizeA[index] + _offsetB[find_pos(index, _perm)],
                                                      (commId + 1) * workPerThread * currentNode->inc + _offsetB[find_pos(index, _perm)]);

                        currentNode->lda       = _lda[index];
                        currentNode->ldb       = _ldb[find_pos(index, _perm)];
                        currentNode->offDiffAB = (ptrdiff_t)_offsetA[index] - (ptrdiff_t)_offsetB[find_pos(index, _perm)];

                        if (_perm[0] != 0 || l != _dim - 1) {
                            currentNode->next = std::make_unique<ComputeNode>();
                            currentNode       = currentNode->next.get();
                        }
                    }

                    // macro-kernel
                    if (_perm[0] != 0) {
                        if (posStride1A_inB == 0)
                            currentNode->indexB = true;
                        currentNode->start     = -1;
                        currentNode->end       = -1;
                        currentNode->inc       = -1;
                        currentNode->lda       = _lda[posStride1B_inA];
                        currentNode->ldb       = _ldb[posStride1A_inB];
                        currentNode->offDiffAB = (ptrdiff_t)_offsetA[posStride1B_inA] - (ptrdiff_t)_offsetB[posStride1A_inB];
                        currentNode->next.reset();
                    }
                }
                plans.push_back(plan);
                if (_selectionMethod == ESTIMATE || (_selectionMethod == MEASURE && plans.size() > 200) ||
                    (_selectionMethod == PATIENT && plans.size() > 400) || (_selectionMethod == CRAZY && plans.size() > 800))
                    done = true;
            }
        }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
template <typename floatType>
float TransposeImpl<floatType>::estimate_execution_time(std::shared_ptr<Plan> const plan) {
    auto startTime = std::chrono::high_resolution_clock::now();
    this->execute_estimate(plan.get());
    double elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();

    double const minMeasurementTime = 0.1; // in seconds

    // do at least 3 repetitions or spent at least 'minMeasurementTime' seconds
    // for each candidate
    int nRepeat = std::min(3, (int)std::ceil(minMeasurementTime / elapsedTime));

    // execute just a few iterations and exterpolate the result
    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nRepeat; ++i) // ATTENTION: we are not clearing the caches inbetween runs
        this->execute_estimate(plan.get());
    elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();
    elapsedTime /= nRepeat;

    EINSUMS_LOG_DEBUG("HPTT: estimated time: {:.3e} ms.", elapsedTime);
    return elapsedTime;
}

template <typename floatType>
double TransposeImpl<floatType>::get_time_limit() const {
    if (_selectionMethod == ESTIMATE)
        return 0.0;
    else if (_selectionMethod == MEASURE)
        return 10.; // 10s
    else if (_selectionMethod == PATIENT)
        return 60.; // 1m
    else if (_selectionMethod == CRAZY)
        return 3600.; // 1h
    else {
        EINSUMS_LOG_ERROR("HPTT: selectionMethod unknown.");
        exit(-1);
    }
    return -1;
}

template <typename floatType>
std::shared_ptr<Plan> TransposeImpl<floatType>::select_plan(std::vector<std::shared_ptr<Plan>> const &plans) {
    if (plans.size() <= 0) {
        EINSUMS_LOG_ERROR("HPTT: internal error: not enough plans generated.");
        exit(-1);
    }
    if (_selectionMethod == ESTIMATE) // fast return
        return plans[0];

    double timeLimit               = this->get_time_limit() * 1000; // in ms
    int    maxAutotuningCandidates = plans.size();
    if (_maxAutotuningCandidates != -1) {
        maxAutotuningCandidates = _maxAutotuningCandidates;
        timeLimit               = 1e9;
    }

    float minTime     = std::numeric_limits<float>::max();
    int   bestPlan_id = 0;

    if (plans.size() > 1) {
        int  plansEvaluated = 0;
        auto startTime      = std::chrono::high_resolution_clock::now();
        for (int plan_id = 0; plan_id < maxAutotuningCandidates; plan_id++) {
            auto const &p = plans[plan_id];

            double elapsedTime =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
                    .count();
            if (elapsedTime >= timeLimit) // timelimit reached
                break;

            float estimatedTime = this->estimate_execution_time(p);
            plansEvaluated++;

            if (estimatedTime < minTime) {
                bestPlan_id = plan_id;
                minTime     = estimatedTime;
            }
            if (this->infoLevel_ > 1) {
                EINSUMS_LOG_INFO("HPTT: plan {} will take roughly {} ms.", plan_id, estimatedTime * 1000.);
                plans[plan_id]->print();
            }
        }
        if (this->infoLevel_ > 0)
            EINSUMS_LOG_INFO("HPTT: evaluated {}/{} candidates and selected candidate {}.", plansEvaluated, plans.size(), bestPlan_id);
    }
    return plans[bestPlan_id];
}

template <typename FloatType>
void TransposeImpl<FloatType>::write_to_file(std::FILE *fp) const {
    setup_file(fp);

    // Get the file header.
    FileHeader header;
    uint32_t   check;
    size_t     error2;

    TransposeConstants constants;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Trying to write to a file that is not a HPTT transpose file!");
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    constants = {.dim                      = _dim,
                 .numThreads               = _numThreads,
                 .innerStrideA             = _innerStrideA,
                 .innerStrideB             = _innerStrideB,
                 .selectedParallelStrategy = _selectedParallelStrategyId,
                 .selectedLoopOrderId      = _selectedLoopOrderId,
                 .conjA                    = _conjA,
                 .pad                      = 0};

    error2 = fwrite(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_sizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_outerSizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_outerSizeB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_offsetA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_offsetB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_lda.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_ldb.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_perm.data(), sizeof(int), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    _masterPlan->write_to_file(fp);

    check = compute_checksum(fp);

    error1 = fseek(fp, offsetof(FileHeader, checksum), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fwrite(&check, sizeof(uint32_t), 1, fp);

    std::fflush(fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    return;

write_to_file_error:
    EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
}

template <typename FloatType>
TransposeImpl<FloatType>::TransposeImpl(std::FILE *fp, FloatType alpha, FloatType const *A, FloatType beta, FloatType *B) {
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif
    // Get the file header.
    FileHeader         header;
    size_t             error2;
    TransposeConstants constants;
    uint32_t           check;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Trying to read from a file that is not a HPTT transpose file!");
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    _A     = A;
    _B     = B;
    _alpha = alpha;
    _beta  = beta;
    _dim   = constants.dim;

    if (endian_char() != header.version[3]) {
        _dim = byteswap(_dim);
    }

    _sizeA.resize(_dim);
    _perm.resize(_dim);
    _outerSizeA.resize(_dim);
    _outerSizeB.resize(_dim);
    _offsetA.resize(_dim);
    _offsetB.resize(_dim);
    _innerStrideA = constants.innerStrideA;
    _innerStrideB = constants.innerStrideB;
    _lda.resize(_dim);
    _ldb.resize(_dim);
    _threadIds.reserve(_dim);
    _numThreads                 = constants.numThreads;
    _selectedParallelStrategyId = constants.selectedParallelStrategy;
    _selectedLoopOrderId        = constants.selectedLoopOrderId;
    _conjA                      = constants.conjA;

    if (endian_char() != header.version[3]) {
        _innerStrideA               = byteswap(constants.innerStrideA);
        _innerStrideB               = byteswap(constants.innerStrideB);
        _numThreads                 = byteswap(constants.numThreads);
        _selectedParallelStrategyId = byteswap(constants.selectedParallelStrategy);
        _selectedLoopOrderId        = byteswap(constants.selectedLoopOrderId);
    }

    for (int i = 0; i < _numThreads; ++i)
        _threadIds.push_back(i);

    error2 = fread(_sizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_outerSizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_outerSizeB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_offsetA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_offsetB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_lda.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_ldb.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_perm.data(), sizeof(int), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    if (endian_char() != header.version[3]) {
        for (int i = 0; i < _dim; i++) {
            _sizeA[i]      = byteswap(_sizeA[i]);
            _perm[i]       = byteswap(_perm[i]);
            _outerSizeA[i] = byteswap(_outerSizeA[i]);
            _outerSizeB[i] = byteswap(_outerSizeB[i]);
            _offsetA[i]    = byteswap(_offsetA[i]);
            _offsetB[i]    = byteswap(_offsetB[i]);
            _lda[i]        = byteswap(_lda[i]);
            _ldb[i]        = byteswap(_ldb[i]);
        }
    }

    _masterPlan = std::make_shared<Plan>(fp, endian_char() != header.version[3]);

    return;

read_from_file_error:
    EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
}

template class TransposeImpl<float>;
template class TransposeImpl<double>;
template class TransposeImpl<FloatComplex>;
template class TransposeImpl<DoubleComplex>;

template void TransposeImpl<float>::execute_expert<true, true, true>();
template void TransposeImpl<float>::execute_expert<true, false, true>();
template void TransposeImpl<float>::execute_expert<false, true, true>();
template void TransposeImpl<float>::execute_expert<false, false, true>();
template void TransposeImpl<float>::execute_expert<true, true, false>();
template void TransposeImpl<float>::execute_expert<true, false, false>();
template void TransposeImpl<float>::execute_expert<false, true, false>();
template void TransposeImpl<float>::execute_expert<false, false, false>();

template void TransposeImpl<double>::execute_expert<true, true, true>();
template void TransposeImpl<double>::execute_expert<false, true, true>();
template void TransposeImpl<double>::execute_expert<true, false, true>();
template void TransposeImpl<double>::execute_expert<false, false, true>();
template void TransposeImpl<double>::execute_expert<true, true, false>();
template void TransposeImpl<double>::execute_expert<false, true, false>();
template void TransposeImpl<double>::execute_expert<true, false, false>();
template void TransposeImpl<double>::execute_expert<false, false, false>();

template void TransposeImpl<FloatComplex>::execute_expert<true, true, true>();
template void TransposeImpl<FloatComplex>::execute_expert<false, true, true>();
template void TransposeImpl<FloatComplex>::execute_expert<true, false, true>();
template void TransposeImpl<FloatComplex>::execute_expert<false, false, true>();
template void TransposeImpl<FloatComplex>::execute_expert<true, true, false>();
template void TransposeImpl<FloatComplex>::execute_expert<false, true, false>();
template void TransposeImpl<FloatComplex>::execute_expert<true, false, false>();
template void TransposeImpl<FloatComplex>::execute_expert<false, false, false>();

template void TransposeImpl<DoubleComplex>::execute_expert<true, true, true>();
template void TransposeImpl<DoubleComplex>::execute_expert<false, true, true>();
template void TransposeImpl<DoubleComplex>::execute_expert<true, false, true>();
template void TransposeImpl<DoubleComplex>::execute_expert<false, false, true>();
template void TransposeImpl<DoubleComplex>::execute_expert<true, true, false>();
template void TransposeImpl<DoubleComplex>::execute_expert<false, true, false>();
template void TransposeImpl<DoubleComplex>::execute_expert<true, false, false>();
template void TransposeImpl<DoubleComplex>::execute_expert<false, false, false>();

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template class TransposeImpl<einsums::simd::half_t>;

template void TransposeImpl<einsums::simd::half_t>::execute_expert<true, true, true>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<false, true, true>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<true, false, true>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<false, false, true>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<true, true, false>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<false, true, false>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<true, false, false>();
template void TransposeImpl<einsums::simd::half_t>::execute_expert<false, false, false>();
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
template class TransposeImpl<einsums::simd::bfloat16_t>;

template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<true, true, true>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<false, true, true>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<true, false, true>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<false, false, true>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<true, true, false>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<false, true, false>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<true, false, false>();
template void TransposeImpl<einsums::simd::bfloat16_t>::execute_expert<false, false, false>();
#endif

// ---------------------------------------------------------------------------
// Per-rung factory entry points. The arch-neutral factory in
// TransposeFactory.cpp selects among these at plan creation; explicit
// instantiations below give it symbols to link against.
// ---------------------------------------------------------------------------

template <typename floatType>
std::shared_ptr<hptt::Transpose<floatType>>
make_transpose(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
               size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB, int const dim, floatType const *A,
               floatType const alpha, floatType *B, floatType const beta, SelectionMethod const selectionMethod, int const numThreads,
               int const *threadIds, bool const useRowMajor) {
    return std::make_shared<TransposeImpl<floatType>>(sizeA, perm, outerSizeA, outerSizeB, offsetA, offsetB, innerStrideA, innerStrideB,
                                                      dim, A, alpha, B, beta, selectionMethod, numThreads, threadIds, useRowMajor);
}

template <typename floatType>
std::shared_ptr<hptt::Transpose<floatType>> make_transpose_from_file(std::FILE *fp, floatType alpha, floatType const *A, floatType beta,
                                                                     floatType *B) {
    return std::make_shared<TransposeImpl<floatType>>(fp, alpha, A, beta, B);
}

#define EINSUMS_HPTT_INSTANTIATE_FACTORIES(T)                                                                                              \
    template std::shared_ptr<hptt::Transpose<T>> make_transpose<T>(                                                                        \
        size_t const *, int const *, size_t const *, size_t const *, size_t const *, size_t const *, size_t const, size_t const,           \
        int const, T const *, T const, T *, T const, SelectionMethod const, int const, int const *, bool const);                           \
    template std::shared_ptr<hptt::Transpose<T>> make_transpose_from_file<T>(std::FILE *, T, T const *, T, T *);

EINSUMS_HPTT_INSTANTIATE_FACTORIES(float)
EINSUMS_HPTT_INSTANTIATE_FACTORIES(double)
EINSUMS_HPTT_INSTANTIATE_FACTORIES(FloatComplex)
EINSUMS_HPTT_INSTANTIATE_FACTORIES(DoubleComplex)

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
EINSUMS_HPTT_INSTANTIATE_FACTORIES(einsums::simd::half_t)
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
EINSUMS_HPTT_INSTANTIATE_FACTORIES(einsums::simd::bfloat16_t)
#endif

#undef EINSUMS_HPTT_INSTANTIATE_FACTORIES

} // namespace EINSUMS_SIMD_ARCH_NS
} // namespace hptt
