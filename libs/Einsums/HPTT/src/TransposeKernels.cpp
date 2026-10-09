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
 * The transpose kernels, compiled once per SIMD dispatch rung.
 *
 * Each copy lives in the rung's namespace and hands the planner (Transpose.cpp, compiled once)
 * a table of its entry points; see TransposeKernels.hpp for why nothing else belongs here.
 */

#include <Einsums/Config/ForceInline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Macros.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>

#include <Stripes/ComplexVec.hpp>
#include <Stripes/Convert.hpp>
#include <Stripes/Gather.hpp>
#include <Stripes/Operations.hpp>
#include <Stripes/Partial.hpp>
#include <Stripes/Platform.hpp>
#include <Stripes/Prefetch.hpp>
#include <Stripes/Shuffle.hpp>
#include <Stripes/Vec.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "TransposeKernels.hpp"

// The rung this copy of the file is compiled for, as an stripes::InstructionSet value.
// The dispatch wrappers define STRIPES_DISPATCH_RUNG to the same ordinals.
#if defined(STRIPES_DISPATCH_RUNG)
#    define EINSUMS_HPTT_PLAN_RUNG STRIPES_DISPATCH_RUNG
#else
#    define EINSUMS_HPTT_PLAN_RUNG 0
#endif

#if !defined(STRIPES_ARCH_NS)
#    define STRIPES_ARCH_NS arch_native
#endif

EINSUMS_NAMESPACE_BEGIN(hptt)
namespace STRIPES_ARCH_NS {

// std::abs has no overload for __fp16 / __bf16, so promote those to float
// before calling. Real and complex types pass straight through.
namespace detail_hptt {
template <typename T>
EINSUMS_FORCEINLINE auto abs_promoted(T x) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
    if constexpr (std::is_same_v<T, stripes::half_t>) {
        return std::abs(static_cast<float>(x));
    } else
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
        if constexpr (std::is_same_v<T, stripes::bfloat16_t>) {
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
        constexpr size_t n = stripes::native_lanes<floatType>;

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
// The SIMD micro_kernel pipeline (gather, transpose, scale, fmadd, scatter) shared by float and
// double. A unit inner stride compiles to plain loads and stores.
// ---------------------------------------------------------------------------
namespace detail_hptt {

template <typename T, bool betaIsZero>
static EINSUMS_FORCEINLINE void micro_kernel_simd(T const *A, size_t lda, size_t innerStrideA, T *B, size_t ldb, size_t innerStrideB,
                                                  T alpha, T beta) {
    using namespace stripes;
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
// SIMD micro_kernel for float and double.
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
struct MicroKernel<stripes::half_t, betaIsZero, conjA> {
    using half_t = stripes::half_t;
    static void execute(half_t const *A, size_t const lda, size_t const innerStrideA, half_t *B, size_t const ldb,
                        size_t const innerStrideB, half_t const alpha, half_t const beta) {
        detail_hptt::micro_kernel_simd<half_t, betaIsZero>(A, lda, innerStrideA, B, ldb, innerStrideB, alpha, beta);
    }
};
#endif

// ---------------------------------------------------------------------------
// micro_kernel for bfloat16_t: transpose in BF16, then scale in FP32 and round back. Needs a
// Vec<bf16> transpose, which only NEON has; AVX-512 BF16 uses the scalar kernel below.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
template <bool betaIsZero, bool conjA>
struct MicroKernel<stripes::bfloat16_t, betaIsZero, conjA> {
    using bf16_t = stripes::bfloat16_t;
    static void execute(bf16_t const *A, size_t const lda, size_t const innerStrideA, bf16_t *B, size_t const ldb,
                        size_t const innerStrideB, bf16_t const alpha, bf16_t const beta) {
        using namespace stripes;
        constexpr int N = Vec<bf16_t>::lanes;

        Vec<bf16_t> rows[N]; // NOLINT
        if (innerStrideA == 1) {
            for (int i = 0; i < N; ++i)
                rows[i] = loadu(A + i * lda);
        } else {
            for (int i = 0; i < N; ++i)
                rows[i] = gather(A + i * lda, static_cast<std::ptrdiff_t>(innerStrideA));
        }

        transpose_inplace(rows);

        Vec<float> const va = broadcast(static_cast<float>(alpha));
        Vec<float>       vb{};
        if constexpr (!betaIsZero) {
            vb = broadcast(static_cast<float>(beta));
        }

        for (int i = 0; i < N; ++i) {
            Vec<float> lo = mul(convert_low<float>(rows[i]), va);
            Vec<float> hi = mul(convert_high<float>(rows[i]), va);

            if constexpr (!betaIsZero) {
                Vec<bf16_t> const b_row =
                    innerStrideB == 1 ? loadu(B + i * ldb) : gather(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB));
                lo = fmadd(convert_low<float>(b_row), vb, lo);
                hi = fmadd(convert_high<float>(b_row), vb, hi);
            }

            Vec<bf16_t> const result = convert<bf16_t>(lo, hi);
            if (innerStrideB == 1) {
                storeu(B + i * ldb, result);
            } else {
                scatter(B + i * ldb, static_cast<std::ptrdiff_t>(innerStrideB), result);
            }
        }
    }
};
#elif defined(__AVX512BF16__)
// The FP32-promoted scalar path (AVX-512BF16 SIMD is unvalidated for lack of hardware).
template <bool betaIsZero, bool conjA>
struct MicroKernel<stripes::bfloat16_t, betaIsZero, conjA> {
    using bf16_t = stripes::bfloat16_t;
    static void execute(bf16_t const *A, size_t const lda, size_t const innerStrideA, bf16_t *B, size_t const ldb,
                        size_t const innerStrideB, bf16_t const alpha, bf16_t const beta) {
        constexpr size_t n     = stripes::native_lanes<bf16_t>;
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
        using namespace stripes;
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
        using namespace stripes;
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
// streamingStore and prefetch
// ---------------------------------------------------------------------------
template <typename floatType>
static void streamingStore(floatType *out, floatType const *in) {
    using namespace stripes;
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

template <typename floatType, stripes::PrefetchHint Hint = stripes::PrefetchHint::T2>
static EINSUMS_FORCEINLINE void prefetch_block(floatType const *A, size_t const lda) {
    constexpr int n = stripes::native_bits / 8 / sizeof(floatType);
    for (int i = 0; i < n; ++i)
        stripes::prefetch<Hint>(A + i * lda);
}
template <bool betaIsZero, typename floatType, bool conjA>
static EINSUMS_FORCEINLINE void macro_kernel_scalar(floatType const *A, size_t const lda, int blockingA, size_t innerStrideA, floatType *B,
                                                    size_t const ldb, int blockingB, size_t innerStrideB, floatType const alpha,
                                                    floatType const beta) {
    // The standard assert, not EINSUMS_ASSERT: its handler formats with fmt, which would put fmt
    // templates in every rung's copy of this file under one name (see TransposeKernels.hpp).
    assert(blockingA > 0 && blockingB > 0);

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
    constexpr int blocking_micro_ = stripes::native_bits / 8 / sizeof(floatType);
    constexpr int blocking_       = blocking_micro_ * 4;

    // Non-temporal stores need B's base, row stride and tile rows aligned to the rung's vector width.
    constexpr size_t stream_align       = stripes::native_bits / 8;
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                           innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_),
                                                                      ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == 2 * blocking_micro_ && blockingB == blocking_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == blocking_ && blockingB == 2 * blocking_micro_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        MicroKernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                           Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                           alpha, beta);
        MicroKernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
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
            prefetch_block<floatType, stripes::PrefetchHint::WriteT2>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
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
            // Which loop this is comes from the plan, not from a unit lda or ldb: with a non-unit
            // inner stride the stride-1 index's lda (ldb) is that stride, and taking it for an outer
            // loop ran the remainder at the full block width past the end of B.
            if (plan->indexA)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, scalarRemainder, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, sizeStride1B, innerStrideB, alpha, beta);
            else if (plan->indexB)
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

    constexpr int blocking_micro_ = stripes::native_bits / 8 / sizeof(floatType);
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

/// Element types the contiguous paths stream: those with a streaming store.
template <typename floatType>
inline constexpr bool streams_run_v = std::is_same_v<floatType, float> || std::is_same_v<floatType, double>;

/// B[0 .. n) = alpha * A[0 .. n), streamed past the cache (stripes::stream_store_span): a
/// contiguous run of the paths that keep the fastest index, which a loop alone never streamed (the
/// `vector nontemporal` pragma they relied on is Intel's, and GCC and Clang ignore it). Conjugation is
/// the identity on the real types this takes. The caller fences once after its last run.
template <typename floatType>
static EINSUMS_FORCEINLINE void stream_scaled_run(floatType *B, floatType const *A, size_t n, floatType const alpha) {
    using namespace stripes;
    constexpr size_t     L  = Vec<floatType>::lanes;
    Vec<floatType> const va = broadcast(alpha);
    stream_store_span(B, n, [A, va](size_t i, size_t count) { return mul(va, count == L ? loadu(A + i) : loadu_partial(A + i, count)); });
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
        if constexpr (useStreamingStores && streams_run_v<floatType>) {
            if (lda == 1 && ldb == 1 && end > plan->start) {
                stream_scaled_run(&B[plan->start], &A[plan->start + offDiffAB], static_cast<size_t>(end - plan->start), alpha);
                return;
            }
        }
        if constexpr (conjA) {
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

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_1D(floatType const *A, floatType *B, size_t const myStart, size_t const myEnd, ptrdiff_t const offDiffAB_,
                    size_t const lda, size_t const ldb, floatType const alpha, floatType const beta, int numThreads) {
    if constexpr (!betaIsZero) {
        HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if (conjA) B[i * ldb] =
                                         alpha * conj(A[(i + offDiffAB_) * lda]) + beta * B[i * ldb];
                       else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda] + beta * B[i * ldb];)
    } else {
        if constexpr (useStreamingStores && streams_run_v<floatType>) {
            if (lda == 1 && ldb == 1) {
                // Each thread streams its contiguous share, from this region or the caller's.
                if constexpr (spawnThreads) {
#ifdef _OPENMP
#    pragma omp parallel num_threads(numThreads)
#endif
                    {
#ifdef _OPENMP
                        size_t const t = static_cast<size_t>(omp_get_thread_num()), nt = static_cast<size_t>(omp_get_num_threads());
#else
                        size_t const t = 0, nt = 1;
#endif
                        size_t const len = myEnd - myStart;
                        size_t const b = myStart + len * t / nt, e = myStart + len * (t + 1) / nt;
                        stream_scaled_run(B + b, A + b + offDiffAB_, e - b, alpha);
                        stripes::stream_fence();
                    }
                } else {
                    stream_scaled_run(B + myStart, A + myStart + offDiffAB_, myEnd - myStart, alpha);
                    stripes::stream_fence();
                }
                return;
            }
        }
        HPTT_DUPLICATE(spawnThreads,
                       for (size_t i = myStart; i < myEnd; i++) if constexpr (conjA) B[i * ldb] = alpha * conj(A[(i + offDiffAB_) * lda]);
                       else B[i * ldb]                                                          = alpha * A[(i + offDiffAB_) * lda];)
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
        if constexpr (useStreamingStores && streams_run_v<floatType>) {
            if (lda[0] == 1 && ldb[0] == 1) {
                // Every column is one contiguous run. Each thread streams its columns and drains once,
                // after the last of them, instead of once per column.
                auto const column = [&](size_t j) {
                    stream_scaled_run(B + offsetB_ + j * ldb[1], A + (offsetB_ + offDiffAB_[0]) + (j + offDiffAB_[1]) * lda[1], n0, alpha);
                };
                if constexpr (spawnThreads) {
#ifdef _OPENMP
#    pragma omp parallel num_threads(numThreads)
#endif
                    {
#ifdef _OPENMP
#    pragma omp for schedule(static) nowait
#endif
                        for (size_t j = myStart; j < myEnd; j++) {
                            column(j);
                        }
                        stripes::stream_fence();
                    }
                } else {
                    for (size_t j = myStart; j < myEnd; j++) {
                        column(j);
                    }
                    stripes::stream_fence();
                }
                return;
            }
        }
        HPTT_DUPLICATE(spawnThreads, for (size_t j = myStart; j < myEnd; j++) for (size_t i = offsetB_; i < n0 + offsetB_; i++) if (conjA)
                                         B[(i * ldb[0]) + j * ldb[1]] =
                                             alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]);
                       else B[(i * ldb[0]) + j * ldb[1]] = alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]];)
    }
}

// ---------------------------------------------------------------------------
// Execution: what TransposeImpl::execute and execute_estimate run, over the flat KernelArgs.
// ---------------------------------------------------------------------------

/// Macro-kernel tile edge, in elements: four vectors. The planner steps its loops by this.
template <typename floatType>
inline constexpr int blocking_v = stripes::native_bits / 8 / sizeof(floatType) * 4;

/// The local index of OpenMP thread @p myThreadId in the plan's thread list, or -1 if it takes no part.
template <typename floatType>
static int get_local_thread_id(KernelArgs<floatType> const &a, int myThreadId) {
    int myLocalId = -1;
    for (int i = 0; i < a.numThreads; ++i)
        if (myThreadId == a.threadIds[i])
            myLocalId = i;
    return myLocalId;
}

/// The share [myStart, myEnd) of @p n work items that the calling thread takes.
template <bool spawnThreads, typename floatType>
static void get_start_end(KernelArgs<floatType> const &a, size_t n, size_t &myStart, size_t &myEnd) {
#ifdef _OPENMP
    int myLocalThreadId = get_local_thread_id(a, omp_get_thread_num());
#else
    int myLocalThreadId = 0;
#endif

    // A single-threaded plan with default thread ids belongs to whoever calls it, even from inside
    // another team, where omp_get_thread_num() is not 0.
    if (!a.callerManagedThreads && a.numThreads == 1) {
        myStart = 0;
        myEnd   = n;
        return;
    }

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

    size_t const workPerThread = (n + a.numThreads - 1) / a.numThreads;
    myStart                    = std::min(n, myLocalThreadId * workPerThread);
    myEnd                      = std::min(n, (myLocalThreadId + 1) * workPerThread);
}

template <bool useStreamingStores, bool spawnThreads, bool betaIsZero, typename floatType>
static void execute_expert(KernelArgs<floatType> const &a) noexcept {
    constexpr int blocking_ = blocking_v<floatType>;

    floatType const *const _A            = a.A;
    floatType *const       _B            = a.B;
    floatType const        _alpha        = a.alpha;
    floatType const        _beta         = a.beta;
    size_t const           _innerStrideA = a.innerStrideA;
    size_t const           _innerStrideB = a.innerStrideB;

    size_t myStart = 0;
    size_t myEnd   = 0;

    if (a.dim == 1) {
        get_start_end<spawnThreads>(a, a.sizeA[0], myStart, myEnd);
        ptrdiff_t const offDiffAB_ = (ptrdiff_t)a.offsetA[0] - (ptrdiff_t)a.offsetB[0];
        if (a.conjA)
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(
                _A, _B, myStart + a.offsetB[0], myEnd + a.offsetB[0], offDiffAB_, a.lda[0], a.ldb[0], _alpha, _beta, a.numThreads);
        else
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(
                _A, _B, myStart + a.offsetB[0], myEnd + a.offsetB[0], offDiffAB_, a.lda[0], a.ldb[0], _alpha, _beta, a.numThreads);
        return;
    } else if (a.dim == 2 && a.perm0 == 0) {
        get_start_end<spawnThreads>(a, a.sizeA[1], myStart, myEnd);
        ptrdiff_t const offDiffAB_[2] = {((ptrdiff_t)a.offsetA[0] - (ptrdiff_t)a.offsetB[0]),
                                         ((ptrdiff_t)a.offsetA[1] - (ptrdiff_t)a.offsetB[1])};
        if (a.conjA)
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(_A, {a.lda[0], a.lda[1]}, _B, {a.ldb[0], a.ldb[1]},
                                                                                   a.sizeA[0], myStart + a.offsetB[1], myEnd + a.offsetB[1],
                                                                                   offDiffAB_, a.offsetB[0], _alpha, _beta, a.numThreads);
        else
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(
                _A, {a.lda[0], a.lda[1]}, _B, {a.ldb[0], a.ldb[1]}, a.sizeA[0], myStart + a.offsetB[1], myEnd + a.offsetB[1], offDiffAB_,
                a.offsetB[0], _alpha, _beta, a.numThreads);
        return;
    }

    Plan const *const plan       = a.plan;
    int const         numTasks   = plan->get_num_tasks();
    int const         numThreads = a.numThreads;
    get_start_end<spawnThreads>(a, numTasks, myStart, myEnd);

    HPTT_DUPLICATE(
        spawnThreads,
        for (int taskId = myStart; taskId < myEnd; taskId++) if (a.perm0 != 0) {
            auto rootNode = plan->get_root_node(taskId);
            if (a.conjA)
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, true>(
                    _A, _A, _innerStrideA, _B, _B, _innerStrideB, _alpha, _beta, rootNode);
            else
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, false>(
                    _A, _A, _innerStrideA, _B, _B, _innerStrideB, _alpha, _beta, rootNode);
            // Streamed stores are weakly ordered: fence once per task, on the issuing thread.
            if constexpr (useStreamingStores && betaIsZero)
                stripes::stream_fence();
        } else {
            auto rootNode = plan->get_root_node(taskId);
            if (a.conjA)
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, true>(_A, _B, _alpha, _beta, rootNode);
            else
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, false>(_A, _B, _alpha, _beta, rootNode);
            // Its runs stream as the tiled path's tiles do, so the same drain once per task.
            if constexpr (useStreamingStores && betaIsZero)
                stripes::stream_fence();
        })
}

template <typename floatType>
static void execute(KernelArgs<floatType> const &a) noexcept {
    bool const     spawnThreads       = a.numThreads > 1;
    bool const     betaIsZero         = (a.beta == (floatType)0.0);
    constexpr bool useStreamingStores = true;
    if (spawnThreads) {
        if (betaIsZero) {
            execute_expert<useStreamingStores, true, true>(a);
        } else {
            execute_expert<useStreamingStores, true, false>(a);
        }
    } else {
        if (betaIsZero) {
            execute_expert<useStreamingStores, false, true>(a);
        } else {
            execute_expert<useStreamingStores, false, false>(a);
        }
    }
}

// Times a candidate plan: every task once, with alpha 0 and beta 1 so B keeps its contents.
template <typename floatType>
static void execute_estimate(KernelArgs<floatType> const &a) noexcept {
    constexpr int  blocking_          = blocking_v<floatType>;
    constexpr bool useStreamingStores = false;

    floatType const *const _A            = a.A;
    floatType *const       _B            = a.B;
    size_t const           _innerStrideA = a.innerStrideA;
    size_t const           _innerStrideB = a.innerStrideB;
    Plan const *const      plan          = a.plan;
    bool const             betaIsSmall   = detail_hptt::abs_promoted(a.beta) < get_zero_threshold<floatType>();

    int const numTasks = plan->get_num_tasks();
#ifdef _OPENMP
#    pragma omp parallel for num_threads(a.numThreads) if (a.numThreads > 1)
#endif
    for (int taskId = 0; taskId < numTasks; taskId++)
        if (a.perm0 != 0) {
            auto rootNode = plan->get_root_node(taskId);
            if (betaIsSmall) {
                if (a.conjA)
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, true>(_A, _A, _innerStrideA, _B, _B,
                                                                                                _innerStrideB, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, false>(_A, _A, _innerStrideA, _B, _B,
                                                                                                 _innerStrideB, 0.0, 1.0, rootNode);
            } else {
                if (a.conjA)
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, true>(_A, _A, _innerStrideA, _B, _B,
                                                                                                _innerStrideB, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, false>(_A, _A, _innerStrideA, _B, _B,
                                                                                                 _innerStrideB, 0.0, 1.0, rootNode);
            }
        } else {
            auto rootNode = plan->get_root_node(taskId);
            if (betaIsSmall) {
                if (a.conjA)
                    transpose_int_constStride1<1, floatType, useStreamingStores, true>(_A, _B, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<1, floatType, useStreamingStores, false>(_A, _B, 0.0, 1.0, rootNode);
            } else {
                if (a.conjA)
                    transpose_int_constStride1<0, floatType, useStreamingStores, true>(_A, _B, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<0, floatType, useStreamingStores, false>(_A, _B, 0.0, 1.0, rootNode);
            }
        }
}

template <typename floatType>
TransposeKernels<floatType> const &transpose_kernels() noexcept {
    static constexpr TransposeKernels<floatType> table{.vector_bits      = stripes::native_bits,
                                                       .rung             = EINSUMS_HPTT_PLAN_RUNG,
                                                       .blocking         = blocking_v<floatType>,
                                                       .execute          = &execute<floatType>,
                                                       .execute_estimate = &execute_estimate<floatType>};
    return table;
}

template TransposeKernels<float> const         &transpose_kernels<float>() noexcept;
template TransposeKernels<double> const        &transpose_kernels<double>() noexcept;
template TransposeKernels<FloatComplex> const  &transpose_kernels<FloatComplex>() noexcept;
template TransposeKernels<DoubleComplex> const &transpose_kernels<DoubleComplex>() noexcept;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template TransposeKernels<stripes::half_t> const &transpose_kernels<stripes::half_t>() noexcept;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
template TransposeKernels<stripes::bfloat16_t> const &transpose_kernels<stripes::bfloat16_t>() noexcept;
#endif

} // namespace STRIPES_ARCH_NS
EINSUMS_NAMESPACE_END(hptt)
