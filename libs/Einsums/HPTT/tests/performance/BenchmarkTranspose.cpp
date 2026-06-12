//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Throughput benchmarks for HPTT 2D transpose across the four real-element
// dtypes Einsums supports: double, float, half_t (FP16), bfloat16_t.
//
// Drives HPTT through its direct create_plan API rather than tensor_algebra::
// permute() because Tensor<half_t> / Tensor<bfloat16_t> aren't wired into the
// rest of the framework yet; the lower-level path lets us still cover the
// half-precision MicroKernels.
//
// Reports per-call wall time via `time_us` so the benchmark CLI's regex-based
// parser can pick it up. The interesting comparisons:
//   - bf16 / fp16 vs double: sees the SIMD MicroKernel rewrite + the
//     in-register 8x8 NEON transpose
//   - 2048x2048: the size where double exceeds L2 and bandwidth dominates
//   - 1024x1024: the size where everyone fits in L2 and per-call overhead matters

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cstddef>
#include <vector>

#include <Einsums/Testing.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace einsums::performance;

namespace {

inline int hptt_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <typename T>
void bench_transpose_2d(char const *label, int N) {
    LabeledSection0();
    std::vector<T> A(static_cast<size_t>(N) * N), B(static_cast<size_t>(N) * N);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<T>(static_cast<float>(i % 64) * 0.5f);

    int const    perm[2]{1, 0};
    size_t const size[2]{static_cast<size_t>(N), static_cast<size_t>(N)};
    auto const plan = hptt::create_plan<T>(perm, 2, static_cast<T>(1.0f), A.data(), size, nullptr, static_cast<T>(0.0f), B.data(), nullptr,
                                           hptt::ESTIMATE, hptt_threads());

    ProfileAnnotate("rank", int64_t(2));
    ProfileAnnotate("pattern", "ji<-ij");
    ProfileAnnotate("dtype", label);
    ProfileAnnotate("elements", int64_t(N) * N);
    ProfileAnnotate("bytes", int64_t(2) * N * N * static_cast<int64_t>(sizeof(T)));
    ProfileAnnotate("threads", int64_t(hptt_threads()));

    auto t = time_us(label, [&] { plan->execute(); });
    publish_benchmark_result(label, "t_transpose", N, t);
}

} // namespace

EINSUMS_TEST_CASE("Transpose 2D double", "[performance][hptt][transpose]") {
    for (int const N : {256, 512, 1024, 2048})
        bench_transpose_2d<double>("transpose-2d-double", N);
}

EINSUMS_TEST_CASE("Transpose 2D float", "[performance][hptt][transpose]") {
    for (int const N : {256, 512, 1024, 2048})
        bench_transpose_2d<float>("transpose-2d-float", N);
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
EINSUMS_TEST_CASE("Transpose 2D fp16", "[performance][hptt][transpose][half]") {
    for (int const N : {256, 512, 1024, 2048})
        bench_transpose_2d<einsums::simd::half_t>("transpose-2d-fp16", N);
}
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
EINSUMS_TEST_CASE("Transpose 2D bf16", "[performance][hptt][transpose][bfloat]") {
    for (int const N : {256, 512, 1024, 2048})
        bench_transpose_2d<einsums::simd::bfloat16_t>("transpose-2d-bf16", N);
}
#endif
