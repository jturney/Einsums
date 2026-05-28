//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Round-trip tests for HPTT with half_t (FP16) and bfloat16_t.
//
// These types lack `is_arithmetic`, no Tensor<half_t>/Tensor<bfloat16_t>
// support exists, so the higher-level permute() entry point would not
// compile. Instead we drive HPTT through its direct create_plan API and
// check correctness lane-by-lane.
//
// The half_t test path exercises the SIMD MicroKernel<half_t>; the
// bfloat16_t test path exercises the FP32-promoted scalar MicroKernel
// since BF16 has no native SIMD arithmetic on either NEON or AVX-512.

#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_all.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace {
// HPTT picks work for the current OMP thread by matching omp_get_thread_num()
// against the plan's threadIds. Outside an explicit OMP parallel region the
// thread number is unstable in libomp, so we mirror what TensorAlgebra does
// in src/Permute.cpp and pass omp_get_max_threads() as the plan's worker
// count — that creates threadIds {0..max-1} which always covers the calling
// thread.
inline int hptt_test_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
} // namespace

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)

TEST_CASE("HPTT half_t 2D transpose round-trip", "[hptt][half]") {
    using half_t = einsums::simd::half_t;

    constexpr size_t N = 32;

    std::vector<half_t> A(N * N), B(N * N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i + j * N] = static_cast<half_t>(static_cast<float>(i) * 0.25f + static_cast<float>(j) * 0.125f + 1.0f);

    half_t const alpha = static_cast<half_t>(1.0f);
    half_t const beta  = static_cast<half_t>(0.0f);

    int const    perm[2] = {1, 0};
    size_t const size[2] = {N, N};
    auto         plan =
        hptt::create_plan<half_t>(perm, 2, alpha, A.data(), size, nullptr, beta, B.data(), nullptr, hptt::ESTIMATE, hptt_test_threads());
    plan->execute();

    // Column-major: B(j,i) = A(i,j) → B[j + i*N] = A[i + j*N].
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float const got      = static_cast<float>(B[j + i * N]);
            float const expected = static_cast<float>(A[i + j * N]);
            CHECK(std::abs(got - expected) < 1e-3f);
        }
    }
}

TEST_CASE("HPTT half_t 2D transpose with alpha", "[hptt][half]") {
    using half_t = einsums::simd::half_t;

    constexpr size_t N = 32;

    std::vector<half_t> A(N * N), B(N * N, static_cast<half_t>(0.0f));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i + j * N] = static_cast<half_t>(static_cast<float>(i + j) * 0.5f + 1.0f);

    half_t const alpha = static_cast<half_t>(2.0f);
    half_t const beta  = static_cast<half_t>(0.0f);

    int const    perm[2] = {1, 0};
    size_t const size[2] = {N, N};
    auto         plan =
        hptt::create_plan<half_t>(perm, 2, alpha, A.data(), size, nullptr, beta, B.data(), nullptr, hptt::ESTIMATE, hptt_test_threads());
    plan->execute();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float const got      = static_cast<float>(B[j + i * N]);
            float const expected = static_cast<float>(alpha) * static_cast<float>(A[i + j * N]);
            CHECK(std::abs(got - expected) < 1e-2f);
        }
    }
}

#endif // FP16

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)

TEST_CASE("HPTT bfloat16_t 2D transpose round-trip", "[hptt][bfloat]") {
    using bf16_t = einsums::simd::bfloat16_t;

    constexpr size_t N = 32;

    std::vector<bf16_t> A(N * N), B(N * N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i + j * N] = static_cast<bf16_t>(static_cast<float>(i) * 0.25f + static_cast<float>(j) * 0.125f + 1.0f);

    bf16_t const alpha = static_cast<bf16_t>(1.0f);
    bf16_t const beta  = static_cast<bf16_t>(0.0f);

    int const    perm[2] = {1, 0};
    size_t const size[2] = {N, N};
    auto         plan =
        hptt::create_plan<bf16_t>(perm, 2, alpha, A.data(), size, nullptr, beta, B.data(), nullptr, hptt::ESTIMATE, hptt_test_threads());
    plan->execute();

    // BF16 has only ~2 decimal digits of precision; loose threshold matches the format.
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float const got      = static_cast<float>(B[j + i * N]);
            float const expected = static_cast<float>(A[i + j * N]);
            CHECK(std::abs(got - expected) < 1e-2f);
        }
    }
}

TEST_CASE("HPTT bfloat16_t 2D transpose with alpha", "[hptt][bfloat]") {
    using bf16_t = einsums::simd::bfloat16_t;

    constexpr size_t N = 32;

    std::vector<bf16_t> A(N * N), B(N * N, static_cast<bf16_t>(0.0f));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i + j * N] = static_cast<bf16_t>(static_cast<float>(i + j) * 0.5f + 1.0f);

    bf16_t const alpha = static_cast<bf16_t>(2.0f);
    bf16_t const beta  = static_cast<bf16_t>(0.0f);

    int const    perm[2] = {1, 0};
    size_t const size[2] = {N, N};
    auto         plan =
        hptt::create_plan<bf16_t>(perm, 2, alpha, A.data(), size, nullptr, beta, B.data(), nullptr, hptt::ESTIMATE, hptt_test_threads());
    plan->execute();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float const got      = static_cast<float>(B[j + i * N]);
            float const expected = static_cast<float>(alpha) * static_cast<float>(A[i + j * N]);
            CHECK(std::abs(got - expected) < 1e-1f);
        }
    }
}

#endif // BF16

// Fallback so the test binary always has at least one registered TEST_CASE.
// Without this, building on a target that provides neither FP16 nor BF16
// SIMD (e.g. CI's -march=nocona) produces a Catch2 binary that reports
// "No tests ran", which ctest treats as a hard failure.
#if !(defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) ||         \
      defined(__AVX512BF16__))
TEST_CASE("Half-precision HPTT transpose unavailable on this build target", "[hptt][half][skip]") {
    SUCCEED("FP16/BF16 SIMD intrinsics not enabled by this build's target arch; tests skipped.");
}
#endif
