//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Shuffle.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

using namespace einsums::simd;

TEMPLATE_TEST_CASE("transpose_inplace correctness", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;

    // Build an N×N matrix: m[i][j] = i * N + j
    std::vector<TestType> matrix(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            matrix[i * N + j] = TestType(i * N + j);

    // Load rows into Vec array
    Vec<TestType> rows[N]; // NOLINT: C-style array for SIMD register alignment
    for (int i = 0; i < N; ++i)
        rows[i] = loadu(&matrix[i * N]);

    // Transpose in-place
    transpose_inplace(rows);

    // Store back
    std::vector<TestType> result(N * N);
    for (int i = 0; i < N; ++i)
        storeu(&result[i * N], rows[i]);

    // Verify: result[i][j] should equal original[j][i]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            TestType expected = TestType(j * N + i); // transposed
            INFO("i=" << i << " j=" << j << " N=" << N);
            CHECK(result[i * N + j] == Catch::Approx(expected));
        }
    }
}

// A weak guard on its own, kept only as a cheap cross-check of the strong one
// above. A transpose that permutes both indices by the same involution passes
// this and the identity case below while scrambling every real matrix, which is
// exactly how the AVX-512 Vec<double> kernel stayed wrong: its permutation
// p = {0,1,4,5,2,3,6,7} satisfies p(p(i)) == i. Only the element-by-element
// comparison in "transpose_inplace correctness" can catch that class of bug.
TEMPLATE_TEST_CASE("transpose_inplace is its own inverse", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;

    std::vector<TestType> original(N * N);
    for (int i = 0; i < N * N; ++i)
        original[i] = TestType(i * 0.5 + 1.0);

    Vec<TestType> rows[N]; // NOLINT
    for (int i = 0; i < N; ++i)
        rows[i] = loadu(&original[i * N]);

    // Transpose twice should give back the original
    transpose_inplace(rows);
    transpose_inplace(rows);

    std::vector<TestType> result(N * N);
    for (int i = 0; i < N; ++i)
        storeu(&result[i * N], rows[i]);

    for (int i = 0; i < N * N; ++i) {
        CHECK(result[i] == Catch::Approx(original[i]));
    }
}

TEMPLATE_TEST_CASE("transpose_inplace identity matrix", "[simd]", float, double) {
    constexpr int N = Vec<TestType>::lanes;

    // Identity matrix should be unchanged by transpose
    std::vector<TestType> identity(N * N, TestType(0));
    for (int i = 0; i < N; ++i)
        identity[i * N + i] = TestType(1);

    Vec<TestType> rows[N]; // NOLINT
    for (int i = 0; i < N; ++i)
        rows[i] = loadu(&identity[i * N]);

    transpose_inplace(rows);

    std::vector<TestType> result(N * N);
    for (int i = 0; i < N; ++i)
        storeu(&result[i * N], rows[i]);

    for (int i = 0; i < N * N; ++i) {
        CHECK(result[i] == Catch::Approx(identity[i]));
    }
}

namespace {

// storeu_interleaved<R> against its definition, dst[k * R + r] = rows[r][k]: into an allocation of
// exactly R * lanes elements, so a write past it is a heap overflow the sanitizer legs report, and
// into a larger buffer whose sentinels on both sides must survive.
template <typename T, int R>
void check_interleaved() {
    constexpr int L = Vec<T>::lanes;
    INFO("R = " << R << " of " << L << " lanes");
    T      src[R][L];
    Vec<T> rows[R];
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < L; ++k) {
            src[r][k] = static_cast<T>(100 * r + k + 1);
        }
        rows[r] = loadu(src[r]);
    }

    auto const exact = std::make_unique<T[]>(static_cast<std::size_t>(R * L));
    storeu_interleaved<R>(exact.get(), rows);
    for (int k = 0; k < L; ++k) {
        for (int r = 0; r < R; ++r) {
            CHECK(exact[k * R + r] == src[r][k]);
        }
    }

    std::vector<T> padded(static_cast<std::size_t>(R * L + 2 * L), static_cast<T>(-7));
    storeu_interleaved<R>(padded.data() + L, rows);
    for (int i = 0; i < L; ++i) {
        CHECK(padded[i] == static_cast<T>(-7));
        CHECK(padded[L + R * L + i] == static_cast<T>(-7));
    }
}

template <typename T, int... Rm1>
void check_every_row_count(std::integer_sequence<int, Rm1...>) {
    (check_interleaved<T, Rm1 + 1>(), ...);
}

} // namespace

TEMPLATE_TEST_CASE("storeu_interleaved writes every row count exactly", "[simd][shuffle]", float, double) {
    check_every_row_count<TestType>(std::make_integer_sequence<int, Vec<TestType>::lanes>{});
}
