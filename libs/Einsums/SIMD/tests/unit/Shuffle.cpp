//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Shuffle.hpp>
#include <Einsums/SIMD/Vec.hpp>

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
