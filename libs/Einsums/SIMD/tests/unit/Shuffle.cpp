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

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

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

// loadu_deinterleaved<R> against its definition, rows[r][k] = src[k * R + r]: from an allocation of
// exactly R * lanes elements, so a read past it is a heap overflow the sanitizer legs report, and as
// the inverse of storeu_interleaved.
template <typename T, int R>
void check_deinterleaved() {
    constexpr int L = Vec<T>::lanes;
    INFO("R = " << R << " of " << L << " lanes");
    auto const src = std::make_unique<T[]>(static_cast<std::size_t>(R * L));
    for (int i = 0; i < R * L; ++i) {
        src[i] = static_cast<T>(3 * i + 1);
    }
    Vec<T> rows[R];
    loadu_deinterleaved<R>(src.get(), rows);
    for (int r = 0; r < R; ++r) {
        T out[L];
        storeu(out, rows[r]);
        for (int k = 0; k < L; ++k) {
            CHECK(out[k] == src[k * R + r]);
        }
    }

    auto const back = std::make_unique<T[]>(static_cast<std::size_t>(R * L));
    storeu_interleaved<R>(back.get(), rows);
    for (int i = 0; i < R * L; ++i) {
        CHECK(back[i] == src[i]);
    }
}

#if defined(__linux__)
/// The same read with the input ending at an inaccessible page: a read past it faults.
template <typename T, int R>
void check_deinterleaved_at_page_end() {
    constexpr int     L    = Vec<T>::lanes;
    std::size_t const page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void *const       base = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(base != MAP_FAILED);
    REQUIRE(mprotect(static_cast<char *>(base) + page, page, PROT_NONE) == 0);
    T *const src = reinterpret_cast<T *>(static_cast<char *>(base) + page) - R * L;
    for (int i = 0; i < R * L; ++i) {
        src[i] = static_cast<T>(i);
    }
    Vec<T> rows[R];
    loadu_deinterleaved<R>(src, rows);
    T out[L];
    storeu(out, rows[R - 1]);
    CHECK(out[L - 1] == static_cast<T>(R * L - 1));
    munmap(base, 2 * page);
}
#endif

template <typename T, int... Rm1>
void check_every_row_count(std::integer_sequence<int, Rm1...>) {
    (check_interleaved<T, Rm1 + 1>(), ...);
}

template <typename T, int... Rm1>
void check_every_deinterleave(std::integer_sequence<int, Rm1...>) {
    (check_deinterleaved<T, Rm1 + 1>(), ...);
#if defined(__linux__)
    (check_deinterleaved_at_page_end<T, Rm1 + 1>(), ...);
#endif
}

} // namespace

TEMPLATE_TEST_CASE("storeu_interleaved writes every row count exactly", "[simd][shuffle]", float, double) {
    check_every_row_count<TestType>(std::make_integer_sequence<int, Vec<TestType>::lanes>{});
}

TEMPLATE_TEST_CASE("loadu_deinterleaved reads every row count exactly", "[simd][shuffle]", float, double) {
    check_every_deinterleave<TestType>(std::make_integer_sequence<int, Vec<TestType>::lanes>{});
}
