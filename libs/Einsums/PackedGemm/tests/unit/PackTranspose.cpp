//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The resolved rung's panel transpose against a scalar reference.
//
// pack_A and pack_B hand their K-contiguous panels to this kernel, which works
// a lanes x lanes register tile at a time and leaves ragged edges to a scalar
// tail: fewer rows than a tile, a K that is not a whole number of tiles, and a
// panel stride wider than the rows it holds. The tile width is the rung's, so
// the case runs once per rung through the SIMD rung tests, and an edge that is
// right at one width and wrong at another shows up by rung.

#include <Einsums/PackedGemm/Packing.hpp>

#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::packed_gemm;

namespace {

template <typename T>
void check_transpose(int64_t nrows, int64_t kc, int64_t ld, std::mt19937 &rng) {
    std::uniform_real_distribution<T> dist(T{-1}, T{1});

    // Each row is its own allocation at an arbitrary offset, as pack_A's rows are
    // runs of A at arbitrary strides; +1 staggers them off vector alignment.
    std::vector<std::vector<T>> storage(static_cast<size_t>(nrows));
    std::vector<T const *>      rows(static_cast<size_t>(nrows));
    for (int64_t r = 0; r < nrows; ++r) {
        auto &row = storage[static_cast<size_t>(r)];
        row.resize(static_cast<size_t>(kc + 1));
        for (auto &x : row) {
            x = dist(rng);
        }
        rows[static_cast<size_t>(r)] = row.data() + 1;
    }

    // A sentinel fills the panel and a guard band past it, so a store outside
    // panel[r + k * ld] for r < nrows, k < kc is caught.
    T const        sentinel = T{7};
    int64_t const  guard    = 64;
    std::vector<T> panel(static_cast<size_t>(ld * kc + guard), sentinel);

    PackTransposeFn<T> const fn = pack_transpose_entry<T>();
    REQUIRE(fn != nullptr);
    fn(panel.data(), rows.data(), nrows, kc, ld);

    for (int64_t k = 0; k < kc; ++k) {
        for (int64_t r = 0; r < ld; ++r) {
            T const got = panel[static_cast<size_t>(r + k * ld)];
            if (r < nrows) {
                REQUIRE(got == rows[static_cast<size_t>(r)][k]);
            } else {
                REQUIRE(got == sentinel);
            }
        }
    }
    for (int64_t g = 0; g < guard; ++g) {
        REQUIRE(panel[static_cast<size_t>(ld * kc + g)] == sentinel);
    }
}

} // namespace

TEMPLATE_TEST_CASE("pack transpose matches the scalar copy at every edge", "[packed_gemm][pack]", float, double) {
    std::mt19937 rng(20260928);
    // Rows up to twice the widest tile (AVX-512 float is 16), K through every
    // remainder of the widest tile and past it, and panels exactly as wide as
    // their rows or wider, as pack_B's NR = 6 panels and a tail panel are.
    for (int64_t nrows = 1; nrows <= 33; ++nrows) {
        for (int64_t kc : {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 67}) {
            for (int64_t extra : {0, 3}) {
                check_transpose<TestType>(nrows, kc, nrows + extra, rng);
            }
        }
    }
}

TEMPLATE_TEST_CASE("pack transpose is absent for the complex types", "[packed_gemm][pack]", std::complex<float>, std::complex<double>) {
    // pack_A keeps the scalar copy when the entry is null; complex has no
    // per-rung transpose, and must not resolve to a real one by accident.
    REQUIRE(pack_transpose_entry<TestType>() == nullptr);
}
