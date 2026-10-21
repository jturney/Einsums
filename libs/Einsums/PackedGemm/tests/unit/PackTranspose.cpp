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
#include <type_traits>
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

namespace {

/// A rank-2 operand at arbitrary strides, read through the same (row, k) pair a panel definition uses.
template <typename T>
struct StridedOperand2 {
    int64_t        rows, k, row_stride, k_stride;
    std::vector<T> data;
    StridedOperand2(int64_t r, int64_t kk, int64_t rs, int64_t ks, std::mt19937 &rng) : rows{r}, k{kk}, row_stride{rs}, k_stride{ks} {
        data.resize(static_cast<size_t>((r - 1) * rs + (kk - 1) * ks + 1));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto &x : data) {
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                x = T(static_cast<typename T::value_type>(dist(rng)), static_cast<typename T::value_type>(dist(rng)));
            } else {
                x = static_cast<T>(dist(rng));
            }
        }
    }
    [[nodiscard]] T at(int64_t r, int64_t kk, bool conj) const {
        T v = data[static_cast<size_t>(r * row_stride + kk * k_stride)];
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            if (conj) {
                v = std::conj(v);
            }
        }
        return v;
    }
};

/// The one-index plan pack_A reads: M at position 0, K at position 1, with the operand's strides.
PackingPlan one_index_plan(int64_t rows, int64_t k, int64_t row_stride, int64_t k_stride) {
    PackingPlan plan;
    plan.valid       = true;
    plan.m_dims      = {{.tensor_pos = 0, .size = rows, .tensor_stride = row_stride}};
    plan.n_dims      = plan.m_dims;
    plan.k_dims_in_a = {{.tensor_pos = 1, .size = k, .tensor_stride = k_stride}};
    plan.k_dims_in_b = plan.k_dims_in_a;
    plan.M_total = plan.N_total = rows;
    plan.K_total                = k;
    return plan;
}

} // namespace

TEMPLATE_TEST_CASE("pack_A and pack_B write the panels their layout defines", "[packed_gemm][pack]", float, double, std::complex<double>) {
    // A panel holds R rows of the operand by kc_len of K, panel p at p * R * kc_len, element
    // (i, k) at i + k * R, zero past the last row. The single-index fast paths used to handle
    // every one-index layout themselves, a scalar gather for the strided ones, writing the panels
    // k by k: R * kc_len elements apart, a power-of-two stride whenever kc_len is a power of two,
    // where those writes collided in the cache and pack_A ran 5.5x and pack_B 1.9x slower. They now
    // keep only the unit-stride copy, panel by panel, and every other layout - a contiguous K, both
    // strided, conjugated - takes the general path. Every layout is checked against the definition.
    using T = TestType;
    std::mt19937 rng(20261021U);
    struct Layout {
        int64_t     rows, k, row_stride, k_stride;
        char const *what;
    };
    for (Layout const layout :
         {Layout{37, 29, 1, 37, "rows contiguous"}, Layout{37, 29, 29, 1, "K contiguous"}, Layout{37, 29, 2, 2 * 37 + 3, "both strided"},
          Layout{64, 128, 1, 64, "rows contiguous, power-of-two K stride"},
          Layout{64, 128, 128, 1, "K contiguous, power-of-two row stride"}}) {
        StridedOperand2<T> const op(layout.rows, layout.k, layout.row_stride, layout.k_stride, rng);
        PackingPlan const        plan = one_index_plan(layout.rows, layout.k, layout.row_stride, layout.k_stride);
        for (int const R : {6, 16}) {
            for (bool const conj : {false, true}) {
                CAPTURE(layout.what, R, conj);
                int64_t const        row_start = 3, row_len = layout.rows - 3, k_start = 2, k_len = layout.k - 2;
                int64_t const        panels = (row_len + R - 1) / R;
                std::vector<T> const fill(static_cast<size_t>(panels * R * k_len), T{9});
                auto                 a_panels = fill;
                auto                 b_panels = fill;
                pack_A(a_panels.data(), op.data.data(), plan, row_start, row_len, k_start, k_len, R, conj);
                pack_B(b_panels.data(), op.data.data(), plan, k_start, k_len, row_start, row_len, R, conj);
                for (int64_t p = 0; p < panels; ++p) {
                    for (int64_t kk = 0; kk < k_len; ++kk) {
                        for (int64_t i = 0; i < R; ++i) {
                            int64_t const row  = p * R + i;
                            T const       want = row < row_len ? op.at(row_start + row, k_start + kk, conj) : T{};
                            auto const    at   = static_cast<size_t>(p * R * k_len + kk * R + i);
                            REQUIRE(a_panels[at] == want);
                            REQUIRE(b_panels[at] == want);
                        }
                    }
                }
            }
        }
    }
}
