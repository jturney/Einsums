//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The resolved rung's tile kernel against a scalar reference.
//
// The kernel is the one place the packed engine does arithmetic, and its
// contract has edges the contraction tests reach only by accident: a ragged
// tail (mr_eff or nr_eff below the register block), a C whose row stride is
// not 1, an alpha other than 1, and accumulation onto whatever C already
// holds. Every rung's kernel is registered here through the SIMD rung tests,
// so a tile that is right at one width and wrong at another shows up by rung.

#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::packed_gemm;

namespace {

template <typename T>
std::vector<T> random_panel(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<T> dist(T{-1}, T{1});
    std::vector<T>                    v(n);
    for (auto &x : v) {
        x = dist(rng);
    }
    return v;
}

template <typename T>
void check_tile(int64_t kc, int64_t mr_eff, int64_t nr_eff, int64_t rs_c, int64_t cs_c, T alpha) {
    auto const   kernel = micro_kernel_entry<T>();
    auto const   shape  = micro_kernel_shape<T>();
    int const    MR     = shape.mr;
    int const    NR     = shape.nr;
    std::mt19937 rng(static_cast<unsigned>(kc * 131 + mr_eff * 17 + nr_eff * 3 + rs_c + cs_c));
    auto const   Ap = random_panel<T>(static_cast<size_t>(MR) * static_cast<size_t>(kc), rng);
    auto const   Bp = random_panel<T>(static_cast<size_t>(NR) * static_cast<size_t>(kc), rng);
    // C is a strided window with slack on both sides so an out-of-tile write shows.
    size_t const c_len = static_cast<size_t>((MR + 1) * rs_c + (NR + 1) * cs_c) + 8;
    auto         C     = random_panel<T>(c_len, rng);
    auto         want  = C;

    kernel(MR, NR, kc, alpha, Ap.data(), Bp.data(), mr_eff, nr_eff, C.data(), rs_c, cs_c);

    for (int64_t j = 0; j < nr_eff; ++j) {
        for (int64_t i = 0; i < mr_eff; ++i) {
            T sum{};
            for (int64_t k = 0; k < kc; ++k) {
                sum += Ap[static_cast<size_t>(i + k * MR)] * Bp[static_cast<size_t>(k * NR + j)];
            }
            want[static_cast<size_t>(i * rs_c + j * cs_c)] += alpha * sum;
        }
    }

    // A tile's K-sum reassociates between the reference and a vector kernel;
    // the bound scales with kc and stays well clear of a wrong-element error.
    double const tol = 64.0 * static_cast<double>(kc) * std::numeric_limits<T>::epsilon();
    for (size_t idx = 0; idx < c_len; ++idx) {
        REQUIRE_THAT(static_cast<double>(C[idx]), Catch::Matchers::WithinAbs(static_cast<double>(want[idx]), tol));
    }
}

} // namespace

TEMPLATE_TEST_CASE("MicroKernel - full tile into column-major C", "[PackedGemm][MicroKernel]", float, double) {
    using T          = TestType;
    auto const shape = micro_kernel_shape<T>();
    for (int64_t kc : {int64_t{1}, int64_t{7}, int64_t{64}, int64_t{513}}) {
        check_tile<T>(kc, shape.mr, shape.nr, 1, shape.mr + 3, T{1});
        check_tile<T>(kc, shape.mr, shape.nr, 1, shape.mr, T{-0.5});
    }
}

TEMPLATE_TEST_CASE("MicroKernel - ragged tails write only their own elements", "[PackedGemm][MicroKernel]", float, double) {
    using T          = TestType;
    auto const shape = micro_kernel_shape<T>();
    for (int64_t kc : {int64_t{3}, int64_t{40}}) {
        check_tile<T>(kc, 1, 1, 1, shape.mr + 3, T{1});
        check_tile<T>(kc, shape.mr - 1, shape.nr, 1, shape.mr + 1, T{2});
        check_tile<T>(kc, shape.mr, shape.nr - 1, 1, shape.mr + 5, T{1});
        check_tile<T>(kc, 3, 2, 1, shape.mr + 2, T{0.25});
    }
}

TEMPLATE_TEST_CASE("MicroKernel - row-major and strided C", "[PackedGemm][MicroKernel]", float, double) {
    using T          = TestType;
    auto const shape = micro_kernel_shape<T>();
    for (int64_t kc : {int64_t{5}, int64_t{96}}) {
        check_tile<T>(kc, shape.mr, shape.nr, shape.nr + 2, 1, T{1});
        check_tile<T>(kc, shape.mr - 2, shape.nr - 1, shape.nr + 1, 1, T{-1});
        check_tile<T>(kc, shape.mr, shape.nr, 3, 3 * shape.mr + 1, T{1.5});
    }
}

// The tile the packers cut and the blocking the loops use are both derived from
// the resolved kernel's shape, and that shape is stated in the SELECTED rung's
// vectors: two of them along M by six columns. This pins the derivation to the
// rung ladder so a width read from the wrong place (the library's compile
// flags, which is where it came from once) fails here on the first AVX2 machine.
TEMPLATE_TEST_CASE("MicroKernel - tile and blocking follow the selected rung", "[PackedGemm][MicroKernel]", float, double) {
    using T           = TestType;
    auto const  shape = micro_kernel_shape<T>();
    auto const  rung  = simd::selected_arch();
    int const   lanes = simd::vector_bits(rung) / (8 * static_cast<int>(sizeof(T)));
    auto const &hw    = hardware::cpu_info();

#if defined(__x86_64__) || defined(_M_X64)
    REQUIRE(shape.mr == 2 * lanes);
    REQUIRE(shape.nr == 6);
    REQUIRE_FALSE(shape.block_gemm); // real types run the rung's own tile on x86
#endif
    REQUIRE(hw.simd_width_f64 == simd::vector_bits(rung) / 64);

    // Blocking from that tile: one packed A column (MR * KC) within L1, the A
    // panel (MC * KC) within half the L2, the B panel (KC * NC) within half the
    // L3, every block a multiple of its register block.
    auto const blk = compute_blocking(static_cast<int64_t>(sizeof(T)), shape.mr, shape.nr);
    REQUIRE(blk.KC % 8 == 0);
    REQUIRE(blk.KC >= 64);
    REQUIRE(blk.MC % shape.mr == 0);
    REQUIRE(blk.NC % shape.nr == 0);
    REQUIRE(blk.MC * blk.KC * static_cast<int64_t>(sizeof(T)) <= hw.cache.l2 / 2 + blk.KC * static_cast<int64_t>(sizeof(T)) * shape.mr);
    REQUIRE(blk.KC * blk.NC * static_cast<int64_t>(sizeof(T)) <= hw.cache.l3 / 2 + blk.KC * static_cast<int64_t>(sizeof(T)) * shape.nr);
    if (blk.KC > 64) {
        REQUIRE(shape.mr * blk.KC * static_cast<int64_t>(sizeof(T)) <= hw.cache.l1);
    }
}
