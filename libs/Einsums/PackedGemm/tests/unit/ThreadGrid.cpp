//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The packed loops' thread grid: how threads divide a contraction between N blocks and M groups
// (see choose_thread_grid), and that the result does not depend on the thread count.
//
// The loops used to split N alone. A contraction with few N columns then had fewer blocks than
// threads and left the rest idle, and every thread packed all of A for its own N block. They now
// split M as well, so the threads of one N block each take a disjoint group of whole M blocks. That
// moves the beta prescale, the scatter and every engine's M loop onto a group's range, which is
// what the correctness cases below exercise at thread counts that do and do not divide the work.

#include <Einsums/Options/Get.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/PackedGemmIndices.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <cmath>
#include <complex>
#include <string>
#include <tuple>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using packed_gemm::choose_thread_grid;

// The cache model's NC block on this project's development machine, as a representative cap.
constexpr int64_t kNcCap = 1020;

TEST_CASE("thread grid: few N columns put the threads on M", "[PackedGemm][ThreadGrid]") {
    // The TCB's intensli shape: N = 24 is four blocks of NR = 6, so an N-only split stops at four
    // threads. The rest go to M, whose extent has room for them.
    auto const g = choose_thread_grid(24, 677376, 24, 256, 6, kNcCap);
    CHECK(g.tn * g.tm == 24);
    CHECK(g.tn <= 4);
    CHECK(g.tm >= 6);
}

TEST_CASE("thread grid: few M rows keep the threads on N", "[PackedGemm][ThreadGrid]") {
    // One MC block of M: there is nothing to split along it.
    auto const g = choose_thread_grid(24, 200, 100000, 256, 6, kNcCap);
    CHECK(g.tm == 1);
    CHECK(g.tn == 24);
}

TEST_CASE("thread grid: the split packs the fewer elements", "[PackedGemm][ThreadGrid]") {
    // Square: A and B cost the same to pack, so the grid is as square as the divisors allow.
    auto const sq = choose_thread_grid(16, 8192, 8192, 256, 6, kNcCap);
    CHECK(sq.tn * sq.tm == 16);
    CHECK(sq.tn == 4);
    CHECK(sq.tm == 4);
    // Tall: M is 64 times N, so re-packing B for more M groups is the cheap direction.
    auto const tall = choose_thread_grid(16, 65536, 1024, 256, 6, kNcCap);
    CHECK(tall.tn * tall.tm == 16);
    CHECK(tall.tm > tall.tn);
}

TEST_CASE("thread grid: N blocks are equal and the work divides evenly", "[PackedGemm][ThreadGrid]") {
    // The first cut of the grid capped each N block at the cache model's width and let the last one
    // take what was left: M = N = 5184 on 12 threads became five 1020-column blocks and one of 84,
    // and the two threads that drew the runt idled while ten did two full blocks each (measured at
    // 0.71x of the N-only split on ccsd's rank-4 single). The blocks are now cut equal, a whole
    // number per N group, so the work items divide over the threads with nothing left over.
    for (auto [threads, M, N, MC] :
         {std::tuple{12, int64_t{5184}, int64_t{5184}, int64_t{256}}, std::tuple{12, int64_t{8064}, int64_t{8064}, int64_t{128}},
          std::tuple{24, int64_t{11520}, int64_t{8000}, int64_t{256}}}) {
        CAPTURE(threads, M, N, MC);
        auto const    g        = choose_thread_grid(threads, M, N, MC, 6, kNcCap);
        int64_t const n_blocks = (N + g.nc_blk - 1) / g.nc_blk;
        CHECK(g.nc_blk <= kNcCap);
        CHECK(g.nc_blk % 6 == 0);
        // Equal blocks: the last is at most one NR narrower per block than the rest.
        CHECK(g.nc_blk * n_blocks - N < 6 * n_blocks);
        CHECK((n_blocks * g.tm) % threads == 0);
    }
}

TEST_CASE("thread grid: one thread, or nothing to split, is the loop as it was", "[PackedGemm][ThreadGrid]") {
    auto const one = choose_thread_grid(1, 100000, 100000, 256, 6, kNcCap);
    CHECK(one.tn == 1);
    CHECK(one.tm == 1);
    // One NR block of N and one MC block of M: a single work item whatever the thread count.
    auto const tiny = choose_thread_grid(24, 100, 6, 256, 6, kNcCap);
    CHECK(tiny.tn == 1);
    CHECK(tiny.tm == 1);
}

TEST_CASE("thread grid: never more groups than blocks", "[PackedGemm][ThreadGrid]") {
    for (int threads : {2, 3, 5, 7, 12, 24, 48}) {
        for (int64_t M : {1, 255, 256, 257, 4096, 100000}) {
            for (int64_t N : {1, 6, 7, 24, 1000}) {
                CAPTURE(threads, M, N);
                auto const g = choose_thread_grid(threads, M, N, 256, 6, kNcCap);
                CHECK(g.tn >= 1);
                CHECK(g.tm >= 1);
                CHECK(g.tn * g.tm <= threads);
                CHECK(g.tn <= (N + 5) / 6);
                CHECK(g.tm <= (M + 255) / 256);
            }
        }
    }
}

#ifdef _OPENMP

namespace {

/// Runs one scope at @p threads OpenMP threads and restores the previous count after.
class ThreadCount {
  public:
    explicit ThreadCount(int threads) : _previous{omp_get_max_threads()} { omp_set_num_threads(threads); }
    ~ThreadCount() { omp_set_num_threads(_previous); }

    ThreadCount(ThreadCount const &)            = delete;
    ThreadCount &operator=(ThreadCount const &) = delete;

  private:
    int _previous;
};

/// Sets the complex-1m flag for one scope and restores it after.
class Complex1mFlag {
  public:
    explicit Complex1mFlag(bool value) : _previous{config::get(option::PackedGemmComplex1m)} {
        config::set(option::PackedGemmComplex1m, value);
    }
    ~Complex1mFlag() { config::set(option::PackedGemmComplex1m, _previous); }

    Complex1mFlag(Complex1mFlag const &)            = delete;
    Complex1mFlag &operator=(Complex1mFlag const &) = delete;

  private:
    bool _previous;
};

// Thread counts that divide the work evenly, that do not, and one past the machine.
constexpr int kThreadCounts[] = {1, 2, 3, 4, 7, 24};

template <typename T, size_t Rank>
void check_against(Tensor<T, Rank> const &C, Tensor<T, Rank> const &C_ref, int64_t k, T ab_pf, T c_pf) {
    // Every term is a product of two operands from the unit box, so no term exceeds 2 in modulus.
    double const magnitude = 2.0 * static_cast<double>(k) * std::abs(ab_pf) + std::sqrt(2.0) * std::abs(c_pf);
    for (size_t e = 0; e < C.size(); e++) {
        CAPTURE(e);
        REQUIRE_THAT(C.data()[e], CheckWithinMagnitude(C_ref.data()[e], magnitude));
    }
}

/// C(a,b,c) = A(b,d,a) * B(d,c): the intensli shape at test size. M = (a, b) = 3072 is many MC
/// blocks, N = c = 24 is four NR blocks, and K = d = 128, about 19 MFLOP: past the parallel gate.
template <typename T>
void tall_thin(int threads, T c_pf, T ab_pf) {
    ThreadCount const guard{threads};
    auto              A     = create_random_tensor<T>("A", 48, 128, 64);
    auto              B     = create_random_tensor<T>("B", 128, 24);
    auto              C     = create_random_tensor<T>("C", 64, 48, 24);
    Tensor<T, 3>      C_ref = C;
    testing::reference_einsum("abc <- bda ; dc", c_pf, &C_ref, ab_pf, A, B);

    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(c_pf, Indices{a, b, c}, &C, ab_pf, Indices{b, d, a},
                                                                                       A, Indices{d, c}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    check_against(C, C_ref, 128, ab_pf, c_pf);
}

/// C(a,b,i,j) = A(a,e,i,m) * B(e,b,m,j): the CCSD ring, a scatter (C's M and N groups interleave).
template <typename T>
void ring(int threads, T c_pf, T ab_pf) {
    ThreadCount const guard{threads};
    auto              A     = create_random_tensor<T>("A", 24, 12, 13, 9);
    auto              B     = create_random_tensor<T>("B", 12, 22, 9, 11);
    auto              C     = create_random_tensor<T>("C", 24, 22, 13, 11);
    Tensor<T, 4>      C_ref = C;
    testing::reference_einsum("abij <- aeim ; ebmj", c_pf, &C_ref, ab_pf, A, B);

    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(c_pf, Indices{a, b, i, j}, &C, ab_pf,
                                                                                       Indices{a, e, i, m}, A, Indices{e, b, m, j}, B);
    REQUIRE(handled);
    REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
    check_against(C, C_ref, 12 * 9, ab_pf, c_pf);
}

/// C(a,c,b) = A(d,a,b) * B(d,c) deep and tall enough that teams form and re-pack their shared panel
/// every K block: K = 600 is more than one K block at the team blocking (at least 512), and M = 2688
/// gives a team of three the M blocks it needs to share while staying above 4K, past which the
/// engine takes all of M as one block. The reference is computed once and every thread count is
/// checked against it, since it is the expensive part.
template <typename T>
void deep_wide(T c_pf, T ab_pf) {
    auto         A     = create_random_tensor<T>("A", 600, 48, 56);
    auto         B     = create_random_tensor<T>("B", 600, 300);
    auto         C0    = create_random_tensor<T>("C", 48, 300, 56);
    Tensor<T, 3> C_ref = C0;
    testing::reference_einsum("acb <- dab ; dc", c_pf, &C_ref, ab_pf, A, B);

    for (int const threads : {1, 3, 4, 6, 7, 12, 24}) {
        CAPTURE(threads);
        ThreadCount const guard{threads};
        Tensor<T, 3>      C       = C0;
        bool const        handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(c_pf, Indices{a, c, b}, &C, ab_pf,
                                                                                                  Indices{d, a, b}, A, Indices{d, c}, B);
        REQUIRE(handled);
        REQUIRE(std::string(packed_gemm::last_contraction_route()) == "packed");
        // Teams form on the tile engine wherever the threads divide into the cores sharing an L3;
        // elsewhere every thread works alone, which the same loop also runs. The block-GEMM
        // strategy, which complex takes on this scatter shape, keeps panels of its own.
        int const  per_l3 = packed_gemm::cpu_config().cores_per_l3;
        bool const tile   = std::string(packed_gemm::last_packed_engine()) == "tile";
        CHECK(packed_gemm::last_team_size() == ((tile && threads > 1 && per_l3 > 1 && threads % per_l3 == 0) ? per_l3 : 1));
        check_against(C, C_ref, 600, ab_pf, c_pf);
    }
}

/// A contraction with too few M blocks for a team to share keeps every thread alone, taking N
/// columns of its own: ccsd's `abc-ad-bdc` shape (M = 312, N much wider) ran at 0.36-0.45x of that
/// split when its three-member teams divided five M blocks two, two and one.
template <typename T>
void short_wide(int threads) {
    ThreadCount const guard{threads};
    auto              A     = create_random_tensor<T>("A", 12, 64, 8);
    auto              B     = create_random_tensor<T>("B", 64, 4000);
    auto              C     = create_random_tensor<T>("C", 8, 12, 4000);
    Tensor<T, 3>      C_ref = C;
    T const           c_pf  = testing::prefactor<T>(0.3, -0.7);
    T const           ab_pf = testing::prefactor<T>(1.1, 0.4);
    testing::reference_einsum("abc <- bda ; dc", c_pf, &C_ref, ab_pf, A, B);

    bool const handled = tensor_algebra::detail::try_packed_gemm_indices<false, false>(c_pf, Indices{a, b, c}, &C, ab_pf, Indices{b, d, a},
                                                                                       A, Indices{d, c}, B);
    REQUIRE(handled);
    CHECK(packed_gemm::last_team_size() == 1);
    check_against(C, C_ref, 64, ab_pf, c_pf);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("thread grid: tall thin-N contraction at every thread count", "[PackedGemm][ThreadGrid]", testing::AllScalarTypes) {
    using T = TestType;
    for (int const threads : kThreadCounts) {
        CAPTURE(threads);
        // A C prefactor of 0 stores on the first K block; a complex one prescales C per M group.
        tall_thin<T>(threads, T{0}, testing::prefactor<T>(1.1, 0.4));
        tall_thin<T>(threads, testing::prefactor<T>(0.3, -0.7), testing::prefactor<T>(1.1, 0.4));
    }
}

TEMPLATE_LIST_TEST_CASE("thread grid: ring scatter at every thread count", "[PackedGemm][ThreadGrid]", testing::AllScalarTypes) {
    using T = TestType;
    for (int const threads : kThreadCounts) {
        CAPTURE(threads);
        ring<T>(threads, testing::prefactor<T>(0.3, -0.7), testing::prefactor<T>(-0.9, 0.2));
    }
}

TEMPLATE_LIST_TEST_CASE("thread grid: teams share a panel across K and N blocks", "[PackedGemm][ThreadGrid]", testing::AllScalarTypes) {
    // Threads that share an L3 share one packed B panel, pack it together and meet before and after
    // each K block. Counts that divide into whole teams (3, 6, 12 and 24 on a three-core-per-L3 part)
    // and ones that do not (4, 7) both run, as does one thread, whose blocking and loop must be the
    // ones the engine always had.
    using T = TestType;
    deep_wide<T>(testing::prefactor<T>(0.3, -0.7), testing::prefactor<T>(1.1, 0.4));
    deep_wide<T>(T{0}, testing::prefactor<T>(-0.9, 0.2));
}

TEMPLATE_LIST_TEST_CASE("thread grid: too few M blocks keeps every thread alone", "[PackedGemm][ThreadGrid]", testing::RealScalarTypes) {
    using T = TestType;
    for (int const threads : {3, 12, 24}) {
        CAPTURE(threads);
        short_wide<T>(threads);
    }
}

TEMPLATE_LIST_TEST_CASE("thread grid: complex 1m at every thread count", "[PackedGemm][ThreadGrid]", testing::ComplexScalarTypes) {
    // The 1m engine walks its M range in real units (twice the complex extent) with its own M block,
    // so a group boundary must land on a whole complex row; this is where it would not.
    using T = TestType;
    Complex1mFlag const flag{true};
    for (int const threads : kThreadCounts) {
        CAPTURE(threads);
        ring<T>(threads, testing::prefactor<T>(0.3, -0.7), testing::prefactor<T>(-0.9, 0.2));
        tall_thin<T>(threads, T{0}, testing::prefactor<T>(1.1, 0.4));
    }
}

#endif
