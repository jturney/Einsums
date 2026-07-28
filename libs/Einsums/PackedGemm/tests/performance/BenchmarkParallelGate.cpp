//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkParallelGate.cpp
/// @brief When is a packed contraction big enough to be worth an OpenMP region?
///
/// `blis_contraction` parallelizes its NC loop, and entering the region costs a
/// fork and a join barrier whatever the work is. Below some size that barrier is
/// more than the arithmetic it distributes, and threading makes the call slower -
/// a tiled einsum expanded into thousands of small contractions pays it thousands
/// of times. `cpu_config().min_parallel_flops` is the line, and this measures
/// where the line belongs.
///
/// The shapes are the ones that matter in practice: the per-tile contractions a
/// tiled CCSD residual expands to (small M, narrow N) and the densified ones it
/// produces instead when the tiles are too small (large N, deep K), plus a grid
/// around them that varies M, N and K one at a time so the answer is not read off
/// flops alone.
///
/// A/B it from ONE binary, since rebuilds move unrelated things:
/// @code
/// B=./build/libs/Einsums/PackedGemm/tests/performance/BenchmarkParallelGate_test
/// EINSUMS_PACKED_MIN_PARALLEL_FLOPS=0             $B  # always parallel
/// EINSUMS_PACKED_MIN_PARALLEL_FLOPS=1000000000000 $B  # always serial
/// @endcode
/// The threshold is read once per process, so each run reports one policy; the
/// header line says which.
///
/// @par What this can and cannot tell you
/// It measures ONE shape re-entered in a tight loop, which keeps the OpenMP team
/// hot, so the region looks nearly free: at the shape that dominates a 50
/// spin-orbital tiled residual the two policies differ by under a microsecond
/// here, while in the residual itself they differ by ten. Read it for two things
/// it does establish - the serial time of a shape (which matches the residual's
/// per-node time exactly) and the fact that serial matches or beats parallel
/// everywhere below ~2.7 MFLOP, so declining under the threshold gives nothing
/// up. Do NOT read the parallel column as the cost of threading in a real
/// stream of thousands of contractions; for that, time the replay.

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <array>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;

namespace {

/// One particle-particle ladder `ijab <- ijcd ; cdab`, the shape a tiled CCSD
/// residual expands to. Rank 2 is deliberately not measured: StringDispatch's
/// GEMM path takes those before try_packed_gemm sees them, so the packed engine
/// defers and there is no decision to make.
struct Case {
    int64_t     occ, vir; ///< block extents: M = occ^2, N = K = vir^2
    char const *note;
};

void bench(Case const &c) {
    auto const o = static_cast<size_t>(c.occ);
    auto const v = static_cast<size_t>(c.vir);

    RuntimeTensor<double> A("A", std::vector<size_t>{o, o, v, v}); // ijcd
    RuntimeTensor<double> B("B", std::vector<size_t>{v, v, v, v}); // cdab
    RuntimeTensor<double> C("C", std::vector<size_t>{o, o, v, v}); // ijab
    for (size_t i = 0; i < A.size(); i++) {
        A.data()[i] = 1.0 + 0.5 * static_cast<double>(i % 11);
    }
    for (size_t i = 0; i < B.size(); i++) {
        B.data()[i] = 2.0 - 0.25 * static_cast<double>(i % 7);
    }
    C.zero();

    packed_gemm::ContractionSpec spec;
    spec.c_indices      = {"i", "j", "a", "b"};
    spec.a_indices      = {"i", "j", "c", "d"};
    spec.b_indices      = {"c", "d", "a", "b"};
    spec.link_indices   = {"c", "d"};
    spec.target_indices = {"i", "j", "a", "b"};
    spec.all_indices    = {"i", "j", "a", "b", "c", "d"};

    auto call = [&]() {
        return packed_gemm::try_packed_gemm<RuntimeTensor<double>, RuntimeTensor<double>, RuntimeTensor<double>>(spec, 1.0, &C, 1.0, A, B);
    };
    int64_t const M = c.occ * c.occ, N = c.vir * c.vir, K = c.vir * c.vir;
    if (!call()) { // warm the plan cache; a replay never pays the first call
        fmt::println("[ParallelGate {:>22}] M={:5d} N={:5d} K={:5d}  DEFERRED by packed_gemm", c.note, M, N, K);
        return;
    }

    double const flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    int const    reps  = flops > 1e8 ? 20 : 200;
    auto const   t     = time_us("call", call, reps);

    fmt::println("[ParallelGate {:>22}] M={:5d} N={:5d} K={:5d}  {:11.0f} flops  {:9.2f} us  {:6.1f} GFLOP/s", c.note, M, N, K, flops,
                 t.avg, flops / (t.avg * 1e3));
    publish_benchmark_result(fmt::format("ParallelGate o{}v{}", c.occ, c.vir).c_str(), "t_call", static_cast<int>(flops / 1000.0), t);
}

/// Spanning both regimes the tiled lowering produces: per-tile contractions
/// (occupied blocks of 6, virtual of 8 at 50 spin orbitals) and the densified
/// whole-tensor ones (o=10, v=16 at 26 spin orbitals), plus the range around
/// them. Flops span 5e2 to 4e9.
constexpr std::array<Case, 10> kCases{{
    {.occ = 2, .vir = 2, .note = "tiny tile"},
    {.occ = 4, .vir = 4, .note = "toy tile"},
    {.occ = 6, .vir = 8, .note = "50-so per-tile"}, // the shape that dominates that replay
    {.occ = 8, .vir = 8, .note = "square tile"},
    {.occ = 8, .vir = 12, .note = "wider tile"},
    {.occ = 10, .vir = 16, .note = "26-so densified"}, // the shape that replay runs
    {.occ = 12, .vir = 20, .note = "larger densified"},
    {.occ = 16, .vir = 24, .note = "production-ish"},
    {.occ = 18, .vir = 32, .note = "50-so densified"},
    {.occ = 24, .vir = 40, .note = "large"},
}};

} // namespace

EINSUMS_TEST_CASE("Bench ParallelGate: where an OpenMP region starts paying", "[PackedGemm][ParallelGate][benchmark]") {
    LabeledSection0();
    fmt::println("[ParallelGate] min_parallel_flops = {}  omp_region_cost_ns = {:.0f}  max_threads = {}",
                 packed_gemm::cpu_config().min_parallel_flops, packed_gemm::cpu_config().omp_region_cost_ns,
#ifdef _OPENMP
                 omp_get_max_threads()
#else
                 1
#endif
    );
    for (auto const &c : kCases) {
        bench(c);
    }
}
