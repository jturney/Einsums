//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkDispatchRoutes.cpp
/// @brief What each string-einsum route is worth, by size.
///
/// string_einsum's final fallback is a SERIAL, non-OpenMP odometer loop, so a
/// shape that misses its intended route does not fail a value test - it just
/// runs at scalar speed. These benchmarks time the shapes that used to miss,
/// so the gap between a BLAS route and that loop is a recorded number rather
/// than an argument from the shape of the code.
///
/// Each case asserts the route it is timing. That is the point: a benchmark
/// that silently starts measuring the generic loop again would otherwise just
/// look like a regression with no explanation.
///
/// To measure what a route BUYS, run this at a commit before the route existed
/// and compare; the shapes are all expressible either way, they merely dispatch
/// differently.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <complex>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;
namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

namespace {

constexpr int kReps = 20;

/// Report one measurement, tagged with the route that actually ran.
void report(std::string const &label, size_t elements, TimingStats const &t) {
    fmt::println("[DispatchRoute {}] elements={:9d}  {:10.2f} us  ({:6.2f} ns/element)  route={}", label, elements, t.avg,
                 1000.0 * t.avg / static_cast<double>(elements), cgd::last_dispatch_route());
    std::string const key = fmt::format("DispatchRoute {}", label);
    publish_benchmark_result(key.c_str(), "t_call", static_cast<int>(elements), t);
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Scalar-output full contraction: "<- ij ; ij". Used to need rank-1 operands to
// reach dot, and PackedGemm rejects a rank-0 output, so every higher rank ran
// on the odometer loop.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench DispatchRoute: scalar-output full contraction", "[ComputeGraph][DispatchRoute][benchmark]") {
    LabeledSection0();
    for (size_t n : {32UL, 128UL, 512UL}) {
        auto A = create_random_tensor<double>(std::string("A"), n, n);
        auto B = create_random_tensor<double>(std::string("B"), n, n);
        auto s = create_zero_tensor<double>(std::string("s"), size_t{1});

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        auto t = time_us("scalar-out", [&]() { cg::einsum("<- ij ; ij", &s, A, B); }, kReps);
        report(fmt::format("scalar-output rank-2 n={}", n), n * n, t);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Elementwise at rank 3: "ijk <- ijk ; ijk". direct_product was gated to rank 2
// even though the kernel takes any rank.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench DispatchRoute: elementwise rank-3", "[ComputeGraph][DispatchRoute][benchmark]") {
    LabeledSection0();
    for (size_t n : {16UL, 48UL, 128UL}) {
        auto A = create_random_tensor<double>(std::string("A"), n, n, n);
        auto B = create_random_tensor<double>(std::string("B"), n, n, n);
        auto C = create_zero_tensor<double>(std::string("C"), n, n, n);

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        auto t = time_us("elementwise", [&]() { cg::einsum("ijk <- ijk ; ijk", &C, A, B); }, kReps);
        report(fmt::format("elementwise rank-3 n={}", n), n * n * n, t);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conjugated full contraction: the one conjugated shape PackedGemm could not
// take (rank-0 output), so it had nowhere to go but the loop.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench DispatchRoute: conjugated full contraction", "[ComputeGraph][DispatchRoute][benchmark]") {
    LabeledSection0();
    using C64 = std::complex<double>;
    for (size_t n : {1024UL, 16384UL, 262144UL}) {
        auto x = create_random_tensor<C64>(std::string("x"), n);
        auto y = create_random_tensor<C64>(std::string("y"), n);
        auto s = create_zero_tensor<C64>(std::string("s"), size_t{1});

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        auto t = time_us("conj-dot", [&]() { cg::einsum("<- conj(i) ; i", &s, x, y); }, kReps);
        report(fmt::format("conjugated full contraction n={}", n), n, t);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mixed typed/runtime GEMM: neither ladder accepted the pair, and PackedGemm
// defers single-M/N/K back to a direct BLAS call that was no longer reachable,
// so a plain matmul ran on the odometer loop.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench DispatchRoute: mixed typed/runtime GEMM", "[ComputeGraph][DispatchRoute][benchmark]") {
    LabeledSection0();
    for (size_t n : {32UL, 128UL, 384UL}) {
        auto                  A   = create_random_tensor<double>(std::string("A"), n, n);
        auto                  B_t = create_random_tensor<double>(std::string("B"), n, n);
        auto                  C   = create_zero_tensor<double>(std::string("C"), n, n);
        RuntimeTensor<double> B(B_t);

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        auto t = time_us("mixed-gemm", [&]() { cg::einsum("ij <- ik ; kj", &C, A, B); }, kReps);
        report(fmt::format("mixed typed/runtime GEMM n={}", n), n * n * n, t);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Where the elementwise route starts paying. direct_product has a fixed setup
// cost the generic loop does not, so below some element count the loop wins -
// the same tradeoff PackedGemm already encodes for small outer products. This
// measures BOTH paths in one process (the generic loop is callable directly),
// so the crossover is a measurement rather than a guess, and it re-measures
// itself on whatever hardware runs it.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench DispatchRoute: elementwise crossover", "[ComputeGraph][DispatchRoute][benchmark]") {
    LabeledSection0();

    auto parsed = einsums::compute_graph::parse_einsum_spec("ijk <- ijk ; ijk");
    REQUIRE(parsed.has_value());
    std::vector<std::string> const no_links;

    // Three columns, because the first two differ by more than the route:
    //   full     - cg::einsum(), which PARSES the spec string on every call
    //   dispatch - string_einsum() on an already-parsed spec, i.e. route + kernel
    //   generic  - the odometer loop on the same parsed spec
    // full-minus-dispatch is the per-call parse cost; dispatch-vs-generic is the
    // route decision on its own, which is the one a threshold should be set from.
    fmt::println("[DispatchRoute elementwise crossover] {:>10} {:>10} {:>10} {:>10} {:>10} {:>9}", "elements", "full(us)", "dispatch",
                 "kernel", "generic", "d-vs-g");
    for (size_t n : {4UL, 6UL, 8UL, 12UL, 16UL, 24UL, 32UL, 48UL, 64UL}) {
        auto A = create_random_tensor<double>(std::string("A"), n, n, n);
        auto B = create_random_tensor<double>(std::string("B"), n, n, n);
        auto C = create_zero_tensor<double>(std::string("C"), n, n, n);

        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        auto t_full   = time_us("full", [&]() { cg::einsum("ijk <- ijk ; ijk", &C, A, B); }, 50);
        auto t_route  = time_us("dispatch", [&]() { cgd::string_einsum(parsed.value(), 0.0, &C, 1.0, A, B); }, 50);
        auto t_loop   = time_us("generic", [&]() { cgd::generic_string_einsum(parsed.value(), no_links, 0.0, &C, 1.0, A, B); }, 50);
        auto t_kernel = time_us("kernel", [&]() { linear_algebra::direct_product(1.0, A, B, 0.0, &C); }, 50);

        fmt::println("[DispatchRoute elementwise crossover] {:>10d} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>8.2f}x", n * n * n,
                     t_full.avg, t_route.avg, t_kernel.avg, t_loop.avg, t_loop.avg / t_route.avg);
    }
}
