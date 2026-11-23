//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkPackedGemmSmall.cpp
/// @brief What a small contraction that is one GEMM in disguise costs on its route.
///
/// A contraction with a higher-rank operand whose index groups are adjacent in column-major storage
/// is a single GEMM over reshaped operands: ``ijk <- ijl ; lk`` is an (ij) x l times l x k product.
/// Such contractions used to route through PackedGemm, which reached the same vendor GEMM after a
/// microsecond of analysis; they now take the merged-axis route. Each case times three spellings of
/// the same arithmetic on the same buffers:
///
///   packed - the einsum as written, on the higher-rank tensors (asserted to route gemm_fused_runtime)
///   direct - the same einsum on two-dimensional views of those buffers (the vendor route)
///   vendor - the vendor GEMM or GEMV called directly, the floor
///
/// so packed minus direct is what the route costs and direct minus vendor is the string einsum's
/// own call. One rep is a batch of calls, best of several, so a sub-microsecond call is resolved.

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;
namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

namespace {

/// Best-of-reps time of one call, in nanoseconds, with enough calls per rep to resolve it.
template <typename Fn>
double best_ns_per_call(Fn &&fn, size_t flops) {
    size_t const calls = std::clamp<size_t>(2'000'000 / std::max<size_t>(flops, 1), 20, 4000);
    for (size_t i = 0; i < calls; ++i) {
        fn();
    }
    double best = std::numeric_limits<double>::infinity();
    for (int rep = 0; rep < 15; ++rep) {
        auto const t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < calls; ++i) {
            fn();
        }
        auto const t1 = std::chrono::steady_clock::now();
        best          = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls));
    }
    return best;
}

RuntimeTensor<double> filled(std::string const &name, std::vector<size_t> const &dims) {
    RuntimeTensor<double> t(name, dims);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = std::sin(0.37 * static_cast<double>(i)) + 0.5;
    }
    return t;
}

void report(std::string_view label, size_t n, size_t flops, double packed, double direct, double vendor) {
    fmt::println("[PackedGemmSmall {:22}] n={:3d} flops={:9d}  packed {:9.1f} ns  direct {:9.1f} ns  vendor {:9.1f} ns  "
                 "packed-direct {:8.1f} ns  packed/vendor {:5.2f}x",
                 label, n, flops, packed, direct, vendor, packed - direct, packed / vendor);
    std::string const key = fmt::format("PackedGemmSmall {} n={}", label, n);
    TimingStats       stats{};
    stats.avg = packed / 1000.0;
    publish_benchmark_result(key.c_str(), "t_packed", static_cast<int>(n), stats);
}

void require_close(RuntimeTensor<double> const &a, RuntimeTensor<double> const &b) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::abs(a.data()[i] - b.data()[i]) <= 1e-12 * (1.0 + std::abs(b.data()[i])));
    }
}

constexpr std::array<size_t, 6> kSizes{2, 4, 8, 16, 32, 64};

} // namespace

EINSUMS_TEST_CASE("Bench PackedGemmSmall: one-GEMM contractions, packed vs direct vs vendor",
                  "[ComputeGraph][PackedGemmSmall][benchmark]") {
    auto const i = [](size_t v) { return static_cast<blas::int_t>(v); };

    for (size_t const n : kSizes) {
        size_t const n2 = n * n;

        // ijk <- ijl ; lk : M = n^2 (ij fused), K = n, N = n.
        {
            auto                      A = filled("A", {n, n, n});
            auto                      B = filled("B", {n, n});
            auto                      C = filled("C", {n, n, n});
            auto                      R = filled("R", {n, n, n});
            RuntimeTensorView<double> A2(A, std::vector<size_t>{n2, n});
            RuntimeTensorView<double> C2(C, std::vector<size_t>{n2, n});
            size_t const              flops = 2 * n2 * n * n;
            cg::einsum("ijk <- ijl ; lk", &C, A, B);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_fused_runtime"});
            double const packed = best_ns_per_call([&] { cg::einsum("ijk <- ijl ; lk", &C, A, B); }, flops);
            cg::einsum("ij <- ik ; kj", &C2, A2, B);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_direct_runtime"});
            double const direct = best_ns_per_call([&] { cg::einsum("ij <- ik ; kj", &C2, A2, B); }, flops);
            double const vendor = best_ns_per_call(
                [&] { blas::gemm('n', 'n', i(n2), i(n), i(n), 1.0, A.data(), i(n2), B.data(), i(n), 0.0, R.data(), i(n2)); }, flops);
            cg::einsum("ijk <- ijl ; lk", &C, A, B);
            require_close(C, R);
            report("ijk <- ijl ; lk", n, flops, packed, direct, vendor);
        }

        // ijk <- il ; ljk : M = n, K = n, N = n^2 (jk fused).
        {
            auto                      A = filled("A", {n, n});
            auto                      B = filled("B", {n, n, n});
            auto                      C = filled("C", {n, n, n});
            auto                      R = filled("R", {n, n, n});
            RuntimeTensorView<double> B2(B, std::vector<size_t>{n, n2});
            RuntimeTensorView<double> C2(C, std::vector<size_t>{n, n2});
            size_t const              flops = 2 * n * n * n2;
            cg::einsum("ijk <- il ; ljk", &C, A, B);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_fused_runtime"});
            double const packed = best_ns_per_call([&] { cg::einsum("ijk <- il ; ljk", &C, A, B); }, flops);
            cg::einsum("ij <- ik ; kj", &C2, A, B2);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_direct_runtime"});
            double const direct = best_ns_per_call([&] { cg::einsum("ij <- ik ; kj", &C2, A, B2); }, flops);
            double const vendor = best_ns_per_call(
                [&] { blas::gemm('n', 'n', i(n), i(n2), i(n), 1.0, A.data(), i(n), B.data(), i(n), 0.0, R.data(), i(n)); }, flops);
            cg::einsum("ijk <- il ; ljk", &C, A, B);
            require_close(C, R);
            report("ijk <- il ; ljk", n, flops, packed, direct, vendor);
        }

        // ij <- ikl ; klj : M = n, K = n^2 (kl fused), N = n.
        {
            auto                      A = filled("A", {n, n, n});
            auto                      B = filled("B", {n, n, n});
            auto                      C = filled("C", {n, n});
            auto                      R = filled("R", {n, n});
            RuntimeTensorView<double> A2(A, std::vector<size_t>{n, n2});
            RuntimeTensorView<double> B2(B, std::vector<size_t>{n2, n});
            size_t const              flops = 2 * n * n2 * n;
            cg::einsum("ij <- ikl ; klj", &C, A, B);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_fused_runtime"});
            double const packed = best_ns_per_call([&] { cg::einsum("ij <- ikl ; klj", &C, A, B); }, flops);
            cg::einsum("ij <- ik ; kj", &C, A2, B2);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"gemm_direct_runtime"});
            double const direct = best_ns_per_call([&] { cg::einsum("ij <- ik ; kj", &C, A2, B2); }, flops);
            double const vendor = best_ns_per_call(
                [&] { blas::gemm('n', 'n', i(n), i(n), i(n2), 1.0, A.data(), i(n), B.data(), i(n2), 0.0, R.data(), i(n)); }, flops);
            cg::einsum("ij <- ikl ; klj", &C, A, B);
            require_close(C, R);
            report("ij <- ikl ; klj", n, flops, packed, direct, vendor);
        }

        // ij <- ijk ; k : a GEMV, M = n^2 (ij fused), N = n.
        {
            auto                      A = filled("A", {n, n, n});
            auto                      x = filled("x", {n});
            auto                      C = filled("C", {n, n});
            auto                      R = filled("R", {n, n});
            RuntimeTensorView<double> A2(A, std::vector<size_t>{n2, n});
            RuntimeTensorView<double> C1(C, std::vector<size_t>{n2});
            size_t const              flops = 2 * n2 * n;
            cg::einsum("ij <- ijk ; k", &C, A, x);
            REQUIRE(cgd::last_dispatch_route() == std::string_view{"packed_gemm"});
            double const packed = best_ns_per_call([&] { cg::einsum("ij <- ijk ; k", &C, A, x); }, flops);
            cg::einsum("i <- ik ; k", &C1, A2, x);
            std::string const direct_route{cgd::last_dispatch_route()};
            double const      direct = best_ns_per_call([&] { cg::einsum("i <- ik ; k", &C1, A2, x); }, flops);
            double const      vendor =
                best_ns_per_call([&] { blas::gemv('n', i(n2), i(n), 1.0, A.data(), i(n2), x.data(), 1, 0.0, R.data(), 1); }, flops);
            cg::einsum("ij <- ijk ; k", &C, A, x);
            require_close(C, R);
            report("ij <- ijk ; k (gemv)", n, flops, packed, direct, vendor);
            INFO("direct route " << direct_route);
        }

        // ijb <- ikb ; kjb : n independent n x n GEMMs, the batch index slowest.
        {
            auto         A     = filled("A", {n, n, n});
            auto         B     = filled("B", {n, n, n});
            auto         C     = filled("C", {n, n, n});
            auto         R     = filled("R", {n, n, n});
            size_t const flops = 2 * n * n * n * n;
            cg::einsum("ijb <- ikb ; kjb", &C, A, B);
            std::string const route{cgd::last_dispatch_route()};
            double const      packed = best_ns_per_call([&] { cg::einsum("ijb <- ikb ; kjb", &C, A, B); }, flops);
            double const      vendor = best_ns_per_call(
                [&] {
                    for (size_t b = 0; b < n; ++b) {
                        blas::gemm('n', 'n', i(n), i(n), i(n), 1.0, A.data() + b * n2, i(n), B.data() + b * n2, i(n), 0.0,
                                   R.data() + b * n2, i(n));
                    }
                },
                flops);
            cg::einsum("ijb <- ikb ; kjb", &C, A, B);
            require_close(C, R);
            report(fmt::format("ijb<-ikb;kjb [{}]", route.substr(0, 9)), n, flops, packed, std::nan(""), vendor);
        }
    }
}
