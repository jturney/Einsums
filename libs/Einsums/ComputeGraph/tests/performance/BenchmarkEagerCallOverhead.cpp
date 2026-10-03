//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkEagerCallOverhead.cpp
/// @brief The fixed cost of one eager cg::einsum, on contractions too small to hide it.
///
/// Each case is a contraction of a few dozen flops, so what it times is the call: the spec parse,
/// the checks, the route cascade and whatever boundary the call crosses on its way to a kernel.
/// One rep is a batch of calls, so the clock's resolution does not swamp a call that takes about a
/// microsecond.
///
/// The last case replays a captured graph of the same tiny contractions. Graph replay already runs
/// through the library's rank-erased entry, so a change to the eager path leaves it alone: when two
/// builds are compared, it is the control that says whether the machine moved between them.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;
namespace cg  = einsums::compute_graph;
namespace cgd = einsums::compute_graph::dispatch;

namespace {

constexpr int kCallsPerRep = 1000;
constexpr int kReps        = 30;

/// Time @p call in batches and report nanoseconds per call, best and mean over the reps.
template <typename Call>
void measure(std::string const &label, Call &&call) {
    auto const batch = [&]() {
        for (int i = 0; i < kCallsPerRep; ++i) {
            call();
        }
    };
    auto const   t    = time_us(label.c_str(), batch, kReps);
    double const best = 1000.0 * t.min / kCallsPerRep;
    double const mean = 1000.0 * t.avg / kCallsPerRep;
    fmt::println("[EagerCallOverhead {:28s}] best {:8.1f} ns/call  mean {:8.1f} ns/call  route={}", label, best, mean,
                 cgd::last_dispatch_route());
    std::string const key = fmt::format("EagerCallOverhead {}", label);
    publish_benchmark_result(key.c_str(), "t_batch", kCallsPerRep, t);
}

} // namespace

// NOLINTBEGIN(einsums-cg-call-outside-capture)
EINSUMS_TEST_CASE("Bench EagerCallOverhead: tiny eager contractions", "[ComputeGraph][EagerCallOverhead][benchmark]") {
    WAGGLE_ZONE_FUNC();

    auto A2 = create_random_tensor<double>("A2", 4, 4);
    auto B2 = create_random_tensor<double>("B2", 4, 4);
    auto C2 = create_zero_tensor<double>("C2", 4, 4);
    measure("gemm 4x4", [&] { cg::einsum("ij <- ik ; kj", &C2, A2, B2); });

    auto A3 = create_random_tensor<double>("A3", 4, 4, 4);
    auto v4 = create_random_tensor<double>("v4", 4);
    measure("rank3 x rank1", [&] { cg::einsum("ij <- ijk ; k", &C2, A3, v4); });

    auto s = create_zero_tensor<double>("s", 1);
    measure("dot 4x4", [&] { cg::einsum("<- ij ; ij", &s, A2, B2); });

    measure("elementwise 4x4", [&] { cg::einsum("ij <- ij ; ij", &C2, A2, B2); });

    RuntimeTensor<double> Ar(A2), Br(B2), Cr(C2);
    measure("gemm 4x4 runtime-rank", [&] { cg::einsum("ij <- ik ; kj", &Cr, Ar, Br); });

    auto Af = create_random_tensor<float>("Af", 4, 4);
    measure("mixed f*d->d 4x4", [&] { cg::einsum("ij <- ik ; kj", &C2, Af, B2); });
}
// NOLINTEND(einsums-cg-call-outside-capture)

EINSUMS_TEST_CASE("Bench EagerCallOverhead: control, replay of 100 tiny einsums", "[ComputeGraph][EagerCallOverhead][benchmark]") {
    WAGGLE_ZONE_FUNC();

    auto      A = create_random_tensor<double>("A", 4, 4);
    auto      B = create_random_tensor<double>("B", 4, 4);
    auto      C = create_zero_tensor<double>("C", 4, 4);
    cg::Graph graph("tiny_replay");
    {
        cg::CaptureGuard const guard(graph);
        for (int i = 0; i < 100; ++i) {
            cg::einsum("ij <- ik ; kj", 1.0, &C, 1.0, A, B);
        }
    }

    auto const   t  = time_us("replay 100 tiny", [&] { graph.execute(); }, kReps);
    double const ns = 1000.0 * t.min / 100.0;
    fmt::println("[EagerCallOverhead {:28s}] best {:8.1f} ns/node  (control: replay does not change with the eager path)",
                 "replay 100 x gemm 4x4", ns);
    publish_benchmark_result("EagerCallOverhead replay 100 tiny", "t_replay", 100, t);
}

#if defined(EINSUMS_HAVE_PROFILER)
// NOLINTBEGIN(einsums-cg-call-outside-capture)
EINSUMS_TEST_CASE("Bench EagerCallOverhead: what the profiler adds to one call", "[ComputeGraph][EagerCallOverhead][benchmark]") {
    // Zones per call come from the profiler's own push counter. Annotations have no counter, so their
    // share is what is left once the zones are paid for. On and off are interleaved batch by batch in
    // one process, so load and ordering fall on both alike, and the consumer is drained between
    // batches so the ring never fills.
    auto      &profiler = waggle::Profiler::instance();
    bool const was      = profiler.enabled();

    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto s = create_zero_tensor<double>("s", 1);

    auto const census = [&](std::string const &label, auto &&call) {
        constexpr int kCalls = 200;
        profiler.set_enabled(true);
        profiler.flush();
        auto const pushes0 = profiler.total_push_count();
        for (int i = 0; i < kCalls; ++i) {
            call();
        }
        double const zones = static_cast<double>(profiler.total_push_count() - pushes0) / kCalls;

        std::vector<double> on, off;
        for (int rep = 0; rep < 41; ++rep) {
            for (bool const enabled : {true, false}) {
                profiler.set_enabled(enabled);
                profiler.flush();
                auto const t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kCalls; ++i) {
                    call();
                }
                auto const t1 = std::chrono::steady_clock::now();
                if (rep > 0) {
                    (enabled ? on : off).push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / kCalls);
                }
            }
        }
        std::ranges::sort(on);
        std::ranges::sort(off);
        double const on_med = on[on.size() / 2], off_med = off[off.size() / 2];
        fmt::println("[EagerCallOverhead census {:22s}] {:4.1f} zones/call   on {:7.1f} ns   off {:7.1f} ns   profiler {:7.1f} ns/call "
                     "({:4.1f} ns per zone if zones were all of it)",
                     label, zones, on_med, off_med, on_med - off_med, zones > 0 ? (on_med - off_med) / zones : 0.0);
    };

    census("gemm 4x4", [&] { cg::einsum("ij <- ik ; kj", &C, A, B); });
    census("elementwise 4x4", [&] { cg::einsum("ij <- ij ; ij", &C, A, B); });
    census("dot 4x4", [&] { cg::einsum("<- ij ; ij", &s, A, B); });
    census("rank3 x rank1", [&] {
        static auto A3 = create_random_tensor<double>("A3", 4, 4, 4);
        static auto v4 = create_random_tensor<double>("v4", 4);
        cg::einsum("ij <- ijk ; k", &C, A3, v4);
    });

    profiler.set_enabled(was);
}
// NOLINTEND(einsums-cg-call-outside-capture)
#endif
