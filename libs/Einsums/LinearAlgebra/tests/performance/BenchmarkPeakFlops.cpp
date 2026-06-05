//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Peak FLOP/s calibration benchmark.
//
// Runs tight FMA (fused multiply-add) loops to measure the maximum
// achievable FLOP/s on this hardware. This provides the "roof" for
// roofline model analysis.

#include <Einsums/Performance.hpp>

#include <cstdio>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Profile/Profile.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums::performance;

namespace {

/// Tight FMA loop: each iteration does 2 FLOPs (multiply + add).
/// Uses multiple accumulators to saturate FMA units.
double peak_flops_kernel(size_t iterations) {
    double       a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3;
    double       a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
    double const m = 0.999;

    for (size_t i = 0; i < iterations; ++i) {
        a0 = a0 * m + m;
        a1 = a1 * m + m;
        a2 = a2 * m + m;
        a3 = a3 * m + m;
        a4 = a4 * m + m;
        a5 = a5 * m + m;
        a6 = a6 * m + m;
        a7 = a7 * m + m;
    }

    // Prevent compiler from optimizing away
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

void bench_peak_single_thread(size_t N) {
    LabeledSection0();
    double volatile sink = 0;

    auto s = time_us("peak-flops-1t", [&] { sink = peak_flops_kernel(N); });

    // 8 accumulators × 2 FLOPs (mul + add) × N iterations
    double flops  = 8.0 * 2.0 * N;
    double gflops = flops / (s.avg * 1e3);
    std::printf("[peak-flops-1t N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", N, s.avg,
                s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup, (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[peak-flops-1t-gflops N=%zu] GFLOP/s: %.2f\n", N, gflops);
    std::fflush(stdout);
    ProfileAnnotate("gflops", gflops);
    ProfileAnnotate("threads", int64_t(1));
    publish_benchmark_result("peak-flops-1t", "t_peak", static_cast<int>(N), s);
    (void)sink;
}

void bench_peak_all_threads(size_t N) {
    LabeledSection0();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    double volatile sink = 0;
    auto s               = time_us("peak-flops-all", [&] {
        double total = 0;
#ifdef _OPENMP
#    pragma omp parallel reduction(+ : total)
#endif
        { total += peak_flops_kernel(N); }
        sink = total;
    });

    double flops  = 8.0 * 2.0 * N * nthreads;
    double gflops = flops / (s.avg * 1e3);
    std::printf("[peak-flops-%dt N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", nthreads,
                N, s.avg, s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup,
                (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[peak-flops-%dt-gflops N=%zu] GFLOP/s: %.2f\n", nthreads, N, gflops);
    std::fflush(stdout);
    ProfileAnnotate("gflops", gflops);
    ProfileAnnotate("threads", int64_t(nthreads));
    publish_benchmark_result("peak-flops-all", "t_peak", static_cast<int>(N), s);
    (void)sink;
}

} // namespace

EINSUMS_TEST_CASE("Peak FLOP/s", "[performance][peak]") {
    // Use enough iterations to run ~10ms per call
    progress_init(4);
    size_t N = 10000000; // 10M iterations

    bench_peak_single_thread(N);
    bench_peak_all_threads(N);
    // Repeat at 100M for more stable measurement
    N = 100000000;
    bench_peak_single_thread(N);
    bench_peak_all_threads(N);
}
