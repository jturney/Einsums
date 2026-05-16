//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Memory bandwidth benchmark (STREAM-like).
//
// Measures achievable memory bandwidth with four kernels:
//   Copy:  b[i] = a[i]
//   Scale: b[i] = alpha * a[i]
//   Add:   c[i] = a[i] + b[i]
//   Triad: c[i] = a[i] + alpha * b[i]
//
// Reports both time (us) and bandwidth (GB/s) for comparison
// against peak theoretical bandwidth.

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

void bench_copy(size_t N) {
    LabeledSection0();
    std::vector<double> a(N), b(N);
    for (size_t i = 0; i < N; ++i)
        a[i] = static_cast<double>(i);

    ProfileAnnotate("operation", std::string("copy"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("bytes", int64_t(2 * N * 8));
    auto s = time_us("stream-copy", [&] {
        for (size_t i = 0; i < N; ++i)
            b[i] = a[i];
    });

    // Bytes: read N + write N doubles
    double bytes = 2.0 * N * sizeof(double);
    double gb_s  = bytes / (s.avg * 1e3); // GB/s = bytes / (us * 1e3)
    std::printf("[stream-copy N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", N, s.avg,
                s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup, (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[stream-copy-bw N=%zu] BW: %.2f GB/s\n", N, gb_s);
    std::fflush(stdout);
    ProfileAnnotate("bw_gbs", gb_s);
    publish_benchmark_result("stream-copy", "t_stream", static_cast<int>(N), s);
}

void bench_scale(size_t N) {
    LabeledSection0();
    std::vector<double> a(N), b(N);
    double              alpha = 3.0;
    for (size_t i = 0; i < N; ++i)
        a[i] = static_cast<double>(i);

    ProfileAnnotate("operation", std::string("scale"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("bytes", int64_t(2 * N * 8));
    auto s = time_us("stream-scale", [&] {
        for (size_t i = 0; i < N; ++i)
            b[i] = alpha * a[i];
    });

    double bytes = 2.0 * N * sizeof(double);
    double gb_s  = bytes / (s.avg * 1e3);
    std::printf("[stream-scale N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", N, s.avg,
                s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup, (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[stream-scale-bw N=%zu] BW: %.2f GB/s\n", N, gb_s);
    std::fflush(stdout);
    ProfileAnnotate("bw_gbs", gb_s);
    publish_benchmark_result("stream-scale", "t_stream", static_cast<int>(N), s);
}

void bench_add(size_t N) {
    LabeledSection0();
    std::vector<double> a(N), b(N), c(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = static_cast<double>(i) * 2.0;
    }

    ProfileAnnotate("operation", std::string("add"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("bytes", int64_t(3 * N * 8));
    auto s = time_us("stream-add", [&] {
        for (size_t i = 0; i < N; ++i)
            c[i] = a[i] + b[i];
    });

    double bytes = 3.0 * N * sizeof(double); // 2 reads + 1 write
    double gb_s  = bytes / (s.avg * 1e3);
    std::printf("[stream-add N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", N, s.avg,
                s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup, (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[stream-add-bw N=%zu] BW: %.2f GB/s\n", N, gb_s);
    std::fflush(stdout);
    ProfileAnnotate("bw_gbs", gb_s);
    publish_benchmark_result("stream-add", "t_stream", static_cast<int>(N), s);
}

void bench_triad(size_t N) {
    LabeledSection0();
    std::vector<double> a(N), b(N), c(N);
    double              alpha = 3.0;
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = static_cast<double>(i) * 2.0;
    }

    ProfileAnnotate("operation", std::string("triad"));
    ProfileAnnotate("dtype", std::string("double"));
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("bytes", int64_t(3 * N * 8));
    auto s = time_us("stream-triad", [&] {
        for (size_t i = 0; i < N; ++i)
            c[i] = a[i] + alpha * b[i];
    });

    double bytes = 3.0 * N * sizeof(double);
    double gb_s  = bytes / (s.avg * 1e3);
    std::printf("[stream-triad N=%zu] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", N, s.avg,
                s.min, s.max, s.stddev, (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0, s.warmup, (s.avg > 0) ? (s.warmup / s.avg) : 0.0);
    std::printf("[stream-triad-bw N=%zu] BW: %.2f GB/s\n", N, gb_s);
    std::fflush(stdout);
    ProfileAnnotate("bw_gbs", gb_s);
    publish_benchmark_result("stream-triad", "t_stream", static_cast<int>(N), s);
}

} // namespace

EINSUMS_TEST_CASE("STREAM Bandwidth", "[performance][bandwidth]") {
    // Sizes chosen to span L1 cache through main memory
    // 1K doubles = 8 KB (L1), 64K = 512 KB (L2), 1M = 8 MB (L3), 16M = 128 MB (DRAM)
    progress_init(16); // 4 sizes × 4 ops
    for (size_t N : {1024UL, 65536UL, 1048576UL, 16777216UL}) {
        bench_copy(N);
        bench_scale(N);
        bench_add(N);
        bench_triad(N);
    }
}
