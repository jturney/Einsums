//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

namespace einsums::performance {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Timing statistics
// ---------------------------------------------------------------------------

struct TimingStats {
    double avg;
    double min;
    double max;
    double stddev;
    double warmup;
    int    reps;

    /// Implicit conversion to double returns the average, for backward compatibility
    /// with code that uses `double t = time_us(...)`.
    operator double() const { return avg; }
};

/// Run `fn` `reps` times (after one warmup) and return detailed timing stats.
/// Each rep is timed individually for min/max/stddev analysis.
/// The warmup pass is timed separately to show cold-start cost (JIT, cache).
template <typename Fn>
TimingStats time_us(char const *label, Fn &&fn, int reps = 10) {
    LabeledSection("{}", label);

    // Warmup — timed separately to show cold-start cost
    double warmup_us;
    {
        LabeledSection("warmup");
        auto w0 = Clock::now();
        fn();
        auto w1   = Clock::now();
        warmup_us = std::chrono::duration<double, std::micro>(w1 - w0).count();
    }
    ProfileAnnotate("warmup_us", warmup_us);
    ProfileAnnotate("reps", static_cast<int64_t>(reps));

    // Time each rep individually
    std::vector<double> times(reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1  = Clock::now();
        times[r] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double sum = 0.0, mn = times[0], mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn)
            mn = t;
        if (t > mx)
            mx = t;
    }
    double avg = sum / reps;

    double var = 0.0;
    for (double t : times) {
        double d = t - avg;
        var += d * d;
    }
    double sd = std::sqrt(var / reps);

    ProfileAnnotate("avg_us", avg);
    ProfileAnnotate("min_us", mn);
    ProfileAnnotate("max_us", mx);
    ProfileAnnotate("stddev_us", sd);

    return {avg, mn, mx, sd, warmup_us, reps};
}

/// Convenience overload without a label (for benchmarks that manage their own output).
template <typename Fn>
TimingStats time_us(Fn &&fn, int reps = 10) {
    return time_us("bench", std::forward<Fn>(fn), reps);
}

// ---------------------------------------------------------------------------
// Progress tracking and reporting
// ---------------------------------------------------------------------------

/// Global progress counters. Call progress_init() at the start of a test case,
/// then report() increments and prints a progress bar to stderr.
inline int &bench_done() {
    static int val = 0;
    return val;
}

inline int &bench_total() {
    static int val = 0;
    return val;
}

inline void progress_init(int total) {
    bench_done()  = 0;
    bench_total() = total;
}

/// Print a benchmark result to stdout (parseable by the regression system)
/// and a progress bar to stderr.
///
/// Output format:
///   [label N=size] Time: avg us  min: ...  max: ...  stddev: ...  cv: ...%  warmup: ... us (Nx)
inline void report(char const *label, int N, TimingStats const &s) {
    double cv      = (s.avg > 0) ? (s.stddev / s.avg * 100.0) : 0.0;
    double w_ratio = (s.avg > 0) ? (s.warmup / s.avg) : 0.0;
    std::printf("[%s N=%d] Time: %.2f us  min: %.2f  max: %.2f  stddev: %.2f  cv: %.1f%%  warmup: %.2f us (%.1fx)\n", label, N, s.avg,
                s.min, s.max, s.stddev, cv, s.warmup, w_ratio);
    std::fflush(stdout);

    int total = bench_total();
    if (total > 0) {
        int done   = ++bench_done();
        int pct    = done * 100 / total;
        int filled = pct / 2;
        int empty  = 50 - filled;
        std::fprintf(stderr, "  [%.*s%.*s] %3d%% (%d/%d)\n", filled, "##################################################", empty,
                     "                                                  ", pct, done, total);
        std::fflush(stderr);
    }
}

// ---------------------------------------------------------------------------
// Tensor fill helpers
// ---------------------------------------------------------------------------

/// Fill any rank tensor with deterministic values in (0, 1].
template <typename T, size_t R>
void fill(einsums::Tensor<T, R> &t) {
    size_t total = t.size();
    for (size_t n = 0; n < total; ++n) {
        t.data()[n] = static_cast<T>(n + 1) / static_cast<T>(total);
    }
}

/// Fill a symmetric positive definite matrix (diagonally dominant).
template <typename T>
void fill_spd(einsums::Tensor<T, 2> &A) {
    int N = static_cast<int>(A.dim(0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            T val   = static_cast<T>((i + 1) * (j + 1)) / static_cast<T>(N * N);
            A(i, j) = val;
            A(j, i) = val;
        }
        A(i, i) += static_cast<T>(N); // Make diagonally dominant
    }
}

// ---------------------------------------------------------------------------
// Benchmark result publishing (sends to profiler server)
// ---------------------------------------------------------------------------

/// Publish a benchmark result to the profiler server as a type="benchmark_result" event.
/// Snapshots all annotations from the current zone and all ancestor zones.
/// Also prints human-readable output to stdout via report().
///
/// @param label  Benchmark label (e.g., "gemm N=256")
/// @param metric Metric name (e.g., "t_einsum", "t_generic")
/// @param N      Problem size (for the human-readable report line)
/// @param stats  Timing statistics from time_us()
inline void publish_benchmark_result(char const *label, char const *metric, int N, TimingStats const &stats) {
    // 1. Print human-readable output to stdout (backward compatibility)
    report(label, N, stats);

    // 2. Send structured result to profiler server
#if defined(EINSUMS_HAVE_PROFILER)
    auto &prof = profile::Profiler::instance();
    auto *srv  = prof.server();
    if (srv) {
        // Collect annotations from the full zone stack
        std::string annotations_json = "{";
        auto       *consumer         = prof.consumer();
        if (consumer) {
            // Flush to ensure annotations are aggregated
            consumer->flush();

            auto lock   = consumer->lock_shared();
            auto tid    = profile::Profiler::current_thread_id();
            auto merged = consumer->collect_zone_annotations(tid);

            bool first = true;
            for (auto const &[key, val] : merged) {
                if (!first)
                    annotations_json += ",";
                first = false;
                annotations_json += "\"" + key + "\":\"" + val + "\"";
            }
        }
        annotations_json += "}";

        profile::BenchmarkResultEntry entry;
        // Include N in the label so each size is a distinct benchmark in the DB
        if (N > 0) {
            char lbl_buf[512];
            std::snprintf(lbl_buf, sizeof(lbl_buf), "%s N=%d", label, N);
            entry.label = lbl_buf;
        } else {
            entry.label = label;
        }
        entry.metric           = metric;
        entry.value_us         = stats.avg;
        entry.min_us           = stats.min;
        entry.max_us           = stats.max;
        entry.stddev_us        = stats.stddev;
        entry.warmup_us        = stats.warmup;
        entry.reps             = stats.reps;
        entry.annotations_json = std::move(annotations_json);

        srv->benchmark_queue().push(std::move(entry));
    }
#endif
}

/// Convenience overload that extracts N from the label (if present).
inline void publish_benchmark_result(char const *label, char const *metric, TimingStats const &stats) {
    publish_benchmark_result(label, metric, 0, stats);
}

} // namespace einsums::performance
