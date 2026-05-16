//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ParallelForReduce.cpp
/// @brief Demonstrates parallel_for and parallel_reduce for data-parallel workloads.
///
/// Shows how to:
///   - Use parallel_for to fill and transform arrays in parallel
///   - Use parallel_reduce to accumulate results with thread-local buffers
///   - Compare performance with sequential execution

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace tp = einsums::task_pool;

int einsums_main() {
    using namespace einsums;

    auto            &pool = tp::TaskPool::get_singleton();
    constexpr size_t N    = 10'000'000;

    // ── 1. parallel_for: fill an array ───────────────────────────────────────
    println("--- parallel_for: fill {} elements ---", N);

    std::vector<double> data(N);

    auto t0 = std::chrono::high_resolution_clock::now();
    pool.parallel_for("fill", 0, N, [&data](size_t i) { data[i] = std::sin(static_cast<double>(i) * 0.001); });
    auto t1 = std::chrono::high_resolution_clock::now();

    double par_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  parallel_for: {:.1f} ms", par_ms);

    // Sequential comparison
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        data[i] = std::sin(static_cast<double>(i) * 0.001);
    }
    t1 = std::chrono::high_resolution_clock::now();

    double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  sequential:   {:.1f} ms", seq_ms);
    println("  speedup:      {:.1f}x", seq_ms / par_ms);

    // ── 2. parallel_reduce: sum of squares ───────────────────────────────────
    println("\n--- parallel_reduce: sum of squares ---");

    t0             = std::chrono::high_resolution_clock::now();
    double par_sum = pool.parallel_reduce<double>(
        "sum_squares", 0, N, []() { return 0.0; },                    // init
        [&data](size_t i, double &acc) { acc += data[i] * data[i]; }, // body
        [](double &global, double const &local) { global += local; }  // combine
    );
    t1     = std::chrono::high_resolution_clock::now();
    par_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Sequential comparison
    auto   t2      = std::chrono::high_resolution_clock::now();
    double seq_sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        seq_sum += data[i] * data[i];
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    seq_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();

    println("  parallel result:   {:.6f}", par_sum);
    println("  sequential result: {:.6f}", seq_sum);
    println("  match: {}", std::abs(par_sum - seq_sum) / std::abs(seq_sum) < 1e-10 ? "yes" : "NO");
    println("  parallel:   {:.1f} ms", par_ms);
    println("  sequential: {:.1f} ms", seq_ms);
    println("  speedup:    {:.1f}x", seq_ms / par_ms);

    // ── 3. parallel_reduce: dot product ──────────────────────────────────────
    println("\n--- parallel_reduce: dot product ---");

    std::vector<double> a(N), b(N);
    pool.parallel_for("init_ab", 0, N, [&a, &b](size_t i) {
        a[i] = static_cast<double>(i) * 0.001;
        b[i] = 1.0 / (static_cast<double>(i) + 1.0);
    });

    double dot = pool.parallel_reduce<double>(
        "dot_product", 0, N, []() { return 0.0; }, [&a, &b](size_t i, double &acc) { acc += a[i] * b[i]; },
        [](double &g, double const &l) { g += l; });

    // Reference
    double ref_dot = 0.0;
    for (size_t i = 0; i < N; i++)
        ref_dot += a[i] * b[i];

    println("  parallel:   {:.6f}", dot);
    println("  sequential: {:.6f}", ref_dot);
    println("  match: {}", std::abs(dot - ref_dot) / std::abs(ref_dot) < 1e-10 ? "yes" : "NO");

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
