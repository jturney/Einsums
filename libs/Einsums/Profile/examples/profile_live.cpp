//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// Long-running profiler example for testing the live TUI viewer.
///
/// Run this program, then in another terminal:
///     python devtools/profiling/profile_viewer.py
///
/// The program runs for ~60 seconds, simulating a workload with nested
/// profiling zones, annotations, and multiple threads.

#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

using namespace einsums::profile;

namespace {
std::atomic<bool> g_running{true};

void simulate_microkernel(int M, int N, int K) {
    LabeledSection("microkernel");
    ProfileAnnotate("M", static_cast<int64_t>(M));
    ProfileAnnotate("N", static_cast<int64_t>(N));
    ProfileAnnotate("K", static_cast<int64_t>(K));
    ProfileAnnotate("flops", static_cast<int64_t>(2) * M * N * K);

    // Simulate some work proportional to problem size
    double volatile sum = 0.0;
    for (int i = 0; i < M * N; ++i)
        sum += std::sin(static_cast<double>(i));
}

// NOLINTNEXTLINE(readability-identifier-naming)
void simulate_pack_A(int MC, int KC) {
    LabeledSection("pack_A");
    ProfileAnnotate("MC", static_cast<int64_t>(MC));
    ProfileAnnotate("KC", static_cast<int64_t>(KC));

    // Simulate packing buffer allocation
    int64_t const pack_bytes = static_cast<int64_t>(MC) * KC * 8;
    ProfileMemAlloc(pack_bytes);

    double volatile sum = 0.0;
    for (int i = 0; i < MC * KC / 10; ++i)
        sum += static_cast<double>(i);

    ProfileMemFree(pack_bytes);
}

// NOLINTNEXTLINE(readability-identifier-naming)
void simulate_pack_B(int KC, int NC) {
    LabeledSection("pack_B");
    ProfileAnnotate("KC", static_cast<int64_t>(KC));
    ProfileAnnotate("NC", static_cast<int64_t>(NC));

    // Simulate packing buffer allocation
    int64_t const pack_bytes = static_cast<int64_t>(KC) * NC * 8;
    ProfileMemAlloc(pack_bytes);

    double volatile sum = 0.0;
    for (int i = 0; i < KC * NC / 10; ++i)
        sum += static_cast<double>(i);

    ProfileMemFree(pack_bytes);
}

void simulate_blis_contraction(int M, int N, int K) {
    LabeledSection("blis_contraction");
    ProfileAnnotate("algorithm", "BLIS");
    ProfileAnnotate("M", static_cast<int64_t>(M));
    ProfileAnnotate("N", static_cast<int64_t>(N));
    ProfileAnnotate("K", static_cast<int64_t>(K));

    int constexpr MC = 256, NC = 1024, KC = 256, MR = 16, NR = 6;
    ProfileAnnotate("MC", static_cast<int64_t>(MC));
    ProfileAnnotate("NC", static_cast<int64_t>(NC));
    ProfileAnnotate("KC", static_cast<int64_t>(KC));
    ProfileAnnotate("MR", static_cast<int64_t>(MR));
    ProfileAnnotate("NR", static_cast<int64_t>(NR));

    int const nc_tiles = (N + NC - 1) / NC;
    int const kc_tiles = (K + KC - 1) / KC;
    int const mc_tiles = (M + MC - 1) / MC;

    for (int jc = 0; jc < nc_tiles; ++jc) {
        for (int pc = 0; pc < kc_tiles; ++pc) {
            simulate_pack_B(std::min(KC, K - pc * KC), std::min(NC, N - jc * NC));
            for (int ic = 0; ic < mc_tiles; ++ic) {
                simulate_pack_A(std::min(MC, M - ic * MC), std::min(KC, K - pc * KC));
                simulate_microkernel(std::min(MC, M - ic * MC), std::min(NC, N - jc * NC), std::min(KC, K - pc * KC));
            }
        }
    }
}

void simulate_sort_tensor(int rank, int64_t elements) {
    LabeledSection("sort_tensor");
    ProfileAnnotate("rank", static_cast<int64_t>(rank));
    ProfileAnnotate("elements", elements);

    // Simulate temporary buffer for sort
    int64_t const sort_bytes = elements * 8;
    ProfileMemAlloc(sort_bytes);

    double volatile sum = 0.0;
    for (int64_t i = 0; i < elements / 100; ++i)
        sum += static_cast<double>(i);

    ProfileMemFree(sort_bytes);
}

void simulate_einsum(int M, int N, int K) {
    LabeledSection("einsum {} {} {}", M, N, K);
    ProfileAnnotate("algorithm", "GEMM");
    ProfileAnnotate("C_rank", static_cast<int64_t>(2));
    ProfileAnnotate("A_rank", static_cast<int64_t>(2));
    ProfileAnnotate("B_rank", static_cast<int64_t>(2));
    ProfileAnnotate("flops", static_cast<int64_t>(2) * M * N * K);
    ProfileAnnotate("bytes_read", static_cast<int64_t>(8) * (M * K + K * N));
    ProfileAnnotate("bytes_written", static_cast<int64_t>(8) * M * N);

    simulate_sort_tensor(2, static_cast<int64_t>(M) * K);
    simulate_blis_contraction(M, N, K);

    // Slow down so the viewer has time to display updates
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void worker_loop(int worker_id) {
    LabeledSection("worker_{}", worker_id);

    std::mt19937                       rng(42 + worker_id);
    std::uniform_int_distribution<int> size_dist(32, 256);

    while (g_running.load(std::memory_order_relaxed)) {
        int const M = size_dist(rng);
        int const N = size_dist(rng);
        int const K = size_dist(rng);
        simulate_einsum(M, N, K);
    }
}

int einsums_main() {
    einsums::println("Profile live demo — running for 30 seconds.");
    einsums::println("Connect with:  python devtools/profiling/profile_viewer.py");
    einsums::println("");

    int constexpr num_workers = 3;
    int constexpr run_seconds = 30;

    std::vector<std::thread> workers;
    workers.reserve(num_workers);

    {
        LabeledSection("main");

        // Launch worker threads
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back(worker_loop, i);
        }

        // Main thread also does work
        // NOLINTNEXTLINE(bugprone-random-generator-seed)
        std::mt19937                       rng(0);
        std::uniform_int_distribution<int> size_dist(64, 512);

        auto start = std::chrono::steady_clock::now();
        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= run_seconds)
                break;

            int const M = size_dist(rng);
            int const N = size_dist(rng);
            int const K = size_dist(rng);
            simulate_einsum(M, N, K);
        }

        // Signal workers to stop and wait
        g_running.store(false, std::memory_order_relaxed);
        for (auto &t : workers)
            t.join();
    }

    einsums::println("\nDone. Printing final report:\n");
    Profiler::instance().print(true);

    return 0;
}
} // namespace

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
