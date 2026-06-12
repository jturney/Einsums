//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AsyncIO.cpp
/// @brief Demonstrates async I/O overlap with the DataflowExecutor.
///
/// Shows how to:
///   - Use cg::read() and cg::write() for synchronous disk I/O nodes
///   - Use cg::read_async() for I/O-compute overlap
///   - Apply the IOPrefetch pass to move reads earlier in the schedule
///   - Compare sync vs async execution timing
///
/// The key idea: DiskRead nodes created with read_async() have two phases:
///   1. async_start: begins the I/O (returns quickly)
///   2. async_finish: waits for completion (called before consumers)
///
/// Between start and finish, the DataflowExecutor runs independent compute
/// nodes, hiding I/O latency behind useful work.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>

namespace cg = einsums::compute_graph;

namespace {

/// Simulate a slow disk read (e.g., loading a large integral file).
/// In real code this would be an HDF5 read or similar.
void simulate_disk_read(double *dst, size_t n, double fill_value, int delay_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    for (size_t idx = 0; idx < n; idx++)
        dst[idx] = fill_value;
}

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    constexpr size_t N        = 64;
    constexpr int    IO_DELAY = 50; // ms

    einsums::println("=== ComputeGraph Async I/O Example ===\n");

    // 1. Synchronous I/O, no overlap
    // ═══════════════════════════════════════════════════════════════════════
    einsums::println("--- 1. Synchronous I/O (baseline) ---\n");
    {
        auto data   = create_zero_tensor<double>("data", N, N);
        auto A      = create_random_tensor<double>("A", N, N);
        auto B      = create_random_tensor<double>("B", N, N);
        auto C      = create_zero_tensor<double>("C", N, N);
        auto result = create_zero_tensor<double>("result", N, N);

        cg::Graph graph("sync_io");
        {
            cg::CaptureGuard const guard(graph);

            // Synchronous read: blocks until I/O completes
            cg::read("load data", "mock.h5", "/data", &data, [&]() { simulate_disk_read(data.data(), N * N, 1.0, IO_DELAY); });

            // Independent compute (could overlap but won't with sync read)
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);

            // Consumer of data
            cg::axpy(1.0, data, &result);
        }

        auto pm = cg::PassManager::create_default();
        graph.apply(pm);

        auto t0 = std::chrono::steady_clock::now();
        graph.execute(); // Sequential: read blocks, then compute, then consume
        auto t1 = std::chrono::steady_clock::now();

        double const sync_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        einsums::println("  Sync execution: {:.1f} ms", sync_ms);
        einsums::println("  result[0,0] = {:.4f} (should be ~1.0)", result(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Async I/O with DataflowExecutor, I/O overlaps with compute
    // ═══════════════════════════════════════════════════════════════════════
    einsums::println("\n--- 2. Async I/O with DataflowExecutor ---\n");
    {
        auto data   = create_zero_tensor<double>("data", N, N);
        auto A      = create_random_tensor<double>("A", N, N);
        auto B      = create_random_tensor<double>("B", N, N);
        auto C      = create_zero_tensor<double>("C", N, N);
        auto result = create_zero_tensor<double>("result", N, N);

        // std::future for the async I/O operation
        std::future<void> io_future;

        cg::Graph graph("async_io");
        {
            cg::CaptureGuard const guard(graph);

            // Async read: start kicks off background I/O, finish waits for completion
            cg::read_async(
                "load data", "mock.h5", "/data", &data,
                /*start*/
                [&]() {
                    // Launch the slow read in a background thread
                    io_future = std::async(std::launch::async, [&]() { simulate_disk_read(data.data(), N * N, 1.0, IO_DELAY); });
                },
                /*finish*/
                [&]() {
                    // Block until the background read completes
                    io_future.get();
                },
                /*sync fallback (for Sequential/OpenMP executors)*/
                [&]() { simulate_disk_read(data.data(), N * N, 1.0, IO_DELAY); });

            // Independent compute, this OVERLAPS with the async read!
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);

            // Consumer of data, waits for async read to finish
            cg::axpy(1.0, data, &result);
        }

        // IOPrefetch moves the DiskRead to position 0, maximizing overlap window
        auto pm = cg::PassManager::create_default();
        graph.apply(pm);

        // Verify IOPrefetch moved the read
        einsums::println("  After IOPrefetch: first node is '{}'", graph.nodes()[0].label);

        auto t0 = std::chrono::steady_clock::now();

        cg::DataflowExecutor df;
        graph.execute(df); // read/start overlaps with A*B compute

        auto t1 = std::chrono::steady_clock::now();

        double const async_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        einsums::println("  Async execution: {:.1f} ms", async_ms);
        einsums::println("  result[0,0] = {:.4f} (should be ~1.0)", result(0, 0));
        einsums::println("  -> I/O and compute overlapped (async time < sync time)");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Multiple async reads, all overlap with compute
    // ═══════════════════════════════════════════════════════════════════════
    einsums::println("\n--- 3. Multiple Async Reads ---\n");
    {
        auto integrals = create_zero_tensor<double>("ERI", N, N);
        auto density   = create_zero_tensor<double>("D", N, N);
        auto A         = create_random_tensor<double>("A", N, N);
        auto B         = create_random_tensor<double>("B", N, N);
        auto C         = create_zero_tensor<double>("C", N, N);
        auto F         = create_zero_tensor<double>("F", N, N);

        std::future<void> eri_future;
        std::future<void> den_future;

        cg::Graph graph("multi_read");
        {
            cg::CaptureGuard const guard(graph);

            // Two independent async reads
            cg::read_async(
                "load ERI", "integrals.h5", "/eri", &integrals,
                [&]() {
                    eri_future = std::async(std::launch::async, [&]() { simulate_disk_read(integrals.data(), N * N, 2.0, IO_DELAY); });
                },
                [&]() { eri_future.get(); }, [&]() { simulate_disk_read(integrals.data(), N * N, 2.0, IO_DELAY); });

            cg::read_async(
                "load D", "density.h5", "/density", &density,
                [&]() { den_future = std::async(std::launch::async, [&]() { simulate_disk_read(density.data(), N * N, 0.5, IO_DELAY); }); },
                [&]() { den_future.get(); }, [&]() { simulate_disk_read(density.data(), N * N, 0.5, IO_DELAY); });

            // Independent computation
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);

            // Consumer of both reads
            cg::axpy(1.0, integrals, &F);
            cg::axpy(1.0, density, &F);
        }

        auto pm = cg::PassManager::create_default();
        graph.apply(pm);

        auto t0 = std::chrono::steady_clock::now();

        cg::DataflowExecutor df;
        graph.execute(df);

        auto t1 = std::chrono::steady_clock::now();

        double const multi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        einsums::println("  Two async reads + compute: {:.1f} ms", multi_ms);
        einsums::println("  F[0,0] = {:.4f} (should be ~2.5: 2.0 from ERI + 0.5 from D)", F(0, 0));
        einsums::println("  -> Both reads and A*B all overlap");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Async write, checkpoint without blocking
    // ═══════════════════════════════════════════════════════════════════════
    einsums::println("\n--- 4. Async Write (checkpoint) ---\n");
    {
        auto A = create_random_tensor<double>("A", N, N);
        auto B = create_random_tensor<double>("B", N, N);
        auto C = create_zero_tensor<double>("C", N, N);
        auto D = create_zero_tensor<double>("D", N, N);

        std::future<void> write_future;

        cg::Graph graph("async_write");
        {
            cg::CaptureGuard const guard(graph);

            // Compute C
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);

            // Async checkpoint write, starts in background
            cg::write_async(
                "checkpoint C", "checkpoint.h5", "/C", &C,
                [&]() {
                    write_future = std::async(std::launch::async, [&]() {
                        // Simulate slow write
                        std::this_thread::sleep_for(std::chrono::milliseconds(IO_DELAY));
                    });
                },
                [&]() { write_future.get(); }, [&]() { std::this_thread::sleep_for(std::chrono::milliseconds(IO_DELAY)); });

            // Independent computation, overlaps with the write
            cg::einsum("ik;kj->ij", 0.0, &D, 1.0, B, A);
        }

        auto pm = cg::PassManager::create_default();
        graph.apply(pm);

        auto t0 = std::chrono::steady_clock::now();

        cg::DataflowExecutor df;
        graph.execute(df);

        auto t1 = std::chrono::steady_clock::now();

        double const write_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        einsums::println("  Compute + async checkpoint: {:.1f} ms", write_ms);
        einsums::println("  -> Write overlapped with second einsum");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. Graph visualization showing I/O nodes
    // ═══════════════════════════════════════════════════════════════════════
    einsums::println("\n--- 5. Graph Summary ---\n");
    {
        auto data   = create_zero_tensor<double>("data", 4, 4);
        auto A      = create_random_tensor<double>("A", 4, 4);
        auto result = create_zero_tensor<double>("result", 4, 4);

        cg::Graph graph("io_summary");
        {
            cg::CaptureGuard const guard(graph);
            cg::read("load data", "file.h5", "/data", &data, [&]() {});
            cg::scale(2.0, &A);
            cg::axpy(1.0, data, &result);
        }

        einsums::println("Before IOPrefetch:");
        graph.print_summary(std::cout);

        graph.apply<cg::passes::IOPrefetch>();

        einsums::println("\nAfter IOPrefetch (DiskRead moved to beginning):");
        graph.print_summary(std::cout);
    }

    finalize();
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
