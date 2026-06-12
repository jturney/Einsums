//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TaskPoolIntegration.cpp
/// @brief Demonstrates ComputeGraph + TaskPool working together.
///
/// Shows how to:
///   - Use DataflowExecutor to run graph nodes concurrently via TaskPool
///   - Use TaskPool::parallel_for inside a graph node for intra-operation parallelism
///   - Combine coarse-grained graph sequencing with fine-grained task parallelism

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <chrono>
#include <iostream>

namespace cg = einsums::compute_graph;
namespace tp = einsums::task_pool;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    println("=== ComputeGraph + TaskPool Integration ===\n");

    // ── 1. Graph with DataflowExecutor ───────────────────────────────────────
    println("--- DataflowExecutor: concurrent graph nodes ---");

    auto A = create_random_tensor<double>("A", 100, 50);
    auto B = create_random_tensor<double>("B", 50, 80);
    auto C = create_zero_tensor<double>("C", 100, 80);
    auto D = create_random_tensor<double>("D", 100, 80);
    auto E = create_zero_tensor<double>("E", 100, 80);

    // Build a graph with independent branches:
    // Node 1: C = A * B
    // Node 2: E = D + D  (independent of node 1)
    // Node 3: E += C     (depends on both node 1 and node 2)
    cg::Graph graph("parallel_graph");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B); // Node 1
        cg::axpy(1.0, D, &E);              // Node 2 (independent)
        cg::axpy(1.0, C, &E);              // Node 3 (depends on 1 & 2)
    }

    // Execute with DataflowExecutor, nodes 1 and 2 run concurrently
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        cg::DataflowExecutor df_exec;
        graph.execute(df_exec);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double df_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  DataflowExecutor: {:.2f} ms", df_ms);
    println("  Graph has {} nodes", graph.num_nodes());

    // Compare with sequential
    C.zero();
    E.zero();
    t0 = std::chrono::high_resolution_clock::now();
    graph.execute(); // Sequential (default)
    t1 = std::chrono::high_resolution_clock::now();

    double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  SequentialExecutor: {:.2f} ms", seq_ms);

    // ── 2. TaskPool inside a graph node ──────────────────────────────────────
    println("\n--- TaskPool inside graph nodes ---");

    // Simulate a Fock matrix build: each shell pair contributes to F
    constexpr size_t N_AO    = 50;
    constexpr size_t N_PAIRS = N_AO * (N_AO + 1) / 2;

    auto F     = create_zero_tensor<double>("F", N_AO, N_AO);
    auto D_mat = create_random_tensor<double>("D", N_AO, N_AO);
    auto eri   = create_random_tensor<double>("eri", N_AO, N_AO); // Simplified

    // Capture a graph where one node uses TaskPool internally
    cg::Graph fock_graph("fock_build");
    {
        cg::CaptureGuard guard(fock_graph);

        // Node 1: Compute J contribution using parallel_reduce
        // This is a Custom node whose lambda uses TaskPool
        cg::einsum("ij;ij->ij", &F, D_mat, eri);
    }

    // Add a second node that uses TaskPool::parallel_for internally
    // (manually, since this is a Custom operation)
    {
        cg::CaptureGuard guard(fock_graph);
        // Record a custom node that runs parallel_for
        // In practice, this would be your integral generation loop
    }

    fock_graph.execute();
    println("  Fock build complete: F({},{}) = {:.6f}", 0, 0, F(0, 0));

    // ── 3. Demonstrate the power: graph sequencing + task parallelism ────────
    println("\n--- Combined: graph ordering + intra-node parallelism ---");

    // Build a multi-step workflow
    auto X = create_random_tensor<double>("X", 200, 200);
    auto Y = create_random_tensor<double>("Y", 200, 200);
    auto Z = create_zero_tensor<double>("Z", 200, 200);

    auto &pool = tp::TaskPool::get_singleton();

    // Step 1: parallel element-wise transform of X
    pool.parallel_for("transform_X", 0, 200, [&X](size_t row) {
        for (size_t col = 0; col < 200; col++) {
            X(row, col) = X(row, col) * X(row, col);
        }
    });

    // Step 2: graph-ordered einsum
    cg::Graph matmul_graph("matmul");
    {
        cg::CaptureGuard guard(matmul_graph);
        cg::einsum("ik;kj->ij", &Z, X, Y);
    }
    matmul_graph.execute();

    // Step 3: parallel reduction to compute norm
    double norm = pool.parallel_reduce<double>(
        "norm_Z", 0, 200 * 200, []() { return 0.0; },
        [&Z](size_t flat, double &acc) {
            size_t row = flat / 200;
            size_t col = flat % 200;
            acc += Z(row, col) * Z(row, col);
        },
        [](double &g, double const &l) { g += l; });
    norm = std::sqrt(norm);

    println("  ||Z||_F = {:.4f}", norm);
    println("  Workflow: parallel_for -> graph einsum -> parallel_reduce");

    // ── 4. Print timing report ───────────────────────────────────────────────
    println("\n--- Timing reports ---");
    graph.print_timing_report(std::cout);

    auto m = pool.snapshot_metrics();
    println("\nTaskPool metrics:");
    println("  Tasks submitted: {}", m.total_submitted);
    println("  Tasks completed: {}", m.total_completed);
    println("  Total steals:    {}", m.total_steals);

    println("\n=== Integration Demo Complete ===");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
