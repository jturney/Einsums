//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ContinuationPipeline.cpp
/// @brief Shows how to use TaskPool continuations (.then) to build an async
///        tensor processing pipeline alongside ComputeGraph.
///
/// Demonstrates:
///   - Building an async pipeline: preprocess → compute → postprocess
///   - Using .then() to avoid blocking between stages
///   - Running multiple independent pipelines concurrently
///   - Fan-in with when_all() to synchronize at the end

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <vector>

namespace cg = einsums::compute_graph;
namespace tp = einsums::task_pool;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    auto            &pool = tp::TaskPool::get_singleton();
    constexpr size_t N    = 50;

    println("=== Continuation Pipeline Demo ===\n");

    // ── Pipeline 1: async tensor norm computation ────────────────────────────
    // Step 1: Create and fill tensor (async)
    // Step 2: Compute squared elements (continuation)
    // Step 3: Sum all elements (continuation with reduce)

    println("--- Pipeline: async tensor norm ---");

    auto norm_pipeline = pool.submit("create_tensor",
                                     [N]() {
                                         auto A = create_random_tensor<double>("A", N, N);
                                         // Return the Frobenius norm squared via manual computation
                                         double sum = 0.0;
                                         for (size_t ii = 0; ii < N; ii++)
                                             for (size_t jj = 0; jj < N; jj++)
                                                 sum += A(ii, jj) * A(ii, jj);
                                         return sum;
                                     })
                             .then("sqrt", [](double sum_sq) { return std::sqrt(sum_sq); })
                             .then("report", [](double norm) {
                                 println("  Frobenius norm: {:.6f}", norm);
                                 return norm;
                             });

    // ── Pipeline 2: matrix chain multiplication ──────────────────────────────
    println("--- Pipeline: matrix chain A*B*C ---");

    auto A   = create_random_tensor<double>("A", N, N);
    auto B   = create_random_tensor<double>("B", N, N);
    auto C   = create_random_tensor<double>("C", N, N);
    auto AB  = create_zero_tensor<double>("AB", N, N);
    auto ABC = create_zero_tensor<double>("ABC", N, N);

    // Step 1: AB = A * B (async)
    auto step1 = pool.submit("A*B", [&]() { tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B); });

    // Step 2: ABC = AB * C (continuation, runs after step1 completes)
    auto step2 = step1.then("AB*C", [&]() { tensor_algebra::einsum(Indices{i, j}, &ABC, Indices{i, k}, AB, Indices{k, j}, C); });

    // Step 3: compute trace (continuation)
    auto trace_result = step2.then("trace", [&]() {
        double trace = 0.0;
        for (size_t ii = 0; ii < N; ii++)
            trace += ABC(ii, ii);
        return trace;
    });

    // ── Multiple independent computations with fan-in ────────────────────────
    println("--- Multiple independent tasks with fan-in ---");

    // Launch 4 independent matrix operations concurrently
    std::vector<tp::TaskHandle<double>> independent_results;
    for (int batch = 0; batch < 4; batch++) {
        independent_results.push_back(pool.submit("batch_" + std::to_string(batch), [N, batch]() {
            auto X = create_random_tensor<double>("X_" + std::to_string(batch), N, N);
            auto Y = create_random_tensor<double>("Y_" + std::to_string(batch), N, N);
            auto Z = create_zero_tensor<double>("Z_" + std::to_string(batch), N, N);

            tensor_algebra::einsum(Indices{i, j}, &Z, Indices{i, k}, X, Indices{k, j}, Y);

            // Return trace
            double trace = 0.0;
            for (size_t ii = 0; ii < N; ii++)
                trace += Z(ii, ii);
            return trace;
        }));
    }

    // Fan-in: wait for all 4 batch results
    auto all_traces = tp::when_all(std::move(independent_results));

    // Process combined results
    auto final_result = all_traces.then("combine", [](std::vector<double> const &traces) {
        double sum = 0.0;
        for (size_t idx = 0; idx < traces.size(); idx++) {
            println("  Batch {}: trace = {:.4f}", idx, traces[idx]);
            sum += traces[idx];
        }
        return sum;
    });

    // ── Wait for everything ──────────────────────────────────────────────────
    double norm_val       = norm_pipeline.get();
    double chain_trace    = trace_result.get();
    double combined_trace = final_result.get();

    println("\n--- Results ---");
    println("  Norm pipeline:      {:.6f}", norm_val);
    println("  Chain trace (A*B*C): {:.4f}", chain_trace);
    println("  Combined traces:    {:.4f}", combined_trace);

    // ── Show that ComputeGraph and TaskPool work in the same program ─────────
    println("\n--- ComputeGraph replay while TaskPool is active ---");

    auto D_mat = create_random_tensor<double>("D", N, N);
    auto E_mat = create_random_tensor<double>("E", N, N);
    auto F_mat = create_zero_tensor<double>("F", N, N);

    cg::Graph graph("mixed");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &F_mat, D_mat, E_mat);
        cg::scale(0.5, &F_mat);
    }

    // Execute graph (sequential) while TaskPool workers are still alive
    graph.execute();
    println("  F(0,0) = {:.6f} (graph result)", F_mat(0, 0));

    // Replay with different data
    D_mat.set_all(1.0);
    F_mat.zero();
    graph.execute();
    println("  F(0,0) = {:.6f} (replay with ones)", F_mat(0, 0));

    // ── Metrics ──────────────────────────────────────────────────────────────
    auto m = pool.snapshot_metrics();
    println("\n--- Metrics ---");
    println("  Submitted: {}, Completed: {}, Steals: {}", m.total_submitted, m.total_completed, m.total_steals);

    println("\n=== Continuation Pipeline Complete ===");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
