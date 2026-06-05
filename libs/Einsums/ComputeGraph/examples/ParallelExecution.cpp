//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ParallelExecution.cpp
/// @brief Demonstrates the Executor abstraction and parallel graph execution.
///
/// Shows:
///   - SequentialExecutor (default behavior)
///   - OpenMPExecutor (wavefront parallelism for independent nodes)
///   - How the runtime automatically detects independent nodes
///   - Using executors with Pipeline

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    constexpr size_t N = 50;

    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);
    auto D = create_random_tensor<double>("D", N, N);
    auto E = create_random_tensor<double>("E", N, N);

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Graph with independent operations — parallelism detected automatically
    // ═══════════════════════════════════════════════════════════════════════
    println("=== Independent Operations ===\n");
    {
        cg::Graph graph("parallel_demo");
        auto     &C = graph.create_zero_tensor<double, 2>("C", N, N);
        auto     &F = graph.create_zero_tensor<double, 2>("F", N, N);
        auto     &G = graph.create_zero_tensor<double, 2>("G", N, N);

        {
            cg::CaptureGuard guard(graph);
            // These three einsums are INDEPENDENT — they read different inputs
            // and write to different outputs. The OpenMP executor will run them
            // in parallel automatically.
            cg::einsum("ik;kj->ij", &C, A, B);
            cg::einsum("ik;kj->ij", &F, D, E);
            cg::einsum("ik;kj->ij", &G, A, E);
        }

        println("Graph has {} nodes", graph.num_nodes());

        // Check dependency structure
        auto const &deps         = graph.dependencies();
        size_t      level0_count = 0;
        for (size_t nd = 0; nd < graph.num_nodes(); nd++) {
            if (deps.predecessors[nd].empty())
                level0_count++;
        }
        println("Independent nodes (level 0): {} — these run in parallel", level0_count);

        // Execute with OpenMP
        cg::OpenMPExecutor omp;
        graph.execute(omp);

        println("C[0,0] = {:.4f}", C(0, 0));
        println("F[0,0] = {:.4f}", F(0, 0));
        println("G[0,0] = {:.4f}", G(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Dependent chain — still correct with parallel executor
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Dependent Chain ===\n");
    {
        cg::Graph graph("chain_demo");
        auto     &C = graph.create_zero_tensor<double, 2>("C", N, N);
        auto     &F = graph.create_zero_tensor<double, 2>("F", N, N);

        {
            cg::CaptureGuard guard(graph);
            // C depends on A, B
            cg::einsum("ik;kj->ij", &C, A, B);
            // F depends on C — must run AFTER C is computed
            cg::einsum("ik;kj->ij", &F, C, D);
        }

        // OpenMP executor respects dependencies
        cg::OpenMPExecutor omp;
        graph.execute(omp);

        println("Chain result F[0,0] = {:.4f}", F(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Pipeline with executor
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Pipeline with OpenMP Executor ===\n");
    {
        auto acc = create_zero_tensor<double>("acc", N, N);
        auto C   = create_zero_tensor<double>("C", N, N);

        cg::Pipeline pipeline("parallel_pipeline");

        {
            auto            &setup = pipeline.add_stage("compute");
            cg::CaptureGuard guard(setup);
            cg::einsum("ik;kj->ij", &C, A, B);
        }

        size_t count = 0;
        {
            auto            &loop = pipeline.add_loop("accumulate", 5, [&](size_t iter) {
                count = iter + 1;
                return iter < 4;
            });
            cg::CaptureGuard guard(loop);
            cg::axpy(1.0, C, &acc);
        }

        // Each stage's graph uses the OpenMP executor
        cg::OpenMPExecutor omp;
        pipeline.execute(omp);

        println("Pipeline with {} iterations, acc[0,0] = {:.4f}", count, acc(0, 0));
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
