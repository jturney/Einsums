//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DeferredAllocation.cpp
/// @brief Demonstrates the Workspace/Pipeline/Graph deferred allocation system.
///
/// Shows how to:
///   - Use Workspace for cross-computation tensors
///   - Use Pipeline::declare_tensor() for cross-stage tensors
///   - Use Graph::declare_tensor() for single-computation intermediates
///   - Let MaterializationPass allocate memory just-in-time
///   - Share tensors across multiple Pipelines via Workspace
///
/// This is the recommended pattern for distributed-ready code.
/// The DistributionPlanningPass (included in create_default()) automatically
/// decides which tensors to partition across MPI ranks.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    constexpr size_t N = 8;

    println("=== Deferred Allocation Example ===\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Workspace — cross-computation tensors
    // ═══════════════════════════════════════════════════════════════════════
    println("--- 1. Workspace Scoping ---\n");
    {
        cg::Workspace ws("calculation");

        // Declare tensors — NO memory allocated yet
        auto &H = ws.declare_tensor<double, 2>(std::string("H"), N, N);
        auto &S = ws.declare_tensor<double, 2>(std::string("S"), N, N);

        println("  H: {}x{}, materialized={}", H.dim(0), H.dim(1), H.is_materialized());
        println("  S: {}x{}, materialized={}", S.dim(0), S.dim(1), S.is_materialized());

        // Materialize manually for this demo (normally done by passes)
        H.materialize();
        S.materialize();
        H.zero();
        S.zero();

        println("  After materialize: H materialized={}, S materialized={}", H.is_materialized(), S.is_materialized());

        // Use in a pipeline
        cg::Pipeline p1("pipeline_1");
        p1.set_workspace(ws);

        auto &C = p1.declare_zero_tensor<double, 2>(std::string("C"), N, N);

        {
            auto            &stage = p1.add_stage("compute");
            cg::CaptureGuard guard(stage);
            cg::axpy(1.0, H, &C);
        }

        // Materialize C manually (demo — normally done by MaterializationPass)
        C.materialize();
        C.zero();

        p1.execute();
        println("  C[0,0] after pipeline_1: {:.4f}", C(0, 0));

        // H and S survive — Workspace owns them
        println("  Workspace tensors still alive: {} declared", ws.size());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Pipeline — cross-stage tensors with create_default()
    // ═══════════════════════════════════════════════════════════════════════
    println("\n--- 2. Pipeline + create_default() ---\n");
    {
        // These are regular (immediately allocated) tensors for inputs
        auto A = create_random_tensor<double>("A", N, N);
        auto B = create_random_tensor<double>("B", N, N);

        cg::Pipeline pipeline("deferred_demo");

        // Declare a pipeline-scoped tensor — deferred allocation
        auto &C = pipeline.declare_tensor<double, 2>(std::string("C"), N, N);

        println("  Before apply: C materialized={}", C.is_materialized());

        // Stage 1: compute C = A * B
        {
            auto            &stage = pipeline.add_stage("compute");
            cg::CaptureGuard guard(stage);
            cg::einsum("ik;kj->ij", &C, A, B);
        }

        // Stage 2: scale C
        {
            auto            &stage = pipeline.add_stage("scale");
            cg::CaptureGuard guard(stage);
            cg::scale(2.0, &C);
        }

        // Apply default passes — MaterializationPass inserts allocation nodes
        auto pm = cg::PassManager::create_default();
        pipeline.apply(pm);

        // Execute — Materialize node runs first, then compute, then scale
        pipeline.execute();

        println("  After execute: C materialized={}", C.is_materialized());
        println("  C[0,0] = {:.4f}", C(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Graph — single-computation intermediates
    // ═══════════════════════════════════════════════════════════════════════
    println("\n--- 3. Graph Intermediates ---\n");
    {
        auto A = create_random_tensor<double>("A", N, N);
        auto B = create_random_tensor<double>("B", N, N);

        cg::Graph graph("intermediate_demo");

        // Declare an intermediate with deferred allocation
        auto &T = graph.declare_tensor<double, 2>(std::string("T"), N, N);

        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &T, A, B);
            cg::scale(3.0, &T);
        }

        // Apply passes + execute
        auto pm = cg::PassManager::create_default();
        graph.apply(pm);
        graph.execute();

        println("  T[0,0] = {:.4f}", T(0, 0));
        println("  T materialized = {}", T.is_materialized());
    }

    println("\nDone.");
    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
