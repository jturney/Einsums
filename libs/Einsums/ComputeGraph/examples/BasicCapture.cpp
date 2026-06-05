//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BasicCapture.cpp
/// @brief Demonstrates basic graph capture and execution using Workspace and Pipeline.
///
/// Shows how to:
///   - Create a Workspace with input tensors
///   - Build a Pipeline with stages that capture einsum operations
///   - Apply optimization passes
///   - Execute and replay the pipeline
///   - Inspect the graph with print_summary() and print_dot()

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>

#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    // ── 1. Create Workspace with input tensors ─────────────────────────────
    cg::Workspace workspace("basic_example");

    auto &A = workspace.declare_random_tensor<double, 2>("A", 8, 5);
    auto &B = workspace.declare_random_tensor<double, 2>("B", 5, 6);

    // ── 2. Build a Pipeline with a single compute stage ────────────────────
    cg::Pipeline pipeline("compute");
    pipeline.set_workspace(workspace);

    // Pipeline-owned output tensor
    auto &C = pipeline.declare_zero_tensor<double, 2>("C", 8, 6);

    {
        auto                  &stage = pipeline.add_stage("multiply");
        cg::CaptureGuard const guard(stage);

        // C = A * B  (string-based einsum notation)
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    // ── 3. Apply optimization passes ───────────────────────────────────────
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    workspace.materialize_all();

    // ── 4. Inspect the graph ───────────────────────────────────────────────
    println("Graph captured:");
    if (auto *g = pipeline.stage_graph(0))
        g->print_summary(std::cout);
    std::cout << "\n";

    // ── 5. Execute the pipeline ────────────────────────────────────────────
    pipeline.execute();
    println("After execute:");
    println(C);

    // ── 6. Replay: execute multiple times ──────────────────────────────────
    // Each execution re-runs the captured operations.
    // With c_pf=0 (default), each execution overwrites C.
    pipeline.execute();
    pipeline.execute();
    println("After 2 more executions (same result — overwrites each time):");
    println(C);

    // ── 7. Accumulation example ────────────────────────────────────────────
    // To accumulate across executions, use c_pf=1:
    cg::Pipeline accum_pipeline("accumulate");
    accum_pipeline.set_workspace(workspace);

    auto &D = accum_pipeline.declare_zero_tensor<double, 2>("D", 8, 6);
    {
        auto                  &stage = accum_pipeline.add_stage("accum");
        cg::CaptureGuard const guard(stage);
        // D += A * B each time (c_prefactor=1.0 preserves old D)
        cg::einsum("ij <- ik ; kj", 1.0, &D, 1.0, A, B);
    }
    accum_pipeline.apply(pm);

    accum_pipeline.execute();
    accum_pipeline.execute();
    accum_pipeline.execute();
    println("D after 3 accumulations (3 * A * B):");
    println(D);

    // ── 8. GraphViz DOT output ─────────────────────────────────────────────
    println("\nGraphViz DOT output:");
    if (auto *g = pipeline.stage_graph(0))
        g->print_dot(std::cout);

    // ── 9. Timing report ───────────────────────────────────────────────────
    println("\nTiming report:");
    if (auto *g = pipeline.stage_graph(0))
        g->print_timing_report(std::cout);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
