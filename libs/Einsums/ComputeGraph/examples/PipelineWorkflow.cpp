//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PipelineWorkflow.cpp
/// @brief Demonstrates Workspace + Pipeline with setup, iterative loop, and post-processing.
///
/// This example simulates a simplified iterative workflow similar to an SCF calculation:
///   Stage 1 (Setup):      Initialize matrices via similarity transform
///   Stage 2 (Loop):       Iteratively refine F until convergence
///   Stage 3 (Post):       Final scaling of the result
///
/// Key features demonstrated:
///   - Workspace for cross-pipeline input tensors (H, X)
///   - Pipeline with add_stage() and add_loop()
///   - Pipeline-owned tensors via declare_*_tensor()
///   - LoopCondition with convergence-based early exit
///   - Applying optimization passes to all stages
///   - Profiled execution with hierarchy visible in the profile viewer

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    constexpr size_t N = 5;

    // ── Build the workspace + pipeline ────────────────────────────────────
    //
    // Workspace: owns cross-pipeline tensors (inputs shared across computations)
    // Pipeline:  owns working tensors that are stage-local
    //
    // This hierarchy is visible in the profile viewer as:
    //   Workspace "iterative_solver" → Pipeline "iterative_solver" → Stages

    cg::Workspace workspace("My Workspace");

    // Input tensors — owned by the Workspace (shared across pipelines)
    auto &H = workspace.declare_random_tensor<double, 2>("H", N, N); // "Hamiltonian"
    auto &X = workspace.declare_random_tensor<double, 2>("X", N, N); // "Transformation"

    cg::Pipeline pipeline("iterative_solver");
    pipeline.set_workspace(workspace);

    // Working tensors — owned by the Pipeline (stage-local intermediates)
    auto &F       = pipeline.declare_zero_tensor<double, 2>("F", N, N);     // "Fock matrix"
    auto &F_old   = pipeline.declare_zero_tensor<double, 2>("F_old", N, N); // Previous iteration
    auto &tmp     = pipeline.declare_zero_tensor<double, 2>("tmp", N, N);
    auto &tmp2    = pipeline.declare_zero_tensor<double, 2>("tmp2", N, N);
    auto &contrib = pipeline.declare_zero_tensor<double, 2>("contrib", N, N);

    size_t iteration_count       = 0;
    double convergence_threshold = 1e-6;

    // ── Stage 1: Setup ──────────────────────────────────────────────────────
    {
        auto            &setup = pipeline.add_stage("setup");
        cg::CaptureGuard guard(setup);

        // F = X^T * H * X  (two einsums for the similarity transform)
        cg::einsum("ji;jk->ik", 0.0, &tmp, 1.0, X, H); // tmp = X^T * H
        cg::einsum("ik;kj->ij", 0.0, &F, 1.0, tmp, X); // F = tmp * X
    }

    // ── Stage 2: Iterative refinement loop ──────────────────────────────────
    {
        auto            &loop_body = pipeline.add_loop("refinement",
                                                       /*max_iterations=*/50,
                                                       /*condition=*/[&](size_t iter) -> bool {
                                                iteration_count = iter + 1;
                                                double diff     = 0.0;
                                                for (size_t ii = 0; ii < N; ii++)
                                                    for (size_t jj = 0; jj < N; jj++) {
                                                        double d = F(ii, jj) - F_old(ii, jj);
                                                        diff += d * d;
                                                    }
                                                diff = std::sqrt(diff);
                                                println("  Iteration {}: ||delta F|| = {:.2e}", iter + 1, diff);
                                                return diff >= convergence_threshold;
                                            });
        cg::CaptureGuard guard(loop_body);

        // Save current F for convergence check
        cg::permute("ij <- ij", 0.0, &F_old, 1.0, F);

        // Damped update: F = 0.9 * F + 0.1 * X^T * H * X
        cg::scale(0.9, &F);
        cg::einsum("ji;jk->ik", 0.0, &tmp2, 1.0, X, H);
        cg::einsum("ik;kj->ij", 0.0, &contrib, 1.0, tmp2, X);
        cg::axpy(0.1, contrib, &F);
    }

    // ── Stage 3: Post-processing ────────────────────────────────────────────
    {
        auto            &post = pipeline.add_stage("postprocess");
        cg::CaptureGuard guard(post);

        // Final scaling
        cg::scale(1.0 / static_cast<double>(N), &F);
    }

    // ── Apply optimization passes ───────────────────────────────────────────
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);

    // Materialize workspace tensors (not covered by pipeline passes)
    workspace.materialize_all();

    // ── Print graph structure (after optimization, before execute) ───────────
    std::cout << "\n=== Graph Structure After Optimization ===\n\n";

    for (size_t si = 0; si < pipeline.num_stages(); si++) {
        auto *g = pipeline.stage_graph(si);
        if (!g)
            continue;
        std::cout << "--- Stage " << si << ": " << pipeline.stage_name(si) << " (" << g->num_nodes() << " nodes, " << g->num_tensors()
                  << " tensors) ---\n";
        g->print_summary(std::cout);
        std::cout << "\n";
    }

    std::cout << "=== GraphViz DOT ===\n\n";
    for (size_t si = 0; si < pipeline.num_stages(); si++) {
        auto *g = pipeline.stage_graph(si);
        if (!g)
            continue;
        std::cout << "// Stage " << si << ": " << pipeline.stage_name(si) << "\n";
        g->print_dot(std::cout);
        std::cout << "\n";
    }

    // ── Execute the pipeline ────────────────────────────────────────────────
    println("\n=== Running Pipeline ===");
    pipeline.execute();

    // ── Print timing ────────────────────────────────────────────────────────
    std::cout << "\n=== Timing Reports ===\n";
    for (size_t si = 0; si < pipeline.num_stages(); si++) {
        auto *g = pipeline.stage_graph(si);
        if (!g)
            continue;
        std::cout << "\n" << pipeline.stage_name(si) << ":\n";
        g->print_timing_report(std::cout);
    }

    println("\nPipeline completed after {} iterations", iteration_count);
    println("Final F (scaled):");
    println(F);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
