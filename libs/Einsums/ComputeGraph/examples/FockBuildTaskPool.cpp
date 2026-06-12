//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file FockBuildTaskPool.cpp
/// @brief Simulates a Fock matrix build using TaskPool for parallel integral batches
///        inside a ComputeGraph-managed SCF iteration.
///
/// This demonstrates the two-level parallelism model:
///   - ComputeGraph: orders the high-level SCF steps (build F, diagonalize, update D)
///   - TaskPool: parallelizes the integral contribution loop within the Fock build
///
/// This is the primary use case for combining ComputeGraph and TaskPool in
/// quantum chemistry applications.

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

    constexpr size_t nbf     = 20;                  // Basis functions
    constexpr size_t n_pairs = nbf * (nbf + 1) / 2; // Unique shell pairs

    println("=== Fock Build with TaskPool ({} basis functions, {} shell pairs) ===\n", nbf, n_pairs);

    // ── Setup: create "integral" data and density matrix ─────────────────────
    // In a real code, these would come from an integral engine.
    // Here we simulate with random data.
    auto H_core = create_random_tensor<double>("H_core", nbf, nbf);
    // Make H symmetric
    for (size_t ii = 0; ii < nbf; ii++)
        for (size_t jj = ii + 1; jj < nbf; jj++)
            H_core(jj, ii) = H_core(ii, jj);

    auto D_mat = create_random_tensor<double>("D", nbf, nbf);
    // Make D symmetric
    for (size_t ii = 0; ii < nbf; ii++)
        for (size_t jj = ii + 1; jj < nbf; jj++)
            D_mat(jj, ii) = D_mat(ii, jj);

    // Simulated two-electron integrals (simplified: stored as nbf x nbf)
    auto eri_J = create_random_tensor<double>("eri_J", nbf, nbf);
    auto eri_K = create_random_tensor<double>("eri_K", nbf, nbf);

    auto F_mat = create_zero_tensor<double>("F", nbf, nbf);

    // ── Method 1: Everything in one graph with cg::parallel_for ────────────
    println("--- Method 1: cg::parallel_for + einsum in one graph ---");

    auto &pool = tp::TaskPool::get_singleton();

    auto J_mat = create_zero_tensor<double>("J", nbf, nbf);
    auto K_mat = create_zero_tensor<double>("K", nbf, nbf);

    // Build the ENTIRE Fock build as a single graph:
    //   Node 1: parallel_for fills J and K from shell pairs (TaskPool)
    //   Node 2: F = H_core                                  (permute)
    //   Node 3: F += 2*J                                    (axpy)
    //   Node 4: F -= K                                      (axpy)
    //
    // The graph's topological sort automatically orders Node 1 before
    // Nodes 3 and 4, because parallel_for declares J_mat and K_mat as
    // outputs, and the axpy nodes declare them as inputs.
    cg::Graph fock_assembly("fock_assembly");
    {
        cg::CaptureGuard guard(fock_assembly);

        // Node 1: parallel_for computes J and K over shell pairs
        cg::parallel_for(
            "J_K_build", 0, n_pairs,
            [&](size_t pair_idx) {
                size_t mu = 0, nu = 0;
                size_t count = 0;
                for (size_t ii = 0; ii < nbf && count <= pair_idx; ii++) {
                    for (size_t jj = 0; jj <= ii && count <= pair_idx; jj++) {
                        if (count == pair_idx) {
                            mu = ii;
                            nu = jj;
                        }
                        count++;
                    }
                }
                double j_val = 0.0;
                for (size_t lam = 0; lam < nbf; lam++) {
                    j_val += D_mat(lam, lam) * eri_J(mu, nu);
                }
                J_mat(mu, nu) = j_val;
                J_mat(nu, mu) = j_val;
                double k_val  = D_mat(mu, nu) * eri_K(mu, nu);
                K_mat(mu, nu) = k_val;
                K_mat(nu, mu) = k_val;
            },
            &J_mat, &K_mat); // Declare outputs for dependency tracking

        // Nodes 2-4: Assemble F = H_core + 2*J - K
        cg::permute("ij <- ij", 0.0, &F_mat, 1.0, H_core);
        cg::axpy(2.0, J_mat, &F_mat);
        cg::axpy(-1.0, K_mat, &F_mat);
    }

    fock_assembly.execute();

    println("  F(0,0) = {:.6f}", F_mat(0, 0));
    println("  F is symmetric: {}", std::abs(F_mat(0, 1) - F_mat(1, 0)) < 1e-12 ? "yes" : "no");
    println("  Graph nodes: {} (parallel_for + permute + 2 axpy)", fock_assembly.num_nodes());

    // ── Method 2: DataflowExecutor for concurrent graph nodes ────────────────
    println("\n--- Method 2: DataflowExecutor for independent operations ---");

    // Create a graph with independent branches that can run concurrently
    auto A_mat = create_random_tensor<double>("A", nbf, nbf);
    auto B_mat = create_random_tensor<double>("B", nbf, nbf);
    auto C_mat = create_zero_tensor<double>("C", nbf, nbf);
    auto D2    = create_zero_tensor<double>("D2", nbf, nbf);
    auto E_mat = create_zero_tensor<double>("E", nbf, nbf);

    cg::Graph parallel_graph("parallel_ops");
    {
        cg::CaptureGuard guard(parallel_graph);
        // Branch 1: C = A * B
        cg::einsum("ik;kj->ij", &C_mat, A_mat, B_mat);
        // Branch 2: D2 = A^T (independent of branch 1)
        cg::permute("ji <- ij", 0.0, &D2, 1.0, A_mat);
        // Merge: E = C + D2 (depends on both branches)
        cg::axpy(1.0, C_mat, &E_mat);
        cg::axpy(1.0, D2, &E_mat);
    }

    // DataflowExecutor: branches 1 and 2 run concurrently via TaskPool
    cg::DataflowExecutor df_exec;
    parallel_graph.execute(df_exec);

    println("  E(0,0) = {:.6f}", E_mat(0, 0));
    println("  Graph nodes: {}", parallel_graph.num_nodes());

    // ── Method 3: Full SCF-like iteration, entire loop in one graph ────────
    println("\n--- Method 3: SCF iteration with cg::parallel_for in graph ---");

    double energy     = 0.0;
    double energy_old = 1e10;
    int    max_iter   = 5;

    // Build a graph that captures ONE iteration of the SCF cycle:
    //   1. parallel_for: compute J and K integrals
    //   2. permute + axpy: assemble F = H + 2J - K
    //   3. parallel_reduce: compute energy
    //   4. scale: update density
    //
    // Then REPLAY the graph for each SCF iteration.
    cg::Graph scf_iter_graph("scf_iteration");
    {
        cg::CaptureGuard guard(scf_iter_graph);

        // Step 1: Compute integrals (parallel via TaskPool, inside the graph)
        cg::parallel_for(
            "fock_integrals", 0, n_pairs,
            [&](size_t pair_idx) {
                size_t mu = 0, nu = 0;
                size_t count = 0;
                for (size_t ii = 0; ii < nbf && count <= pair_idx; ii++) {
                    for (size_t jj = 0; jj <= ii && count <= pair_idx; jj++) {
                        if (count == pair_idx) {
                            mu = ii;
                            nu = jj;
                        }
                        count++;
                    }
                }
                double j_val = 0.0;
                for (size_t lam = 0; lam < nbf; lam++) {
                    j_val += D_mat(lam, lam) * eri_J(mu, nu);
                }
                J_mat(mu, nu) = j_val;
                J_mat(nu, mu) = j_val;
                K_mat(mu, nu) = D_mat(mu, nu) * eri_K(mu, nu);
                K_mat(nu, mu) = K_mat(mu, nu);
            },
            &J_mat, &K_mat); // Outputs: J and K

        // Step 2: Assemble F = H_core + 2*J - K
        cg::permute("ij <- ij", 0.0, &F_mat, 1.0, H_core);
        cg::axpy(2.0, J_mat, &F_mat);
        cg::axpy(-1.0, K_mat, &F_mat);

        // Step 3: Compute energy via parallel_reduce
        cg::parallel_reduce<double>(
            "energy", 0, nbf * nbf, &energy, []() { return 0.0; },
            [&](size_t flat, double &acc) {
                size_t mu = flat / nbf;
                size_t nu = flat % nbf;
                acc += D_mat(mu, nu) * (H_core(mu, nu) + F_mat(mu, nu));
            },
            [](double &g, double const &l) { g += l; }, &D_mat, &F_mat); // Inputs: reads D and F

        // Step 4: Update density
        cg::scale(0.99, &D_mat);
    }

    println("  SCF graph: {} nodes", scf_iter_graph.num_nodes());

    // Replay the graph for each SCF iteration
    for (int iter = 0; iter < max_iter; iter++) {
        J_mat.zero();
        K_mat.zero();
        F_mat.zero();

        scf_iter_graph.execute();

        double delta = std::abs(energy - energy_old);
        println("  Iter {}: energy = {:.8f}, delta = {:.2e}", iter, energy, delta);
        energy_old = energy;
    }

    // Print TaskPool metrics
    println("\n--- TaskPool Metrics ---");
    auto m = pool.snapshot_metrics();
    println("  Tasks submitted: {}", m.total_submitted);
    println("  Tasks completed: {}", m.total_completed);
    println("  Work stealing:   {} steals across {} workers", m.total_steals, m.per_worker_executed.size());

    println("\n=== Fock Build Demo Complete ===");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
