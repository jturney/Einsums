//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file IntegralTransform.cpp
/// @brief Demonstrates a multi-step AO→MO integral transformation using
///        ComputeGraph for step ordering and TaskPool for parallelism.
///
/// The four-index integral transformation:
///   (pq|rs)_MO = C_mu,p C_nu,q (mu,nu|lam,sig) C_lam,r C_sig,s
///
/// is done in four half-transformations:
///   Step 1: (mu,nu|lam,s) = sum_sig (mu,nu|lam,sig) C_sig,s
///   Step 2: (mu,nu|r,s)   = sum_lam (mu,nu|lam,s) C_lam,r
///   Step 3: (mu,q|r,s)    = sum_nu  (mu,nu|r,s) C_nu,q
///   Step 4: (p,q|r,s)     = sum_mu  (mu,q|r,s) C_mu,p
///
/// Each step is a graph node; the DataflowExecutor handles ordering.
/// Within each step, TaskPool parallelizes across index pairs.

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

    constexpr size_t nao = 10; // AO basis size
    constexpr size_t nmo = 8;  // MO basis size (nmo <= nao)

    println("=== AO->MO Integral Transformation ===");
    println("  AO basis: {}, MO basis: {}\n", nao, nmo);

    // MO coefficient matrix (nao x nmo)
    auto C_mo = create_random_tensor<double>("C", nao, nmo);

    // AO integrals: (mu,nu|lam,sig) — simplified as a rank-4 tensor
    auto eri_ao = create_random_tensor<double>("eri_ao", nao, nao, nao, nao);

    // Intermediate buffers for half-transformations
    auto half1  = create_zero_tensor<double>("half1", nao, nao, nao, nmo);  // (mu,nu|lam,s)
    auto half2  = create_zero_tensor<double>("half2", nao, nao, nmo, nmo);  // (mu,nu|r,s)
    auto half3  = create_zero_tensor<double>("half3", nao, nmo, nmo, nmo);  // (mu,q|r,s)
    auto eri_mo = create_zero_tensor<double>("eri_mo", nmo, nmo, nmo, nmo); // (p,q|r,s)

    auto &pool = tp::TaskPool::get_singleton();

    // ── Method 1: Manual steps with TaskPool ─────────────────────────────────
    println("--- Method 1: TaskPool parallel_for per transformation step ---");

    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: half1(mu,nu,lam,s) = sum_sig eri_ao(mu,nu,lam,sig) * C(sig,s)
    pool.parallel_for("transform_step1", 0, nao * nao, [&](size_t pair) {
        size_t mu = pair / nao;
        size_t nu = pair % nao;
        for (size_t lam = 0; lam < nao; lam++) {
            for (size_t s = 0; s < nmo; s++) {
                double sum = 0.0;
                for (size_t sig = 0; sig < nao; sig++) {
                    sum += eri_ao(mu, nu, lam, sig) * C_mo(sig, s);
                }
                half1(mu, nu, lam, s) = sum;
            }
        }
    });

    // Step 2: half2(mu,nu,r,s) = sum_lam half1(mu,nu,lam,s) * C(lam,r)
    pool.parallel_for("transform_step2", 0, nao * nao, [&](size_t pair) {
        size_t mu = pair / nao;
        size_t nu = pair % nao;
        for (size_t r = 0; r < nmo; r++) {
            for (size_t s = 0; s < nmo; s++) {
                double sum = 0.0;
                for (size_t lam = 0; lam < nao; lam++) {
                    sum += half1(mu, nu, lam, s) * C_mo(lam, r);
                }
                half2(mu, nu, r, s) = sum;
            }
        }
    });

    // Step 3: half3(mu,q,r,s) = sum_nu half2(mu,nu,r,s) * C(nu,q)
    pool.parallel_for("transform_step3", 0, nao, [&](size_t mu) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    double sum = 0.0;
                    for (size_t nu = 0; nu < nao; nu++) {
                        sum += half2(mu, nu, r, s) * C_mo(nu, q);
                    }
                    half3(mu, q, r, s) = sum;
                }
            }
        }
    });

    // Step 4: eri_mo(p,q,r,s) = sum_mu half3(mu,q,r,s) * C(mu,p)
    pool.parallel_for("transform_step4", 0, nmo, [&](size_t p) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    double sum = 0.0;
                    for (size_t mu = 0; mu < nao; mu++) {
                        sum += half3(mu, q, r, s) * C_mo(mu, p);
                    }
                    eri_mo(p, q, r, s) = sum;
                }
            }
        }
    });

    auto   t1        = std::chrono::high_resolution_clock::now();
    double manual_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  TaskPool manual: {:.2f} ms", manual_ms);
    println("  (0,0|0,0) = {:.8f}", eri_mo(0, 0, 0, 0));

    // ── Method 2: ComputeGraph einsum for each step ──────────────────────────
    println("\n--- Method 2: ComputeGraph einsum with DataflowExecutor ---");

    auto half1_cg  = create_zero_tensor<double>("half1_cg", nao, nao, nao, nmo);
    auto half2_cg  = create_zero_tensor<double>("half2_cg", nao, nao, nmo, nmo);
    auto half3_cg  = create_zero_tensor<double>("half3_cg", nao, nmo, nmo, nmo);
    auto eri_mo_cg = create_zero_tensor<double>("eri_mo_cg", nmo, nmo, nmo, nmo);

    cg::Graph transform_graph("integral_transform");
    {
        cg::CaptureGuard guard(transform_graph);
        // These are rank-5 contractions — each reduces one AO index to MO
        // Using multi-M/multi-N PackedGemm for the higher-rank contractions
        cg::einsum("ijkl;ls->ijks", &half1_cg, eri_ao, C_mo);
        cg::einsum("ijks;kr->ijrs", &half2_cg, half1_cg, C_mo);
        cg::einsum("ijrs;jq->iqrs", &half3_cg, half2_cg, C_mo);
        cg::einsum("iqrs;ip->pqrs", &eri_mo_cg, half3_cg, C_mo);
    }

    // Apply optimization passes
    auto pm = cg::PassManager::create_default();
    transform_graph.apply(pm);

    t0 = std::chrono::high_resolution_clock::now();
    cg::DataflowExecutor df;
    transform_graph.execute(df);
    t1 = std::chrono::high_resolution_clock::now();

    double cg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    println("  ComputeGraph + DataflowExecutor: {:.2f} ms", cg_ms);
    println("  (0,0|0,0) = {:.8f}", eri_mo_cg(0, 0, 0, 0));

    // Verify both methods agree
    double max_diff = 0.0;
    for (size_t pp = 0; pp < nmo; pp++)
        for (size_t qq = 0; qq < nmo; qq++)
            for (size_t rr = 0; rr < nmo; rr++)
                for (size_t ss = 0; ss < nmo; ss++)
                    max_diff = std::max(max_diff, std::abs(eri_mo(pp, qq, rr, ss) - eri_mo_cg(pp, qq, rr, ss)));

    println("\n  Max difference: {:.2e} ({})", max_diff, max_diff < 1e-10 ? "MATCH" : "MISMATCH");

    // Print graph info
    println("\n--- Graph Summary ---");
    transform_graph.print_summary(std::cout);
    transform_graph.print_timing_report(std::cout);

    println("\n=== Integral Transform Demo Complete ===");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
