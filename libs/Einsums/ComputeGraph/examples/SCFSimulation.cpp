//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SCFSimulation.cpp
/// @brief A realistic example demonstrating a simplified Hartree-Fock SCF calculation
///        using the ComputeGraph Pipeline.
///
/// This example shows how the computation graph naturally maps to quantum chemistry
/// workflows:
///   Stage 1 (Setup):   Build orthogonalization matrix X = S^{-1/2}
///   Stage 2 (SCF):     Iteratively build and diagonalize the Fock matrix
///   Stage 3 (Post):    Compute final energy and properties
///
/// The graph captures the full set of linear algebra operations:
///   einsum, syev, scale, permute, gemm, axpy, element_transform
///
/// New features demonstrated:
///   - PassManager::create_default() for full optimization (including GPU passes)
///   - cg::custom() for user-defined operations (Fock build)
///   - Profiler session auto-save via --einsums:profile:save
///   - Runtime destructor handles finalize() automatically
///
/// Run with profiler save:
///   ./CG_SCFSimulation --einsums:profile:save=scf_profile.json

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <iostream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    constexpr size_t nbf = 8; // Number of basis functions

    // ── "One-electron" matrices ─────────────────────────────────────────────
    auto S = create_random_definite<double>("S", nbf, nbf);
    auto H = create_random_tensor<double>("H", nbf, nbf);
    for (size_t ii = 0; ii < nbf; ii++)
        for (size_t jj = ii + 1; jj < nbf; jj++)
            H(jj, ii) = H(ii, jj);

    // ── Working tensors ─────────────────────────────────────────────────────
    auto X     = create_zero_tensor<double>("X", nbf, nbf);
    auto F     = create_zero_tensor<double>("F", nbf, nbf);
    auto C     = create_zero_tensor<double>("C", nbf, nbf);
    auto D     = create_zero_tensor<double>("D", nbf, nbf);
    auto D_old = create_zero_tensor<double>("D_old", nbf, nbf);

    double energy     = 0.0;
    double energy_old = 0.0;
    size_t scf_iter   = 0;

    cg::Pipeline pipeline("hartree_fock");

    auto F_ort   = create_zero_tensor<double>("F_ort", nbf, nbf);
    auto tmp     = create_zero_tensor<double>("tmp", nbf, nbf);
    auto C_prime = create_zero_tensor<double>("C_prime", nbf, nbf);
    auto epsilon = create_zero_tensor<double>("epsilon", nbf);

    // ═════════════════════════════════════════════════════════════════════════
    // Stage 1: Setup — Build X = S^{-1/2}
    // ═════════════════════════════════════════════════════════════════════════
    {
        auto [U, s] = linear_algebra::syev(S);
        tensor_algebra::element_transform(&s, [](double val) { return 1.0 / std::sqrt(val); });
        for (size_t col = 0; col < nbf; col++)
            linear_algebra::scale_column(col, s(col), &U);
        linear_algebra::gemm<false, true>(1.0, U, U, 0.0, &X);
    }

    {
        auto            &setup = pipeline.add_stage("setup");
        cg::CaptureGuard guard(setup);
        cg::permute("ij <- ij", 0.0, &F, 1.0, H);
        println("Setup complete. X = S^{{-1/2}} computed.");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Stage 2: SCF iterations
    // ═════════════════════════════════════════════════════════════════════════
    {
        auto            &scf_body = pipeline.add_loop("scf_iterations", 100, [&](size_t iter) -> bool {
            scf_iter = iter + 1;
            energy   = 0.0;
            for (size_t ii = 0; ii < nbf; ii++)
                for (size_t jj = 0; jj < nbf; jj++)
                    energy += D(ii, jj) * (H(ii, jj) + F(ii, jj));
            energy *= 0.5;

            double delta_e = std::abs(energy - energy_old);
            double rms_d   = 0.0;
            for (size_t ii = 0; ii < nbf; ii++)
                for (size_t jj = 0; jj < nbf; jj++) {
                    double d = D(ii, jj) - D_old(ii, jj);
                    rms_d += d * d;
                }
            rms_d = std::sqrt(rms_d / static_cast<double>(nbf * nbf));

            println("  SCF iter {:3d}: E = {:16.10f}  dE = {:10.3e}  rms(dD) = {:10.3e}", scf_iter, energy, delta_e, rms_d);
            energy_old = energy;

            bool converged = (delta_e < 1e-8 && rms_d < 1e-6 && iter > 0);
            if (converged)
                println("  SCF converged!");
            return !converged;
        });
        cg::CaptureGuard guard(scf_body);

        cg::permute("ij <- ij", 0.0, &D_old, 1.0, D);
        cg::einsum("ji;jk->ik", 0.0, &tmp, 1.0, X, F);
        cg::einsum("ik;kj->ij", 0.0, &F_ort, 1.0, tmp, X);
        cg::syev(&F_ort, &epsilon);
        cg::permute("ij <- ij", 0.0, &C_prime, 1.0, F_ort);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, X, C_prime);
        cg::einsum("ik;jk->ij", 0.0, &D, 1.0, C, C);

        // Fock build: F = 0.6*F + 0.4*H + 0.2*D (damped update)
        cg::scale(0.6, &F);
        cg::axpy(0.4, H, &F);
        cg::axpy(0.2, D, &F);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Stage 3: Post-SCF
    // ═════════════════════════════════════════════════════════════════════════
    {
        auto            &post = pipeline.add_stage("post_scf");
        cg::CaptureGuard guard(post);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, D, H);
    }

    // ── Optimization passes ────────────────────────────────────────────────
    // create_default() is safe for loops and double-precision code.
    // GPU passes are included but won't place double-precision ops on MPS.
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);

    // ── Execute ─────────────────────────────────────────────────────────────
    println("\n--- Starting SCF Calculation ---\n");
    pipeline.execute();

    println("\n--- SCF Complete ---");
    println("Final SCF energy: {:16.10f}", energy);
    println("Total iterations: {}", scf_iter);
    println("\nRun with --einsums:profile:save=scf_profile.json to save profiling data.");

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
