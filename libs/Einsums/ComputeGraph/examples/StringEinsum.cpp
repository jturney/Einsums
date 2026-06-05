//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file StringEinsum.cpp
/// @brief Demonstrates the string-based einsum notation for ComputeGraph.
///
/// Shows:
///   - Arrow notation ("ij <- ik ; kj") and NumPy notation ("ik;kj -> ij")
///   - Single-character and multi-character index names
///   - All supported dispatch patterns: GEMM, GEMV, GER, DOT, direct product
///   - Higher-rank contractions (rank-3, rank-4)
///   - Compile-time validation via EinsumFormatString
///   - String einsum in graph capture and pipeline workflows
///   - Using string einsum with optimization passes

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

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Basic string notation — arrow and NumPy styles
    // ═══════════════════════════════════════════════════════════════════════
    println("=== 1. Basic Notation ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto C = create_zero_tensor<double>("C", 4, 5);
        auto D = create_zero_tensor<double>("D", 4, 5);

        // Arrow notation: output <- input_a ; input_b
        cg::einsum("ij <- ik ; kj", &C, A, B);

        // NumPy notation: input_a ; input_b -> output
        cg::einsum("ik;kj -> ij", &D, A, B);

        // Both produce the same result
        println("Arrow notation C:");
        println(C);
        println("NumPy notation D (should match C):");
        println(D);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Multi-character index names
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 2. Multi-Character Indices ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto C = create_zero_tensor<double>("C", 4, 5);

        // Commas separate multi-character index names
        cg::einsum("mu,nu <- mu,rho ; rho,nu", &C, A, B);
        println("Result with Greek indices (mu,nu <- mu,rho ; rho,nu):");
        println(C);

        // Numbered indices work too
        auto D = create_zero_tensor<double>("D", 4, 5);
        cg::einsum("i1,i2 <- i1,i3 ; i3,i2", &D, A, B);
        println("Result with numbered indices (same computation):");
        println(D);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. All dispatch patterns
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 3. Dispatch Patterns ===\n");
    {
        // GEMM: matrix × matrix → matrix
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto C = create_zero_tensor<double>("C", 4, 5);
        cg::einsum("ij <- ik ; kj", &C, A, B);
        println("GEMM (4x3 * 3x5 -> 4x5): C[0,0] = {:.4f}", C(0, 0));

        // GEMM with transposed A: C[i,j] = A[k,i] * B[k,j]
        auto At = create_random_tensor<double>("At", 3, 4);
        auto Ct = create_zero_tensor<double>("Ct", 4, 5);
        cg::einsum("ij <- ki ; kj", &Ct, At, B);
        println("GEMM transposed A: Ct[0,0] = {:.4f}", Ct(0, 0));

        // GEMV: matrix × vector → vector
        auto x = create_random_tensor<double>("x", 3);
        auto y = create_zero_tensor<double>("y", 4);
        cg::einsum("i <- ik ; k", &y, A, x);
        println("GEMV (4x3 * 3 -> 4): y[0] = {:.4f}", y(0));

        // GER: vector × vector → matrix (outer product)
        auto u = create_random_tensor<double>("u", 4);
        auto v = create_random_tensor<double>("v", 5);
        auto G = create_zero_tensor<double>("G", 4, 5);
        cg::einsum("ij <- i ; j", &G, u, v);
        println("GER (4 ⊗ 5 -> 4x5): G[0,0] = {:.4f}", G(0, 0));

        // DOT: vector · vector → scalar
        auto a = create_random_tensor<double>("a", 10);
        auto b = create_random_tensor<double>("b", 10);
        auto d = create_zero_tensor<double>("d", 1);
        cg::einsum(" <- i ; i", &d, a, b);
        println("DOT (10 · 10 -> scalar): d = {:.4f}", d(0));

        // Direct product: element-wise multiplication
        auto P = create_random_tensor<double>("P", 3, 4);
        auto Q = create_random_tensor<double>("Q", 3, 4);
        auto R = create_zero_tensor<double>("R", 3, 4);
        cg::einsum("ij <- ij ; ij", &R, P, Q);
        println("Direct product (3x4 ⊙ 3x4): R[0,0] = {:.4f}", R(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. Higher-rank contractions
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 4. Higher-Rank Contractions ===\n");
    {
        // Rank-3 × rank-3 → rank-2: contract over j,k
        auto A = create_random_tensor<double>("A", 3, 4, 5);
        auto B = create_random_tensor<double>("B", 4, 5, 2);
        auto C = create_zero_tensor<double>("C", 3, 2);
        cg::einsum("il <- ijk ; jkl", &C, A, B);
        println("Rank-3 contraction (3x4x5 * 4x5x2 -> 3x2): C[0,0] = {:.4f}", C(0, 0));

        // Rank-3 × rank-2 → rank-2: contract over k, keep j as batch
        auto D = create_random_tensor<double>("D", 5, 4);
        auto E = create_zero_tensor<double>("E", 3, 4);
        cg::einsum("ij <- ijk ; kj", &E, A, D);
        println("Rank-3 × rank-2 (3x4x5 * 5x4 -> 3x4): E[0,0] = {:.4f}", E(0, 0));

        // Rank-4 contraction
        auto F = create_random_tensor<double>("F", 2, 3, 4);
        auto G = create_random_tensor<double>("G", 2, 3, 4);
        auto H = create_zero_tensor<double>("H", 2, 3, 2, 3);
        cg::einsum("ijkl <- ijp ; klp", &H, F, G);
        println("Rank-4 contraction (2x3x4 * 2x3x4 -> 2x3x2x3): H[0,0,0,0] = {:.4f}", H(0, 0, 0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. With prefactors
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 5. Prefactors ===\n");
    {
        auto A = create_random_tensor<double>("A", 3, 3);
        auto B = create_random_tensor<double>("B", 3, 3);
        auto C = create_random_tensor<double>("C", 3, 3);

        // C = 2.0 * C + 3.0 * A * B
        cg::einsum("ij <- ik ; kj", 2.0, &C, 3.0, A, B);
        println("With prefactors (C = 2*C + 3*A*B):");
        println(C);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. Graph capture with string einsum
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 6. Graph Capture ===\n");
    {
        auto A = create_random_tensor<double>("A", 5, 3);
        auto B = create_random_tensor<double>("B", 3, 4);

        cg::Graph graph("string_graph");
        auto     &T = graph.create_zero_tensor<double, 2>("T", 5, 4);
        auto     &C = graph.create_zero_tensor<double, 2>("C", 5, 4);

        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ij <- ik ; kj", &T, A, B);
            cg::scale(2.0, &T);
            cg::einsum("ij <- ij ; ij", &C, T, T); // C = T ⊙ T (element-wise square)
        }

        println("Graph summary:");
        graph.print_summary(std::cout);

        // Apply ScaleAbsorption — the scale(2.0) + einsum(ij<-ij;ij) pattern
        // won't fuse because they write to different tensors. But the first
        // scale+einsum pair won't fuse either because the einsum is the first op.
        graph.apply<cg::passes::ScaleAbsorption>();

        graph.execute();

        println("\nResult C = (2*A*B) ⊙ (2*A*B):");
        println(C);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 7. Pipeline with string einsum
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 7. Pipeline ===\n");
    {
        auto A   = create_random_tensor<double>("A", 4, 4);
        auto B   = create_random_tensor<double>("B", 4, 4);
        auto acc = create_zero_tensor<double>("acc", 4, 4);

        auto C = create_zero_tensor<double>("C", 4, 4);

        cg::Pipeline pipeline("string_pipeline");

        // Stage 1: C = A * B
        {
            auto            &setup = pipeline.add_stage("compute");
            cg::CaptureGuard guard(setup);
            cg::einsum("ij <- ik ; kj", &C, A, B);
        }

        // Stage 2: Accumulate C into acc three times
        size_t count = 0;
        {
            auto            &loop_body = pipeline.add_loop("accumulate", 3, [&](size_t iter) {
                count = iter + 1;
                return iter < 2;
            });
            cg::CaptureGuard guard(loop_body);
            cg::axpy(1.0, C, &acc);
        }

        // Stage 3: Scale result
        {
            auto            &post = pipeline.add_stage("normalize");
            cg::CaptureGuard guard(post);
            cg::scale(1.0 / 3.0, &acc);
        }

        pipeline.execute();

        println("Pipeline result (average of 3 accumulations of A*B):");
        println(acc);
        println("Iterations: {}", count);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 8. Mixed: string einsum + template-based einsum in same graph
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== 8. Mixed String + Template Einsum ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto D = create_random_tensor<double>("D", 5, 2);

        cg::Graph graph("mixed_graph");
        auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 5);
        auto     &E = graph.create_zero_tensor<double, 2>("E", 4, 2);

        {
            cg::CaptureGuard guard(graph);

            // String-based einsum
            cg::einsum("ij <- ik ; kj", &T, A, B);

            // Template-based einsum (can mix freely in the same graph)
            cg::einsum("ik;kj->ij", &E, T, D);
        }

        graph.execute();

        println("Mixed graph (string + template einsum):");
        println(E);
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
