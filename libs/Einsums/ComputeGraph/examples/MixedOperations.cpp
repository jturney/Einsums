//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file MixedOperations.cpp
/// @brief Demonstrates capturing diverse operation types using Workspace and Pipeline.
///
/// Shows graph-aware wrappers for:
///   - einsum (tensor contraction), string-based notation
///   - scale (scalar multiply)
///   - permute (index reordering / transpose), string-based notation
///   - gemm (direct BLAS matrix multiply)
///   - axpy (vector add)
///   - element_transform (element-wise unary operation)
///   - syev (eigendecomposition)
///   - custom (user-defined operations)

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

    constexpr size_t N = 6;

    // ── Workspace and Pipeline ─────────────────────────────────────────────
    cg::Workspace workspace("mixed_ops");

    auto &A = workspace.declare_random_tensor<double, 2>("A", N, N);
    auto &B = workspace.declare_random_tensor<double, 2>("B", N, N);

    cg::Pipeline pipeline("operations");
    pipeline.set_workspace(workspace);

    auto &C  = pipeline.declare_zero_tensor<double, 2>("C", N, N);
    auto &D  = pipeline.declare_zero_tensor<double, 2>("D", N, N);
    auto &At = pipeline.declare_zero_tensor<double, 2>("A_transposed", N, N);

    // ── Stage 1: Multi-operation compute ───────────────────────────────────
    {
        auto            &stage = pipeline.add_stage("compute");
        cg::CaptureGuard guard(stage);

        // 1. C = A * B  (string-based einsum)
        cg::einsum("ij <- ik ; kj", &C, A, B);

        // 2. Scale C by 0.5
        cg::scale(0.5, &C);

        // 3. Transpose A into At  (string-based permute)
        cg::permute("ji <- ij", &At, A);

        // 4. D = At * C  (gemm, direct BLAS call)
        cg::gemm<false, false>(1.0, At, C, 0.0, &D);

        // 5. D += 0.1 * C  (axpy)
        cg::axpy(0.1, C, &D);

        // 6. Square every element of D  (element_transform)
        cg::element_transform(&D, [](double x) { return x * x; });
    }

    // ── Apply passes and execute ───────────────────────────────────────────
    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    workspace.materialize_all();

    println("Mixed operations pipeline:");
    if (auto *g = pipeline.stage_graph(0))
        g->print_summary(std::cout);

    pipeline.execute();
    println("\nResult D (squared elements of At*C*0.5 + 0.1*C*0.5):");
    println(D);

    // ── Eigendecomposition (separate pipeline) ─────────────────────────────
    // S must be positive-definite for syev; create it outside workspace
    auto S = create_random_definite<double>("S", N, N);
    auto W = create_zero_tensor<double>("eigenvalues", N);

    cg::Graph eigen_graph("eigendecomp");
    {
        cg::CaptureGuard guard(eigen_graph);
        cg::syev(&S, &W);
    }
    eigen_graph.execute();

    println("\nEigenvalues of S (computed via pipeline execution):");
    println(W);

    // ── Custom operations (raw Graph, custom ops don't need Pipeline) ────
    println("\n--- Custom Operations ---");
    {
        auto M = create_random_tensor<double>("M", N, N);
        auto R = create_zero_tensor<double>("R", N, N);

        cg::Graph custom_graph("custom_ops");
        {
            cg::CaptureGuard guard(custom_graph);

            // Typed form: declare output tensors, then provide the lambda.
            cg::custom(
                "zero_diagonal",
                [&]() {
                    for (size_t ii = 0; ii < N; ii++)
                        M(ii, ii) = 0.0;
                },
                &M);

            // Full form: declare inputs and outputs explicitly for dependency tracking.
            cg::custom("row_sum", std::make_tuple(std::cref(M)), std::make_tuple(std::ref(R)), [&]() {
                for (size_t ii = 0; ii < N; ii++)
                    for (size_t jj = 0; jj < N; jj++)
                        R(ii, 0) += M(ii, jj);
            });
        }

        custom_graph.execute();
        println("After custom ops: R[0,0] = {:.4f}", R(0, 0));
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
