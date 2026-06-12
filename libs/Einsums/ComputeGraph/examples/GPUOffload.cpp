//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GPUOffload.cpp
/// @brief Demonstrates automatic GPU offloading via ComputeGraph.
///
/// Shows how to:
///   - Write standard CPU tensor code that automatically runs on GPU
///   - Use the default pass manager for GPU optimization
///   - Inspect GPU placement decisions with GPUDiagnostics
///   - View the graph structure with print_dot() (GPU nodes colored blue)
///
/// On Apple Silicon (MPS): float32 GEMM runs on the M-series GPU cores.
/// On CUDA/HIP: dispatches to cuBLAS/hipBLAS.
/// Without a GPU: runs on CPU with mock backend (same results).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <iostream>
#include <sstream>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    std::cout << "=== ComputeGraph GPU Offloading Example ===\n\n";

    // ── 1. Create float tensors (GPU-eligible on MPS) ───────────────────────
    // Use float, not double, MPS only accelerates float32 GEMM.
    auto A = create_random_tensor<float>("A", 256, 256);
    auto B = create_random_tensor<float>("B", 256, 256);
    auto C = create_zero_tensor<float>("C", 256, 256);
    auto D = create_zero_tensor<float>("D", 256, 256);

    // ── 2. Compute reference on CPU ─────────────────────────────────────────
    auto C_ref = Tensor<float, 2>(C);
    auto D_ref = Tensor<float, 2>(D);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &D_ref, 1.0, Indices{i, k}, C_ref, Indices{k, j}, B);

    // ── 3. Capture into graph ───────────────────────────────────────────────
    cg::Graph graph("gpu_offload_example");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, C, B);
    }
    std::cout << "Captured " << graph.num_nodes() << " operations.\n\n";

    // ── 4. Apply GPU optimization passes ────────────────────────────────────
    // create_default() includes all safe passes: ConstantFolding (only folds
    // graph-owned intermediates, safe for user-owned tensors), ScaleAbsorption,
    // CSE, DeadNodeElimination, plus GPU passes (GPUPlacement, TransferInsertion,
    // TransferElimination, GPUDiagnostics, StreamAssignment) when a GPU backend
    // is available.
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    std::cout << "After optimization: " << graph.num_nodes() << " nodes.\n";

    // Print GPU diagnostics
    auto [_, diag] = graph.apply<cg::passes::GPUDiagnostics>();
    std::ostringstream oss;
    diag.print_report(oss);
    std::cout << oss.str() << "\n";

    // ── 5. Execute ──────────────────────────────────────────────────────────
    graph.execute();

    // ── 6. Verify ───────────────────────────────────────────────────────────
    float max_err = 0.0f;
    for (size_t ii = 0; ii < 256; ii++) {
        for (size_t jj = 0; jj < 256; jj++) {
            max_err = std::max(max_err, std::abs(D(ii, jj) - D_ref(ii, jj)));
        }
    }
    std::cout << "Verification:\n";
    std::cout << "  Max error vs CPU reference: " << max_err << "\n";
    std::cout << "  Status: " << (max_err < 1e-3f ? "PASSED" : "FAILED") << "\n\n";

    // ── 7. Graph visualization ──────────────────────────────────────────────
    // GPU nodes appear in blue, transfer nodes in orange.
    std::cout << "GraphViz DOT output (pipe to 'dot -Tpng > graph.png'):\n\n";
    graph.print_dot(std::cout);

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
