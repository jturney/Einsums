//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GraphRebind.cpp
/// @brief Demonstrates graph rebind — updating tensor bindings and prefactors
///        without re-capturing the graph.
///
/// Shows:
///   - Capturing a graph once, then rebinding tensors to different data
///   - update_prefactors() for changing scalar parameters
///   - Using rebind for iterative algorithms with changing inputs

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Basic rebind: same graph, different data
    // ═══════════════════════════════════════════════════════════════════════
    println("=== Basic Rebind ===\n");
    {
        auto A1 = create_random_tensor<double>("A1", 4, 3);
        auto A2 = create_random_tensor<double>("A2", 4, 3);
        auto B  = create_random_tensor<double>("B", 3, 5);
        auto C  = create_zero_tensor<double>("C", 4, 5);

        // Capture once
        cg::Graph graph("rebind_demo");
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &C, A1, B);
        }

        // Execute with A1
        graph.execute();
        println("With A1: C[0,0] = {:.4f}", C(0, 0));

        // Rebind to A2 — one line, no ID lookup needed!
        graph.rebind(A1, A2);
        C.zero();
        graph.execute();
        println("With A2: C[0,0] = {:.4f} (different!)", C(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Update prefactors dynamically
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Update Prefactors ===\n");
    {
        auto A = create_random_tensor<double>("A", 3, 3);
        auto B = create_random_tensor<double>("B", 3, 3);
        auto C = create_random_tensor<double>("C", 3, 3);

        cg::Graph graph("prefactor_demo");
        {
            cg::CaptureGuard guard(graph);
            // C = 0*C + 1*A*B
            cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        }

        cg::NodeId einsum_id = graph.nodes()[0].id;

        // Execute: C = A*B
        graph.execute();
        println("c_pf=0, ab_pf=1: C[0,0] = {:.4f}", C(0, 0));

        // Change to C = C + 2*A*B (accumulate with doubled product)
        graph.update_prefactors(einsum_id, 1.0, 2.0);
        graph.execute();
        println("c_pf=1, ab_pf=2: C[0,0] = {:.4f} (accumulated)", C(0, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Rebind in a loop: process multiple datasets with one graph
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Rebind in Loop ===\n");
    {
        auto B = create_random_tensor<double>("B", 4, 4);
        auto C = create_zero_tensor<double>("C", 4, 4);

        // Create 3 different input tensors
        auto inputs = std::vector<Tensor<double, 2>>{
            create_random_tensor<double>("input_0", 4, 4),
            create_random_tensor<double>("input_1", 4, 4),
            create_random_tensor<double>("input_2", 4, 4),
        };

        // Capture graph once with the first input
        cg::Graph graph("batch_demo");
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &C, inputs[0], B);
        }

        // Process each input by rebinding — just pass old and new tensor
        for (size_t idx = 1; idx < inputs.size(); idx++) {
            graph.rebind(inputs[idx - 1], inputs[idx]);
            C.zero();
            graph.execute();
            println("Input {}: C[0,0] = {:.4f}", idx, C(0, 0));
        }
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
