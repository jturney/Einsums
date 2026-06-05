//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ControlFlow.cpp
/// @brief Demonstrates conditional and loop nodes within a flat Graph.
///
/// Unlike Pipeline (which has stages), these are nodes IN the graph that
/// coexist with regular operations and respect data dependencies.
///
/// Shows:
///   - Conditional nodes (if-then-else with predicate)
///   - Loop nodes (while loop with convergence check)
///   - Predicates that inspect tensor values
///   - Mixing control flow with regular operations

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

namespace cg = einsums::compute_graph;

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    // ═══════════════════════════════════════════════════════════════════════
    // 1. Conditional node — if-then-else based on tensor value
    // ═══════════════════════════════════════════════════════════════════════
    println("=== Conditional Node ===\n");
    {
        auto value = Tensor<double, 1>("value", 1);
        value(0)   = 10.0;

        cg::Graph graph("conditional_demo");

        // If value > 5: halve it. Else: double it.
        graph.add_conditional(
            "clamp", [&]() { return value(0) > 5.0; }, [&]() { cg::scale(0.5, &value); }, // then
            [&]() { cg::scale(2.0, &value); }                                             // else
        );

        // Execute repeatedly — value oscillates between halving and doubling
        for (int rep = 0; rep < 6; rep++) {
            graph.execute();
            println("  After execute {}: value = {:.4f}", rep + 1, value(0));
        }
        // 10 → 5 → 10 → 5 → 10 → 5
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. Loop node — convergence-based iteration
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Loop Node ===\n");
    {
        auto value = Tensor<double, 1>("value", 1);
        value(0)   = 1000.0;

        size_t iterations = 0;

        cg::Graph graph("loop_demo");

        // Loop: halve until < 1.0
        graph.add_loop(
            "converge", 100,
            [&](size_t iter) {
                iterations = iter + 1;
                println("  Iteration {}: value = {:.4f}", iter + 1, value(0));
                return value(0) >= 1.0;
            },
            [&]() { cg::scale(0.5, &value); });

        graph.execute();
        println("Converged after {} iterations, value = {:.6f}", iterations, value(0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. Mixed: regular ops + loop + regular ops in one graph
    // ═══════════════════════════════════════════════════════════════════════
    println("\n=== Mixed Operations + Loop ===\n");
    {
        auto A = create_random_tensor<double>("A", 4, 4);
        auto B = create_random_tensor<double>("B", 4, 4);

        cg::Graph graph("mixed_demo");
        auto     &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

        // Pre-loop: C = A * B
        {
            cg::CaptureGuard guard(graph);
            cg::einsum("ik;kj->ij", &C, A, B);
        }

        // Loop: halve C three times (lambda form)
        graph.add_loop(
            "halve", 3, [](size_t iter) { return iter < 2; }, [&]() { cg::scale(0.5, &C); });

        // Post-loop: scale by 8 (restores original)
        {
            cg::CaptureGuard guard(graph);
            cg::scale(8.0, &C);
        }

        graph.execute();

        // Verify: C should equal A * B (halved 3 times then × 8)
        auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
        tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

        double max_diff = 0;
        for (size_t ii = 0; ii < 4; ii++)
            for (size_t jj = 0; jj < 4; jj++)
                max_diff = std::max(max_diff, std::abs(C(ii, jj) - C_ref(ii, jj)));
        println("Max difference from reference: {:.2e} (should be ~0)", max_diff);
    }

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
