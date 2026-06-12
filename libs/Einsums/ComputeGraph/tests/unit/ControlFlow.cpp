//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ─── Conditional node tests ─────────────────────────────────────────────────

TEST_CASE("Conditional node - then branch", "[ComputeGraph][ControlFlow]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cond_then");

    // Setup: C = A
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    // Conditional: if true, scale C by 2; else scale by 10
    bool take_then        = true;
    auto [then_g, else_g] = graph.add_conditional("branch", [&]() { return take_then; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &C);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::scale(10.0, &C);
    }

    graph.execute();

    // Should have taken then branch: C = 2 * A
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - 2.0 * A(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Conditional node - else branch", "[ComputeGraph][ControlFlow]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cond_else");

    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &C, 1.0, A);
    }

    bool take_then        = false;
    auto [then_g, else_g] = graph.add_conditional("branch", [&]() { return take_then; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(2.0, &C);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::scale(10.0, &C);
    }

    graph.execute();

    // Should have taken else branch: C = 10 * A
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - 10.0 * A(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Conditional node - predicate inspects tensor", "[ComputeGraph][ControlFlow]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 5.0;

    cg::Graph graph("cond_tensor");

    // Conditional: if value > 3, halve it; else double it
    auto [then_g, else_g] = graph.add_conditional("check_value", [&]() { return value(0) > 3.0; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(0.5, &value);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::scale(2.0, &value);
    }

    graph.execute();
    REQUIRE(std::abs(value(0) - 2.5) < 1e-12); // 5.0 * 0.5 = 2.5

    // Execute again, now value is 2.5, which is < 3, so else branch
    graph.execute();
    REQUIRE(std::abs(value(0) - 5.0) < 1e-12); // 2.5 * 2.0 = 5.0
}

TEST_CASE("Conditional node - empty else branch", "[ComputeGraph][ControlFlow]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 10.0;

    cg::Graph graph("cond_no_else");

    auto [then_g, else_g] = graph.add_conditional("maybe_scale", [&]() { return value(0) > 5.0; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(0.5, &value);
    }
    // else_g left empty, no-op if predicate is false

    graph.execute();
    REQUIRE(std::abs(value(0) - 5.0) < 1e-12); // 10 * 0.5

    graph.execute();
    REQUIRE(std::abs(value(0) - 5.0) < 1e-12); // 5.0 is not > 5.0 → no-op
}

// ─── Loop node tests ────────────────────────────────────────────────────────

TEST_CASE("Loop node - basic convergence", "[ComputeGraph][ControlFlow]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    size_t iterations = 0;

    cg::Graph graph("loop_basic");

    auto &body = graph.add_loop("halving", 1000, [&](size_t iter) {
        iterations = iter + 1;
        return value(0) >= 1.0; // Continue while >= 1.0
    });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &value);
    }

    graph.execute();

    // 100 * 0.5^7 = 0.78125 < 1.0
    REQUIRE(iterations == 7);
    REQUIRE(std::abs(value(0) - 100.0 * std::pow(0.5, 7)) < 1e-12);
}

TEST_CASE("Loop node - max iterations", "[ComputeGraph][ControlFlow]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    cg::Graph graph("loop_max");

    auto &body = graph.add_loop("never_stop", 10, [](size_t) { return true; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.9, &value);
    }

    graph.execute();

    REQUIRE(std::abs(value(0) - 100.0 * std::pow(0.9, 10)) < 1e-10);
}

TEST_CASE("Loop node - with other nodes before and after", "[ComputeGraph][ControlFlow]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("loop_with_context");

    // Pre-loop: C = A * B
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Loop: halve C 3 times
    auto &body = graph.add_loop("halve", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &C);
    }

    // Post-loop: scale by 8 (should restore original)
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(8.0, &C);
    }

    graph.execute();

    // C = A*B * 0.5^3 * 8 = A*B
    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-10);
        }
    }
}

TEST_CASE("Loop node - with OpenMP executor", "[ComputeGraph][ControlFlow]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    size_t iterations = 0;

    cg::Graph graph("loop_omp");

    auto &body = graph.add_loop("halving", 1000, [&](size_t iter) {
        iterations = iter + 1;
        return value(0) >= 1.0;
    });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &value);
    }

    cg::OpenMPExecutor omp;
    graph.execute(omp);

    REQUIRE(iterations == 7);
    REQUIRE(std::abs(value(0) - 100.0 * std::pow(0.5, 7)) < 1e-12);
}
