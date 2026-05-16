//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Graph capture and execute - simple einsum", "[ComputeGraph]") {
    auto A          = create_random_tensor<double>("A", 10, 5);
    auto B          = create_random_tensor<double>("B", 5, 8);
    auto C          = create_zero_tensor<double>("C", 10, 8);
    auto C_expected = create_zero_tensor<double>("C_expected", 10, 8);

    // Eager execution for reference
    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, k}, A, Indices{k, j}, B);

    // Graph capture and execute
    cg::Graph graph("test_simple");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.num_tensors() == 3);

    graph.execute();

    // Verify results match
    for (size_t ii = 0; ii < 10; ii++) {
        for (size_t jj = 0; jj < 8; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph capture - einsum chain", "[ComputeGraph]") {
    auto A   = create_random_tensor<double>("A", 10, 5);
    auto B   = create_random_tensor<double>("B", 5, 8);
    auto D   = create_random_tensor<double>("D", 8, 3);
    auto T1  = create_zero_tensor<double>("T1", 10, 8);
    auto E   = create_zero_tensor<double>("E", 10, 3);
    auto T1e = create_zero_tensor<double>("T1e", 10, 8);
    auto Ee  = create_zero_tensor<double>("Ee", 10, 3);

    // Eager reference
    tensor_algebra::einsum(Indices{i, j}, &T1e, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, l}, &Ee, Indices{i, j}, T1e, Indices{j, l}, D);

    // Graph capture
    cg::Graph graph("test_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T1, A, B);
        cg::einsum("ij;jl->il", &E, T1, D);
    }

    REQUIRE(graph.num_nodes() == 2);

    graph.execute();

    for (size_t ii = 0; ii < 10; ii++) {
        for (size_t ll = 0; ll < 3; ll++) {
            REQUIRE(std::abs(E(ii, ll) - Ee(ii, ll)) < 1e-12);
        }
    }
}

TEST_CASE("Graph capture - scale operation", "[ComputeGraph]") {
    auto A          = create_random_tensor<double>("A", 5, 5);
    auto A_expected = Tensor<double, 2>(A); // deep copy

    linear_algebra::scale(2.5, &A_expected);

    cg::Graph graph("test_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.5, &A);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(A(ii, jj) - A_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph capture - permute operation", "[ComputeGraph]") {
    auto A          = create_random_tensor<double>("A", 4, 6);
    auto C          = create_zero_tensor<double>("C", 6, 4);
    auto C_expected = create_zero_tensor<double>("C_expected", 6, 4);

    // Eager reference: transpose
    tensor_algebra::permute(0.0, Indices{j, i}, &C_expected, 1.0, Indices{i, j}, A);

    cg::Graph graph("test_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &C, 1.0, A);
    }

    graph.execute();

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph print_dot and print_summary", "[ComputeGraph]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("test_print");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    std::ostringstream dot_out, summary_out;
    graph.print_dot(dot_out);
    graph.print_summary(summary_out);

    REQUIRE(dot_out.str().find("digraph") != std::string::npos);
    REQUIRE(summary_out.str().find("test_print") != std::string::npos);
    REQUIRE(summary_out.str().find("Einsum") != std::string::npos);
}

TEST_CASE("Graph replay executes multiple times", "[ComputeGraph]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    // Each replay accumulates: C += A * B (c_prefactor=1.0)
    cg::Graph graph("test_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
    }

    // Execute 3 times via manual loop
    graph.execute();
    graph.execute();
    graph.execute();

    // Compute reference: 3 * A * B
    auto C_expected = create_zero_tensor<double>("Ce", 3, 3);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_expected, 3.0, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}
