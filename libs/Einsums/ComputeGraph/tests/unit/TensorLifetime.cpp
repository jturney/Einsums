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

TEST_CASE("Graph::create_tensor - basic ownership", "[ComputeGraph][Lifetime]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);

    cg::Graph graph("create_tensor_test");

    // Create an intermediate tensor owned by the graph
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    graph.execute();

    // Verify result
    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph::create_tensor - survives capture block scope", "[ComputeGraph][Lifetime]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);

    cg::Graph graph("scope_test");

    // Graph-owned intermediates
    auto &tmp    = graph.create_zero_tensor<double, 2>("tmp", 5, 5);
    auto &result = graph.create_zero_tensor<double, 2>("result", 5, 5);

    // Capture in a nested block — tmp and result survive because graph owns them
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::scale(2.0, &tmp);
        cg::permute("ij <- ij", 0.0, &result, 1.0, tmp);
    }

    // Execute outside the capture block — no dangling references
    graph.execute();

    // Reference
    auto tmp_ref = create_zero_tensor<double>("tmpref", 5, 5);
    tensor_algebra::einsum(Indices{i, j}, &tmp_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &tmp_ref);

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(result(ii, jj) - tmp_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph::create_tensor - multiple tensors", "[ComputeGraph][Lifetime]") {
    cg::Graph graph("multi_tensor");

    auto &A = graph.create_tensor<double, 2>("A", 3, 3);
    auto &B = graph.create_tensor<double, 1>("B", 3);

    // Fill via raw data access
    for (size_t ii = 0; ii < 3; ii++) {
        B(ii) = static_cast<double>(ii + 1);
        for (size_t jj = 0; jj < 3; jj++) {
            A(ii, jj) = static_cast<double>(ii * 3 + jj + 1);
        }
    }

    // Tensors are usable and valid
    REQUIRE(A.name() == "A");
    REQUIRE(B.name() == "B");
    REQUIRE(A.dim(0) == 3);
    REQUIRE(B.dim(0) == 3);
}

TEST_CASE("Pipeline::create_tensor - owned by pipeline", "[ComputeGraph][Lifetime][Pipeline]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Pipeline pipeline("pipeline_ownership");

    {
        auto                  &stage = pipeline.add_stage("compute");
        cg::CaptureGuard const guard(stage);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    {
        auto                  &stage = pipeline.add_stage("scale");
        cg::CaptureGuard const guard(stage);
        cg::scale(2.0, &C);
    }

    pipeline.execute();

    // C = 2 * A * B
    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph validation - catches destroyed tensor", "[ComputeGraph][Lifetime][Validation]") {
    cg::Graph graph("validation_test");

    {
        // Create a tensor that will be destroyed at end of this block
        auto temp = create_random_tensor<double>("temp_will_die", 3, 3);

        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &temp);
    }
    // temp is destroyed here — the graph has a dangling reference

    // execute() should detect the destroyed tensor and throw instead of segfaulting
    REQUIRE_THROWS_AS(graph.execute(), std::runtime_error);
}

TEST_CASE("Graph validation - passes for valid tensors", "[ComputeGraph][Lifetime][Validation]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("valid_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    // Should not throw — tensor is still alive
    REQUIRE_NOTHROW(graph.execute());
}
