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

TEST_CASE("Rebind - basic tensor rebind", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    // Capture graph with A1
    cg::Graph graph("rebind_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    // Execute with A1
    graph.execute();

    auto C_ref1 = create_zero_tensor<double>("Cr1", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref1, Indices{i, k}, A1, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref1(ii, jj)) < 1e-12);
        }
    }

    // Rebind A1 → A2 (one line!)
    graph.rebind(A1, A2);

    // Execute again — should now use A2
    C.zero();
    graph.execute();

    auto C_ref2 = create_zero_tensor<double>("Cr2", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref2, Indices{i, k}, A2, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref2(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - dimension mismatch throws", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 5, 3); // Different first dim!
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_mismatch");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }

    REQUIRE_THROWS_AS(graph.rebind(A1, A2), std::invalid_argument);
}

TEST_CASE("Rebind - scale operation", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 3, 3);
    auto A2 = create_random_tensor<double>("A2", 3, 3);

    cg::Graph graph("rebind_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A1);
    }

    graph.execute();

    auto A2_copy = Tensor<double, 2>(A2);
    graph.rebind(A1, A2);
    graph.execute();

    // A2 should now be doubled
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A2(ii, jj) - 2.0 * A2_copy(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Rebind - string einsum", "[ComputeGraph][Rebind]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("rebind_string");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A1, B);
    }

    graph.execute();

    graph.rebind(A1, A2);
    C.zero();
    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cr", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A2, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("update_prefactors - changes computation", "[ComputeGraph][Rebind]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);

    cg::Graph graph("update_pf");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    // Get the einsum node's ID
    cg::NodeId const einsum_id = graph.nodes()[0].id;

    // Execute with c_pf=0, ab_pf=1
    auto C_save = Tensor<double, 2>(C);
    graph.execute();
    auto C_v1 = Tensor<double, 2>(C);

    // Update prefactors to c_pf=1, ab_pf=2
    graph.update_prefactors(einsum_id, 1.0, 2.0);

    // Execute again — should compute C = 1*C + 2*A*B
    graph.execute();

    // Verify: C_new = C_v1 + 2*A*B
    auto C_ref = Tensor<double, 2>(C_v1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &C_ref, 2.0, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}
