//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Graph - gemm operation", "[ComputeGraph][Phase2]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C_expected);

    cg::Graph graph("test_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph - gemv operation", "[ComputeGraph][Phase2]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto x          = create_random_tensor<double>("x", 3);
    auto y          = create_zero_tensor<double>("y", 4);
    auto y_expected = create_zero_tensor<double>("ye", 4);

    linear_algebra::gemv<false>(1.0, A, x, 0.0, &y_expected);

    cg::Graph graph("test_gemv");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemv<false>(1.0, A, x, 0.0, &y);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(y(ii) - y_expected(ii)) < 1e-12);
    }
}

TEST_CASE("Graph - syev returning form throws during capture", "[ComputeGraph][Phase2]") {
    auto A = create_random_definite<double>("A", 4, 4);

    cg::Graph graph("test_syev_returning");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_THROWS_AS(cg::syev(A), std::logic_error);
    }
}

TEST_CASE("Graph - syev in-place form", "[ComputeGraph][Phase2]") {
    auto A     = create_random_definite<double>("A", 4, 4);
    auto A_ref = Tensor<double, 2>(A);
    auto W     = create_zero_tensor<double>("W", 4);
    auto W_ref = create_zero_tensor<double>("Wref", 4);

    linear_algebra::syev(&A_ref, &W_ref);

    cg::Graph graph("test_syev_inplace");
    {
        cg::CaptureGuard const guard(graph);
        cg::syev(&A, &W);
    }

    // syev does NOT execute during capture; must call execute()
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(W(ii) - W_ref(ii)) < 1e-10);
    }
}

TEST_CASE("Graph - qr outside capture", "[ComputeGraph][Phase2]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    // qr returning form works outside capture
    auto [Q, R] = cg::qr(A);

    // Q should be orthogonal: Q^T * Q ≈ I
    auto QtQ = create_zero_tensor<double>("QtQ", 4, 4);
    linear_algebra::gemm<true, false>(1.0, Q, Q, 0.0, &QtQ);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            double const expected = (ii == jj) ? 1.0 : 0.0;
            REQUIRE(std::abs(QtQ(ii, jj) - expected) < 1e-10);
        }
    }
}

TEST_CASE("Graph - qr throws during capture", "[ComputeGraph][Phase2]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::Graph graph("test_qr_capture");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_THROWS_AS(cg::qr(A), std::logic_error);
    }
}

TEST_CASE("Graph - element_transform", "[ComputeGraph][Phase2]") {
    auto A          = create_random_tensor<double>("A", 3, 3);
    auto A_expected = Tensor<double, 2>(A);

    // Eager: square each element
    tensor_algebra::element_transform(&A_expected, [](double x) { return x * x; });

    cg::Graph graph("test_element_transform");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, [](double x) { return x * x; });
    }

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A(ii, jj) - A_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Graph - mixed operations pipeline", "[ComputeGraph][Phase2]") {
    // Test a realistic workflow mixing einsum, scale, gemm, and element_transform
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    // Expected: C = A * B, D = 2 * C with elements squared
    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    auto D_ref = create_zero_tensor<double>("Dref", 4, 4);

    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::permute(0.0, Indices{i, j}, &D_ref, 2.0, Indices{i, j}, C_ref);
    tensor_algebra::element_transform(&D_ref, [](double x) { return x * x; });

    // Graph version
    cg::Graph graph("test_mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::permute("ij <- ij", 0.0, &D, 2.0, C);
        cg::element_transform(&D, [](double x) { return x * x; });
    }

    REQUIRE(graph.num_nodes() == 3);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-10);
        }
    }
}

TEST_CASE("Graph - axpby operation", "[ComputeGraph][Phase2]") {
    auto X     = create_random_tensor<double>("X", 5);
    auto Y     = create_random_tensor<double>("Y", 5);
    auto Y_ref = Tensor<double, 1>(Y);

    linear_algebra::axpby(2.0, X, 3.0, &Y_ref);

    cg::Graph graph("test_axpby");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(2.0, X, 3.0, &Y);
    }

    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        REQUIRE(std::abs(Y(ii) - Y_ref(ii)) < 1e-12);
    }
}

TEST_CASE("Graph - invert operation", "[ComputeGraph][Phase2]") {
    auto A      = create_random_definite<double>("A", 3, 3);
    auto A_copy = Tensor<double, 2>(A);
    auto A_ref  = Tensor<double, 2>(A);

    linear_algebra::invert(&A_ref);

    cg::Graph graph("test_invert");
    {
        cg::CaptureGuard const guard(graph);
        cg::invert(&A_copy);
    }

    // invert does NOT execute during capture; must call execute()
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A_copy(ii, jj) - A_ref(ii, jj)) < 1e-10);
        }
    }
}
