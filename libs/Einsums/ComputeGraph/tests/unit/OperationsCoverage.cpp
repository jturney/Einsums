//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Tests for graph operations that previously lacked coverage.
// Tests both capturable operations (ger, direct_product, transpose, gesv)
// and returning-form operations that must throw during capture (dot, norm, det, svd, pow).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <tuple>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ── Capturable operations ──────────────────────────────────────────────

TEST_CASE("Operation - ger (rank-1 update) in graph", "[ComputeGraph][Operations]") {
    auto x = create_random_tensor<double>("x", 4);
    auto y = create_random_tensor<double>("y", 5);
    auto A = create_zero_tensor<double>("A", 4, 5);

    auto A_ref = create_zero_tensor<double>("A_ref", 4, 5);
    linear_algebra::ger(1.0, x, y, &A_ref);

    cg::Graph graph("ger");
    {
        cg::CaptureGuard const guard(graph);
        cg::ger(1.0, x, y, &A);
    }
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(A(ii, jj), Catch::Matchers::WithinRel(A_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Operation - direct product in graph", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    auto C_ref = create_zero_tensor<double>("C_ref", 4, 4);
    linear_algebra::direct_product(1.0, A, B, 0.0, &C_ref);

    cg::Graph graph("direct_product");
    {
        cg::CaptureGuard const guard(graph);
        cg::direct_product(1.0, A, B, 0.0, &C);
    }
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Operation - transpose in graph", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 4, 6);
    auto B = create_zero_tensor<double>("B", 6, 4);

    auto B_ref = create_zero_tensor<double>("B_ref", 6, 4);
    tensor_algebra::transpose(&B_ref, A);

    cg::Graph graph("transpose");
    {
        cg::CaptureGuard const guard(graph);
        cg::transpose(&B, A);
    }
    graph.execute();

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(B(ii, jj), Catch::Matchers::WithinRel(B_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Operation - gesv in graph", "[ComputeGraph][Operations]") {
    auto A = Tensor<double, 2>("A", 3, 3);
    auto B = Tensor<double, 2>("B", 3, 2);

    A.zero();
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = 5.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = 3.0;

    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;
    B(2, 0) = 5.0;
    B(2, 1) = 6.0;

    auto A_copy = Tensor<double, 2>(A);
    auto B_ref  = Tensor<double, 2>(B);
    std::ignore = linear_algebra::gesv(&A_copy, &B_ref);

    auto B_graph = Tensor<double, 2>(B);
    auto A_graph = Tensor<double, 2>(A);

    cg::Graph graph("gesv");
    {
        cg::CaptureGuard const guard(graph);
        cg::gesv(&A_graph, &B_graph);
    }
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            REQUIRE_THAT(B_graph(ii, jj), Catch::Matchers::WithinRel(B_ref(ii, jj), 1e-10));
        }
    }
}

// ── Returning-form operations throw during capture ────────────────────

TEST_CASE("Operation - dot throws during capture", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 10);
    auto B = create_random_tensor<double>("B", 10);

    // Eager use (outside capture) should work fine
    double       result = cg::dot(A, B);
    double const ref    = linear_algebra::dot(A, B);
    REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref, 1e-12));

    // During capture, should throw
    cg::Graph graph("dot_throw");
    REQUIRE_THROWS_AS(([&]() {
                          cg::CaptureGuard const guard(graph);
                          cg::dot(A, B);
                      })(),
                      std::logic_error);
}

TEST_CASE("Operation - det throws during capture", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    // Eager
    double       result = cg::det(A);
    double const ref    = linear_algebra::det(A);
    REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref, 1e-10));

    // Capture should throw
    cg::Graph graph("det_throw");
    REQUIRE_THROWS_AS(([&]() {
                          cg::CaptureGuard const guard(graph);
                          cg::det(A);
                      })(),
                      std::logic_error);
}

TEST_CASE("Operation - svd returning form throws during capture", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 4, 3);

    // Eager
    auto [U, s, Vt] = cg::svd(A);
    REQUIRE(s.dim(0) == 3);

    // Capture should throw
    cg::Graph graph("svd_throw");
    REQUIRE_THROWS_AS(([&]() {
                          cg::CaptureGuard const guard(graph);
                          cg::svd(A);
                      })(),
                      std::logic_error);
}

TEST_CASE("Operation - pow returning form throws during capture", "[ComputeGraph][Operations]") {
    auto A = Tensor<double, 2>("A", 3, 3);
    A.zero();
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.5;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 0.5;
    A(2, 0) = 0.5;
    A(2, 1) = 0.5;
    A(2, 2) = 2.0;

    // Eager
    auto result = cg::pow(A, 2.0);
    REQUIRE(result.dim(0) == 3);

    // Capture should throw
    cg::Graph graph("pow_throw");
    REQUIRE_THROWS_AS(([&]() {
                          cg::CaptureGuard const guard(graph);
                          cg::pow(A, 2.0);
                      })(),
                      std::logic_error);
}

// ── Chain of operations ──────────────────────────────────────────────

TEST_CASE("Operation - chain of mixed ops in graph", "[ComputeGraph][Operations]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    auto C_ref = create_zero_tensor<double>("C_ref", 4, 4);
    auto D_ref = create_zero_tensor<double>("D_ref", 4, 4);
    linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C_ref);
    linear_algebra::scale(0.5, &C_ref);
    tensor_algebra::permute(0.0, Indices{i, j}, &D_ref, 1.0, Indices{i, j}, C_ref);
    linear_algebra::axpy(1.0, A, &D_ref);

    cg::Graph graph("mixed_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
        cg::scale(0.5, &C);
        cg::permute("ij <- ij", 0.0, &D, 1.0, C);
        cg::axpy(1.0, A, &D);
    }

    REQUIRE(graph.num_nodes() == 4);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(D(ii, jj), Catch::Matchers::WithinRel(D_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Operation - ger then gemm in graph", "[ComputeGraph][Operations]") {
    auto x = create_random_tensor<double>("x", 4);
    auto y = create_random_tensor<double>("y", 4);
    auto A = create_zero_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    // Reference: A = x*y^T, then C = A*B
    auto A_ref = create_zero_tensor<double>("A_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 4);
    linear_algebra::ger(1.0, x, y, &A_ref);
    linear_algebra::gemm<false, false>(1.0, A_ref, B, 0.0, &C_ref);

    cg::Graph graph("ger_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::ger(1.0, x, y, &A);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
        }
    }
}
