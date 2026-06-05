//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ─── symmetrize ─────────────────────────────────────────────────────────────

TEST_CASE("Blueprint - symmetrize", "[ComputeGraph][Blueprints]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    // Make a copy for reference
    auto A_ref = Tensor<double, 2>(A);
    // Manual symmetrize
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            A_ref(ii, jj) = 0.5 * (A_ref(ii, jj) + A(jj, ii));
        }
    }

    cg::blueprints::symmetrize(&A);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(A(ii, jj) - A_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(A(ii, jj) - A(jj, ii)) < 1e-12); // Must be symmetric
        }
    }
}

TEST_CASE("Blueprint - symmetrize in graph", "[ComputeGraph][Blueprints]") {
    auto A      = create_random_tensor<double>("A", 3, 3);
    auto A_orig = Tensor<double, 2>(A);

    cg::Graph graph("sym_graph");
    {
        cg::CaptureGuard const guard(graph);
        cg::blueprints::symmetrize(&A);
    }

    // Reset A and execute
    A = A_orig;
    graph.execute();

    // Verify symmetric
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A(ii, jj) - A(jj, ii)) < 1e-12);
        }
    }
}

// ─── antisymmetrize ─────────────────────────────────────────────────────────

TEST_CASE("Blueprint - antisymmetrize", "[ComputeGraph][Blueprints]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::blueprints::antisymmetrize(&A);

    // Must be antisymmetric: A(i,j) = -A(j,i), diagonal = 0
    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(A(ii, ii)) < 1e-12); // Diagonal is zero
        for (size_t jj = ii + 1; jj < 4; jj++) {
            REQUIRE(std::abs(A(ii, jj) + A(jj, ii)) < 1e-12);
        }
    }
}

// ─── tensor_trace ───────────────────────────────────────────────────────────

TEST_CASE("Blueprint - tensor_trace", "[ComputeGraph][Blueprints]") {
    auto A      = create_random_tensor<double>("A", 5, 5);
    auto result = Tensor<double, 1>("trace", 1);

    // Reference: manual trace
    double ref_trace = 0.0;
    for (size_t ii = 0; ii < 5; ii++) {
        ref_trace += A(ii, ii);
    }

    cg::blueprints::tensor_trace(&result, A);

    REQUIRE(std::abs(result(0) - ref_trace) < 1e-12);
}

TEST_CASE("Blueprint - tensor_trace in graph", "[ComputeGraph][Blueprints]") {
    auto A      = create_random_tensor<double>("A", 4, 4);
    auto result = Tensor<double, 1>("trace", 1);

    double ref_trace = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        ref_trace += A(ii, ii);
    }

    cg::Graph graph("trace_graph");
    {
        cg::CaptureGuard const guard(graph);
        cg::blueprints::tensor_trace(&result, A);
    }

    graph.execute();

    REQUIRE(std::abs(result(0) - ref_trace) < 1e-12);
}

// ─── matrix_exponential ─────────────────────────────────────────────────────

TEST_CASE("Blueprint - matrix_exponential of zero matrix", "[ComputeGraph][Blueprints]") {
    auto A    = create_zero_tensor<double>("A", 3, 3);
    auto expA = create_zero_tensor<double>("expA", 3, 3);

    cg::blueprints::matrix_exponential(&expA, A, 5);

    // exp(0) = I
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            double const expected = (ii == jj) ? 1.0 : 0.0;
            REQUIRE(std::abs(expA(ii, jj) - expected) < 1e-10);
        }
    }
}

TEST_CASE("Blueprint - matrix_exponential of diagonal", "[ComputeGraph][Blueprints]") {
    auto A  = create_zero_tensor<double>("A", 3, 3);
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    A(2, 2) = 0.5;

    auto expA = create_zero_tensor<double>("expA", 3, 3);

    cg::blueprints::matrix_exponential(&expA, A, 20);

    // exp(diag) = diag(exp(d_i))
    REQUIRE(std::abs(expA(0, 0) - std::exp(1.0)) < 1e-8);
    REQUIRE(std::abs(expA(1, 1) - std::exp(2.0)) < 1e-8);
    REQUIRE(std::abs(expA(2, 2) - std::exp(0.5)) < 1e-8);
    REQUIRE(std::abs(expA(0, 1)) < 1e-10); // Off-diagonal should be ~0
}

// ─── orthogonalize ──────────────────────────────────────────────────────────

TEST_CASE("Blueprint - orthogonalize", "[ComputeGraph][Blueprints]") {
    auto S = create_random_definite<double>("S", 4, 4);
    auto X = create_zero_tensor<double>("X", 4, 4);

    cg::blueprints::orthogonalize(&X, S);

    // Verify: X^T * S * X ≈ I
    auto tmp     = create_zero_tensor<double>("tmp", 4, 4);
    auto I_check = create_zero_tensor<double>("I", 4, 4);

    // tmp = S * X
    linear_algebra::gemm<false, false>(1.0, S, X, 0.0, &tmp);
    // I_check = X^T * tmp = X^T * S * X
    linear_algebra::gemm<true, false>(1.0, X, tmp, 0.0, &I_check);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            double const expected = (ii == jj) ? 1.0 : 0.0;
            REQUIRE(std::abs(I_check(ii, jj) - expected) < 1e-8);
        }
    }
}

// ─── canonical_orthogonalize ────────────────────────────────────────────────

TEST_CASE("Blueprint - canonical_orthogonalize", "[ComputeGraph][Blueprints]") {
    auto S = create_random_definite<double>("S", 4, 4);
    auto X = create_zero_tensor<double>("X", 4, 4);

    size_t const removed = cg::blueprints::canonical_orthogonalize(&X, S, 1e-6);

    // For a well-conditioned positive definite matrix, nothing should be removed
    REQUIRE(removed == 0);

    // Verify: X^T * S * X ≈ I
    auto tmp     = create_zero_tensor<double>("tmp", 4, 4);
    auto I_check = create_zero_tensor<double>("I", 4, 4);

    linear_algebra::gemm<false, false>(1.0, S, X, 0.0, &tmp);
    linear_algebra::gemm<true, false>(1.0, X, tmp, 0.0, &I_check);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            double const expected = (ii == jj) ? 1.0 : 0.0;
            REQUIRE(std::abs(I_check(ii, jj) - expected) < 1e-8);
        }
    }
}
