//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <limits>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;
using einsums::testing::reference_permute;

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Within @p for_double of @p want, relative to it and absolute near zero; a narrower type gets a
/// hundred ulps of its own precision instead when that is looser.
template <typename T>
bool near(T got, T want, double for_double) {
    double const tol = std::max(for_double, 100.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon());
    return std::abs(got - want) <= tol * (1.0 + std::abs(want));
}

} // namespace

TEMPLATE_LIST_TEST_CASE("Graph - gemm operation", "[ComputeGraph][Phase2]", testing::AllScalarTypes) {
    using T         = TestType;
    auto A          = create_random_tensor<T>("A", 4, 3);
    auto B          = create_random_tensor<T>("B", 3, 5);
    auto C          = create_zero_tensor<T>("C", 4, 5);
    auto C_expected = create_zero_tensor<T>("Ce", 4, 5);

    linear_algebra::gemm<false, false>(T(1.0), A, B, T(0.0), &C_expected);

    cg::Graph graph("test_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(T(1.0), A, B, T(0.0), &C);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(near<T>(C(ii, jj), C_expected(ii, jj), 1e-12));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Graph - gemv operation", "[ComputeGraph][Phase2]", testing::AllScalarTypes) {
    using T         = TestType;
    auto A          = create_random_tensor<T>("A", 4, 3);
    auto x          = create_random_tensor<T>("x", 3);
    auto y          = create_zero_tensor<T>("y", 4);
    auto y_expected = create_zero_tensor<T>("ye", 4);

    linear_algebra::gemv<false>(T(1.0), A, x, T(0.0), &y_expected);

    cg::Graph graph("test_gemv");
    {
        cg::CaptureGuard const guard(graph);
        cg::gemv<false>(T(1.0), A, x, T(0.0), &y);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(near<T>(y(ii), y_expected(ii), 1e-12));
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

TEMPLATE_LIST_TEST_CASE("Graph - syev in-place form", "[ComputeGraph][Phase2]", testing::RealScalarTypes) {
    using T    = TestType;
    auto A     = create_random_definite<T>("A", 4, 4);
    auto A_ref = Tensor<T, 2>(A);
    auto W     = create_zero_tensor<T>("W", 4);
    auto W_ref = create_zero_tensor<T>("Wref", 4);

    linear_algebra::syev(&A_ref, &W_ref);

    cg::Graph graph("test_syev_inplace");
    {
        cg::CaptureGuard const guard(graph);
        cg::syev(&A, &W);
    }

    // syev does NOT execute during capture; must call execute()
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(near<T>(W(ii), W_ref(ii), 1e-10));
    }
}

TEMPLATE_LIST_TEST_CASE("Graph - qr outside capture", "[ComputeGraph][Phase2]", testing::RealScalarTypes) {
    using T = TestType;
    auto A  = create_random_tensor<T>("A", 4, 4);

    // qr returning form works outside capture
    auto [Q, R] = cg::qr(A);

    // Q should be orthogonal: Q^T * Q ≈ I
    auto QtQ = create_zero_tensor<T>("QtQ", 4, 4);
    linear_algebra::gemm<true, false>(T(1.0), Q, Q, T(0.0), &QtQ);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            T const expected = (ii == jj) ? T(1.0) : T(0.0);
            REQUIRE(near<T>(QtQ(ii, jj), expected, 1e-10));
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

TEMPLATE_LIST_TEST_CASE("Graph - element_transform", "[ComputeGraph][Phase2]", testing::AllScalarTypes) {
    using T         = TestType;
    auto A          = create_random_tensor<T>("A", 3, 3);
    auto A_expected = Tensor<T, 2>(A);

    // Eager: square each element
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            A_expected(i, j) = A_expected(i, j) * A_expected(i, j);
        }
    }

    cg::Graph graph("test_element_transform");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, [](T x) { return x * x; });
    }

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(near<T>(A(ii, jj), A_expected(ii, jj), 1e-12));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Graph - mixed operations pipeline", "[ComputeGraph][Phase2]", testing::AllScalarTypes) {
    using T = TestType;
    // Test a realistic workflow mixing einsum, scale, gemm, and element_transform
    auto A = create_random_tensor<T>("A", 4, 4);
    auto B = create_random_tensor<T>("B", 4, 4);
    auto C = create_zero_tensor<T>("C", 4, 4);
    auto D = create_zero_tensor<T>("D", 4, 4);

    // Expected: C = A * B, D = 2 * C with elements squared
    auto C_ref = create_zero_tensor<T>("Cref", 4, 4);
    auto D_ref = create_zero_tensor<T>("Dref", 4, 4);

    reference_einsum("ij <- ik ; kj", &C_ref, A, B);
    reference_permute("ij <- ij", T(0.0), &D_ref, T(2.0), C_ref);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            D_ref(i, j) = D_ref(i, j) * D_ref(i, j);
        }
    }

    // Graph version
    cg::Graph graph("test_mixed");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::permute("ij <- ij", T(0.0), &D, T(2.0), C);
        cg::element_transform(&D, [](T x) { return x * x; });
    }

    REQUIRE(graph.num_nodes() == 3);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(near<T>(D(ii, jj), D_ref(ii, jj), 1e-10));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Graph - axpby operation", "[ComputeGraph][Phase2]", testing::AllScalarTypes) {
    using T    = TestType;
    auto X     = create_random_tensor<T>("X", 5);
    auto Y     = create_random_tensor<T>("Y", 5);
    auto Y_ref = Tensor<T, 1>(Y);

    linear_algebra::axpby(T(2.0), X, T(3.0), &Y_ref);

    cg::Graph graph("test_axpby");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpby(T(2.0), X, T(3.0), &Y);
    }

    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        REQUIRE(near<T>(Y(ii), Y_ref(ii), 1e-12));
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
