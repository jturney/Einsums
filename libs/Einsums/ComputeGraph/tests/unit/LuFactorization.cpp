//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEMPLATE_TEST_CASE("Graph - getrf + getrs round trip matches gesv", "[ComputeGraph][LU]", float, double) {
    using T = TestType;

    constexpr size_t n    = 6;
    constexpr size_t nrhs = 3;

    auto A = create_random_definite<T>("A", n, n);
    auto B = create_random_tensor<T>("B", n, nrhs);

    // gesv is the oracle: the same factorization and the same solve, spelled as one destructive call.
    auto A_ref = Tensor<T, 2>(A);
    auto B_ref = Tensor<T, 2>(B);
    REQUIRE(linear_algebra::gesv(&A_ref, &B_ref) == 0);

    auto         A_graph = Tensor<T, 2>(A);
    auto         B_graph = Tensor<T, 2>(B);
    cg::LuPivots pivots;

    cg::Graph graph("lu");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A_graph, &pivots);
        cg::getrs(A_graph, pivots, &B_graph);
    }

    REQUIRE(graph.num_nodes() == 2);
    // Nothing has run yet: the pivots are sized by the executor, not by capture.
    REQUIRE(pivots.size() == 0);

    graph.execute();

    REQUIRE(pivots.size() == n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nrhs; j++) {
            REQUIRE_THAT(B_graph(i, j), Catch::Matchers::WithinAbs(B_ref(i, j), 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("Graph - one factorization serves many right-hand sides", "[ComputeGraph][LU]", float, double) {
    using T = TestType;

    constexpr size_t n = 5;

    auto A  = create_random_definite<T>("A", n, n);
    auto B1 = create_random_tensor<T>("B1", n, 2);
    auto B2 = create_random_tensor<T>("B2", n, 4);

    auto A_ref  = Tensor<T, 2>(A);
    auto B1_ref = Tensor<T, 2>(B1);
    auto B2_ref = Tensor<T, 2>(B2);
    // Two gesv calls need two copies of A, which is the cost getrs removes.
    auto A_ref2 = Tensor<T, 2>(A);
    REQUIRE(linear_algebra::gesv(&A_ref, &B1_ref) == 0);
    REQUIRE(linear_algebra::gesv(&A_ref2, &B2_ref) == 0);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-many");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &B1);
        cg::getrs(A, pivots, &B2);
    }
    graph.execute();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 2; j++) {
            REQUIRE_THAT(B1(i, j), Catch::Matchers::WithinAbs(B1_ref(i, j), 1e-6));
        }
        for (size_t j = 0; j < 4; j++) {
            REQUIRE_THAT(B2(i, j), Catch::Matchers::WithinAbs(B2_ref(i, j), 1e-6));
        }
    }
}

TEST_CASE("Graph - a captured getrf is ordered before its solves", "[ComputeGraph][LU]") {
    // The pivots ride outside the dataflow, so what orders the pair is the factorization TENSOR. Recording the solve's read of
    // A as an input is what makes the hazard scan draw the edge; if it stopped doing so the two nodes would land on one
    // execution level and the solve could run against an unfactored matrix.
    constexpr size_t n = 4;

    auto A = create_random_definite<double>("A", n, n);
    auto B = create_random_tensor<double>("B", n, 1);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-order");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &B);
    }

    // The schedule is built on demand; asking for it is what the executors do.
    graph.topological_sort();
    auto const &levels = graph.dependencies().levels;
    REQUIRE(levels.size() == 2);
    REQUIRE(levels[0].size() == 1);
    REQUIRE(levels[1].size() == 1);
}

TEST_CASE("Graph - eager getrs leaves the factorization alone", "[ComputeGraph][LU]") {
    constexpr size_t n = 4;

    auto A = create_random_definite<double>("A", n, n);
    auto B = create_random_tensor<double>("B", n, 2);

    std::vector<blas::int_t> pivots(n);
    REQUIRE(linear_algebra::getrf(&A, &pivots) == 0);

    auto const A_factored = Tensor<double, 2>(A);
    REQUIRE(linear_algebra::getrs(A, pivots, &B) == 0);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            REQUIRE(A(i, j) == A_factored(i, j));
        }
    }
}

TEST_CASE("Graph - getrs against a rank-1 right-hand side", "[ComputeGraph][LU]") {
    constexpr size_t n = 5;

    auto A = create_random_definite<double>("A", n, n);
    auto b = create_random_tensor<double>("b", n);

    auto A_ref = Tensor<double, 2>(A);
    auto b_ref = Tensor<double, 1>(b);
    REQUIRE(linear_algebra::gesv(&A_ref, &b_ref) == 0);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-vector");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &b);
    }
    graph.execute();

    for (size_t i = 0; i < n; i++) {
        REQUIRE_THAT(b(i), Catch::Matchers::WithinAbs(b_ref(i), 1e-10));
    }
}

TEST_CASE("Graph - getrs through a view of the right-hand sides", "[ComputeGraph][LU]") {
    // The DLPNO (T0) fit solves against COLUMN SLICES of one right-hand-side store, so the non-owning path is the one the
    // production caller takes.
    constexpr size_t n = 4;

    auto A     = create_random_definite<double>("A", n, n);
    auto store = create_random_tensor<double>("store", n, 6);

    auto A_ref     = Tensor<double, 2>(A);
    auto store_ref = Tensor<double, 2>(store);
    REQUIRE(linear_algebra::gesv(&A_ref, &store_ref) == 0);

    auto columns = store(AllT{}, Range{2, 5});

    cg::LuPivots pivots;
    cg::Graph    graph("lu-view");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &columns);
    }
    graph.execute();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 2; j < 5; j++) {
            REQUIRE_THAT(store(i, j), Catch::Matchers::WithinAbs(store_ref(i, j), 1e-10));
        }
    }
}

TEST_CASE("Graph - zero-extent LU is a quick return", "[ComputeGraph][LU]") {
    // An empty domain is a valid domain. LAPACK quick-returns on it and so must the graph, without touching the pivot buffer
    // and without a workspace formula reaching zero.
    auto A_empty = create_zero_tensor<double>("A", 0, 0);
    auto B_empty = create_zero_tensor<double>("B", 0, 3);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-empty");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A_empty, &pivots);
        cg::getrs(A_empty, pivots, &B_empty);
    }
    REQUIRE_NOTHROW(graph.execute());
    REQUIRE(pivots.size() == 0);

    // An order the pivots do cover, but with no right-hand side to solve for.
    auto A_wide  = create_random_definite<double>("Aw", 3, 3);
    auto B_norhs = create_zero_tensor<double>("Bn", 3, 0);

    cg::LuPivots wide_pivots;
    cg::Graph    wide("lu-no-rhs");
    {
        cg::CaptureGuard const guard(wide);
        cg::getrf(&A_wide, &wide_pivots);
        cg::getrs(A_wide, wide_pivots, &B_norhs);
    }
    REQUIRE_NOTHROW(wide.execute());
    REQUIRE(wide_pivots.size() == 3);
}

TEST_CASE("Graph - getrs refuses a mismatched factorization", "[ComputeGraph][LU]") {
    auto A = create_random_definite<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 5, 2);

    std::vector<blas::int_t> pivots(4);
    REQUIRE(linear_algebra::getrf(&A, &pivots) == 0);

    REQUIRE_THROWS_AS(std::ignore = linear_algebra::getrs(A, pivots, &B), tensor_compat_error);
}

TEST_CASE("Graph - getrs refuses a pivot array too short for the order", "[ComputeGraph][LU]") {
    auto A = create_random_definite<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 2);

    std::vector<blas::int_t> pivots(4);
    REQUIRE(linear_algebra::getrf(&A, &pivots) == 0);

    std::vector<blas::int_t> stunted(2);
    REQUIRE_THROWS_AS(std::ignore = linear_algebra::getrs(A, stunted, &B), std::length_error);
}

TEST_CASE("Graph - a captured LU replays against the same pivots", "[ComputeGraph][LU]") {
    // The pivot buffer is baked into both executors as a shared_ptr, so it outlives the handle the caller held and every
    // replay sees the array the factorization node just wrote.
    constexpr size_t n = 4;

    auto A = create_random_definite<double>("A", n, n);
    auto B = create_random_tensor<double>("B", n, 2);

    auto A_source = Tensor<double, 2>(A);
    auto B_source = Tensor<double, 2>(B);

    auto A_ref = Tensor<double, 2>(A);
    auto B_ref = Tensor<double, 2>(B);
    REQUIRE(linear_algebra::gesv(&A_ref, &B_ref) == 0);

    cg::Graph graph("lu-replay");
    {
        cg::LuPivots           pivots;
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &B);
    }

    for (int replay = 0; replay < 3; replay++) {
        A = A_source;
        B = B_source;
        graph.execute();

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < 2; j++) {
                REQUIRE_THAT(B(i, j), Catch::Matchers::WithinAbs(B_ref(i, j), 1e-10));
            }
        }
    }
}
