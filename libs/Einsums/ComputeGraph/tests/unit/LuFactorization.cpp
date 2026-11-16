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

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Within 1000 epsilon of @p want, relative to it and absolute near zero: a solve's roundoff on
/// these small well-conditioned systems, for any element type.
template <typename T>
bool solved(T got, T want) {
    return std::abs(got - want) <= 1000.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon() * (1.0 + std::abs(want));
}

} // namespace

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
    // The factorization tensor and the pivots both order the pair: the solve reads A and the pivot operand the factorization
    // wrote. If neither edge were drawn the two nodes would land on one execution level and the solve could run against an
    // unfactored matrix.
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

TEMPLATE_LIST_TEST_CASE("Graph - getrs against a rank-1 right-hand side", "[ComputeGraph][LU]", testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t n = 5;

    auto A = create_random_definite<T>("A", n, n);
    auto b = create_random_tensor<T>("b", n);

    auto A_ref = Tensor<T, 2>(A);
    auto b_ref = Tensor<T, 1>(b);
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
        REQUIRE(solved<T>(b(i), b_ref(i)));
    }
}

TEMPLATE_LIST_TEST_CASE("Graph - getrs through a view of the right-hand sides", "[ComputeGraph][LU]", testing::AllScalarTypes) {
    using T = TestType;
    // The DLPNO (T0) fit solves against COLUMN SLICES of one right-hand-side store, so the non-owning path is the one the
    // production caller takes.
    constexpr size_t n = 4;

    auto A     = create_random_definite<T>("A", n, n);
    auto store = create_random_tensor<T>("store", n, 6);

    auto A_ref     = Tensor<T, 2>(A);
    auto store_ref = Tensor<T, 2>(store);
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
            REQUIRE(solved<T>(store(i, j), store_ref(i, j)));
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

    REQUIRE_THROWS_AS(std::ignore = linear_algebra::getrs(A, pivots, &B), TensorCompatError);
}

TEST_CASE("Graph - getrs refuses a pivot array too short for the order", "[ComputeGraph][LU]") {
    auto A = create_random_definite<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 2);

    std::vector<blas::int_t> pivots(4);
    REQUIRE(linear_algebra::getrf(&A, &pivots) == 0);

    std::vector<blas::int_t> stunted(2);
    REQUIRE_THROWS_AS(std::ignore = linear_algebra::getrs(A, stunted, &B), std::length_error);
}

TEMPLATE_LIST_TEST_CASE("Graph - a captured LU replays against the same pivots", "[ComputeGraph][LU]", testing::AllScalarTypes) {
    using T = TestType;
    // The pivot operand's handle holds the buffer as a shared_ptr, so it outlives the LuPivots the caller held and every
    // replay sees the array the factorization node just wrote.
    constexpr size_t n = 4;

    auto A = create_random_definite<T>("A", n, n);
    auto B = create_random_tensor<T>("B", n, 2);

    auto A_source = Tensor<T, 2>(A);
    auto B_source = Tensor<T, 2>(B);

    auto A_ref = Tensor<T, 2>(A);
    auto B_ref = Tensor<T, 2>(B);
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
                REQUIRE(solved<T>(B(i, j), B_ref(i, j)));
            }
        }
    }
}

TEST_CASE("Graph - the pivots alone order a solve after its factorization", "[ComputeGraph][LU]") {
    // The solve below reads a COPY of the factorization, so the pivots are the only operand the two nodes share. When the
    // pivots rode outside the dataflow nothing ordered this pair, and a concurrent executor could run the solve against an
    // empty pivot array. They are the factorization's second output and the solve's second input now.
    constexpr size_t n = 4;

    auto A      = create_random_definite<double>("A", n, n);
    auto LU     = Tensor<double, 2>(A);
    auto B      = create_random_tensor<double>("B", n, 2);
    auto B_ref  = Tensor<double, 2>(B);
    auto A_ref  = Tensor<double, 2>(A);
    auto B_save = Tensor<double, 2>(B);
    REQUIRE(linear_algebra::gesv(&A_ref, &B_ref) == 0);

    // LU holds the factorization the graph's getrf will reproduce, so the solve against it is the right one once the
    // pivots are filled.
    std::vector<blas::int_t> eager_pivots(n);
    REQUIRE(linear_algebra::getrf(&LU, &eager_pivots) == 0);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-pivot-edge");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(LU, pivots, &B);
    }
    REQUIRE(graph.num_nodes() == 2);
    cg::TensorId const p_id = graph.nodes()[0].outputs.at(1);
    CHECK(graph.nodes()[1].inputs.at(1) == p_id);

    graph.topological_sort();
    auto const &preds = graph.dependencies().predecessors;
    REQUIRE(preds.size() == 2);
    CHECK(std::ranges::find(preds[1], std::size_t{0}) != preds[1].end());

    auto A_save = Tensor<double, 2>(A);
    for (auto const &executor :
         std::vector<std::shared_ptr<cg::Executor>>{std::make_shared<cg::DataflowExecutor>(), std::make_shared<cg::OpenMPExecutor>()}) {
        for (int replay = 0; replay < 5; ++replay) {
            A = A_save;
            B = B_save;
            graph.execute(*executor);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    REQUIRE(solved<double>(B(i, j), B_ref(i, j)));
                }
            }
        }
    }
}

TEST_CASE("Graph - a replayed getrf of a singular matrix throws", "[ComputeGraph][LU]") {
    // LAPACK completes the factorization of a singular matrix and reports info > 0, which the eager call hands back. A replay
    // discarded it, so a solve against the factorization ran on and wrote garbage with no word to anyone. The matrix is
    // regular when the graph is captured and singular only in the data the replay sees.
    constexpr size_t n = 3;

    auto A = create_random_definite<double>("A", n, n);
    auto B = create_random_tensor<double>("B", n, 1);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-singular");
    {
        cg::CaptureGuard const guard(graph);
        cg::getrf(&A, &pivots);
        cg::getrs(A, pivots, &B);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = static_cast<double>(i + 1) * static_cast<double>(j + 1); // rank one, so singular
        }
    }
    REQUIRE_THROWS_WITH(graph.execute(), Catch::Matchers::ContainsSubstring("getrf") && Catch::Matchers::ContainsSubstring("singular"));

    // The eager call keeps its contract: it returns the code rather than throwing.
    auto S = create_zero_tensor<double>("S", n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            S(i, j) = static_cast<double>(i + 1) * static_cast<double>(j + 1);
        }
    }
    cg::LuPivots eager;
    CHECK(cg::getrf(&S, &eager) > 0);
}

TEST_CASE("Graph - a factorization whose only consumer reads its pivots survives the lifecycle passes", "[ComputeGraph][LU]") {
    // The getrf below factors graph-owned scratch that nothing reads afterwards. Its pivots are all the solve takes from it,
    // since the solve reads a separate copy of the factorization, so a pass that judged the node by its tensor outputs alone
    // would delete it and leave the solve reading an empty pivot array. And a pivot array is the caller's state, never
    // storage the graph frees.
    constexpr size_t n = 4;

    auto A      = create_random_definite<double>("A", n, n);
    auto LU     = Tensor<double, 2>(A);
    auto B      = create_random_tensor<double>("B", n, 2);
    auto B_ref  = Tensor<double, 2>(B);
    auto A_ref  = Tensor<double, 2>(A);
    auto B_save = Tensor<double, 2>(B);
    REQUIRE(linear_algebra::gesv(&A_ref, &B_ref) == 0);
    std::vector<blas::int_t> eager_pivots(n);
    REQUIRE(linear_algebra::getrf(&LU, &eager_pivots) == 0);

    cg::LuPivots pivots;
    cg::Graph    graph("lu-lifecycle");
    {
        cg::CaptureGuard const guard(graph);
        auto                  &S = graph.create_tensor<double, 2>("S", n, n);
        cg::axpby(1.0, A, 0.0, &S);
        cg::getrf(&S, &pivots);
        cg::getrs(LU, pivots, &B);
    }

    (void)graph.apply<cg::passes::DeadNodeElimination>();
    (void)graph.apply<cg::passes::FreeInsertion>(std::size_t{0});

    auto const factor = std::ranges::find_if(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Getrf; });
    REQUIRE(factor != graph.nodes().end());
    REQUIRE(factor->outputs.size() == 2);
    cg::TensorId const p_id = factor->outputs[1];
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Free) {
            CHECK(std::ranges::find(node.inputs, p_id) == node.inputs.end());
        }
    }

    graph.execute();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            REQUIRE(solved<double>(B(i, j), B_ref(i, j)));
        }
    }
}
