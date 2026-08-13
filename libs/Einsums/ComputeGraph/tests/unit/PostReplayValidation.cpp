//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PostReplayValidation.cpp
/// @brief Tensor-liveness validation when a pass runs AFTER the graph replayed.
///
/// ``execute()`` validates every handle only while the graph has not run
/// (Graph.cpp, the ``if (!_executed)`` guard). A modifying pass calls
/// ``Graph::mark_sorted()``, which clears that flag, so the next replay
/// re-validates a graph whose View executors have already re-emplaced their
/// holders once. The handle a View node registered carried a liveness token
/// taken from the view INSTANCE alive at capture, and that instance died on the
/// first replay: re-validation then reported a live, graph-owned view as
/// destroyed. These cases pin both halves of the contract - the false positive
/// is gone, and a genuinely destroyed operand is still caught, both before a
/// replay and after one.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledRuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Grid type for the tiled cases below: one tile per axis keeps the graph
/// trivial while still giving the handle table a tensor the graph does NOT
/// adopt (TiledRuntimeTensor exposes no shallow_alias, so its handle's
/// validator watches the caller's own object - which is what makes it the
/// vehicle for testing genuine destruction).
using Grid = std::vector<std::vector<int>>;

void fill_tile(TiledRuntimeTensor<double> &t, std::vector<int> const &coord, double base) {
    auto &tile = t.tile(coord);
    tile.materialize();
    for (size_t i = 0; i < tile.dim(0); ++i) {
        for (size_t j = 0; j < tile.dim(1); ++j) {
            tile(i, j) = base + static_cast<double>(2 * i + j);
        }
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. The defect: a pass applied after a replay must not condemn a live view
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Post-replay pass - a typed view operand is not reported destroyed", "[ComputeGraph][Lifetime][Validation][View]") {
    constexpr size_t N = 6;
    constexpr size_t M = 3;

    auto T = create_random_tensor<double>("T", N, N);
    auto E = create_random_tensor<double>("E", M, M);
    auto A = create_random_tensor<double>("A", M, M);
    auto D = create_zero_tensor<double>("D", M, M);
    auto F = create_zero_tensor<double>("F", M, M);

    auto D_twin = create_zero_tensor<double>("Dtwin", M, M);
    auto F_twin = create_zero_tensor<double>("Ftwin", M, M);

    auto build = [&](cg::Graph &g, Tensor<double, 2> *out_d, Tensor<double, 2> *out_f) {
        cg::CaptureGuard const guard(g);
        auto                  &slice = cg::view<double, 2>(T, cg::ViewAxis::range(0, M), cg::ViewAxis::range(0, M));
        cg::einsum("ik;kj->ij", 0.0, out_d, 1.0, E, slice);
        cg::einsum("ik;kj->ij", 0.0, out_f, 1.0, A, slice);
    };

    cg::Graph graph("post_replay_view");
    build(graph, &D, &F);

    // The twin never sees a pass: it is the oracle for the values.
    cg::Graph twin("post_replay_view_twin");
    build(twin, &D_twin, &F_twin);

    graph.execute();
    twin.execute();

    // A modifying pass leaves the graph "not yet executed", so the replay below
    // re-runs validate_tensors() over handles whose views the first replay
    // already re-emplaced. Reorder only marks the graph when it actually moves
    // a node, and this schedule is already optimal, so the state a rewriting
    // pass would leave is spelled out: that flag, not the pass, is the
    // mechanism under test.
    REQUIRE_NOTHROW(graph.apply<cg::passes::Reorder>());
    graph.mark_sorted();
    REQUIRE_NOTHROW(graph.execute());

    twin.execute();

    for (size_t ii = 0; ii < M; ii++) {
        for (size_t jj = 0; jj < M; jj++) {
            REQUIRE(D(ii, jj) == D_twin(ii, jj));
            REQUIRE(F(ii, jj) == F_twin(ii, jj));
        }
    }
}

TEST_CASE("Post-replay pass - a runtime-rank view operand is not reported destroyed", "[ComputeGraph][Lifetime][Validation][View]") {
    RuntimeTensor<double> T("T", {6UL, 6UL});
    RuntimeTensor<double> E("E", {3UL, 3UL});
    RuntimeTensor<double> D("D", {3UL, 3UL});
    RuntimeTensor<double> D_twin("Dtwin", {3UL, 3UL});

    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 6; ++j)
            T(i, j) = 0.5 + static_cast<double>(6 * i + j);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            E(i, j) = 1.0 - 0.25 * static_cast<double>(3 * i + j);

    auto build = [&](cg::Graph &g, RuntimeTensor<double> *out) {
        cg::CaptureGuard const guard(g);
        auto                  &slice = cg::view_runtime(T, {cg::ViewAxis::range(1, 4), cg::ViewAxis::range(2, 5)});
        cg::einsum("ik;kj->ij", 0.0, out, 1.0, E, slice);
    };

    cg::Graph graph("post_replay_view_rt");
    build(graph, &D);
    cg::Graph twin("post_replay_view_rt_twin");
    build(twin, &D_twin);

    graph.execute();
    twin.execute();

    // mark_sorted() on its own is the exact state every modifying pass leaves
    // behind, so this is the mechanism with no pass in the way.
    graph.mark_sorted();
    REQUIRE_NOTHROW(graph.execute());
    twin.execute();

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(D(ii, jj) == D_twin(ii, jj));
}

TEST_CASE("Post-replay pass - a tile view operand is not reported destroyed", "[ComputeGraph][Lifetime][Validation][View][TiledRuntime]") {
    TiledRuntimeTensor<double> A("A", Grid{{2}, {2}});
    TiledRuntimeTensor<double> B("B", Grid{{2}, {2}});
    RuntimeTensor<double>      C("C", {2UL, 2UL});
    RuntimeTensor<double>      C_twin("Ctwin", {2UL, 2UL});

    fill_tile(A, {0, 0}, 1.0);
    fill_tile(B, {0, 0}, 0.5);

    auto build = [&](cg::Graph &g, RuntimeTensor<double> *out) {
        cg::CaptureGuard const guard(g);
        auto                  &a = cg::tile_view_python(A, std::vector<int>{0, 0});
        auto                  &b = cg::tile_view_python(B, std::vector<int>{0, 0});
        cg::einsum("ik;kj->ij", 0.0, out, 1.0, a, b);
    };

    cg::Graph graph("post_replay_tile_view");
    build(graph, &C);
    cg::Graph twin("post_replay_tile_view_twin");
    build(twin, &C_twin);

    graph.execute();
    twin.execute();

    graph.mark_sorted();
    REQUIRE_NOTHROW(graph.execute());
    twin.execute();

    for (size_t ii = 0; ii < 2; ii++)
        for (size_t jj = 0; jj < 2; jj++)
            REQUIRE(C(ii, jj) == C_twin(ii, jj));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Adversarial: a genuinely destroyed operand is still caught, both orderings
// ═══════════════════════════════════════════════════════════════════════════
//
// The vehicle is a TiledRuntimeTensor because the graph does not adopt one
// (no shallow_alias), so its handle's validator watches the CALLER's object and
// its destruction is a real dangling reference rather than a wrapper going away
// while the graph holds a share of the storage.

TEST_CASE("Validation catches a destroyed operand before the first replay", "[ComputeGraph][Lifetime][Validation]") {
    auto A = std::make_unique<TiledRuntimeTensor<double>>("A", Grid{{2}, {2}});
    auto B = std::make_unique<TiledRuntimeTensor<double>>("B", Grid{{2}, {2}});
    fill_tile(*A, {0, 0}, 1.0);
    fill_tile(*B, {0, 0}, 0.5);
    TiledRuntimeTensor<double> C("C", Grid{{2}, {2}});

    cg::Graph graph("dangling_before_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, *A, *B);
    }

    A.reset(); // genuinely gone: nothing holds this tensor any more

    REQUIRE_THROWS_WITH(graph.execute(), Catch::Matchers::ContainsSubstring("appears to have been destroyed"));
}

TEST_CASE("Validation catches a destroyed operand after a replay and mark_sorted", "[ComputeGraph][Lifetime][Validation]") {
    auto A = std::make_unique<TiledRuntimeTensor<double>>("A", Grid{{2}, {2}});
    auto B = std::make_unique<TiledRuntimeTensor<double>>("B", Grid{{2}, {2}});
    fill_tile(*A, {0, 0}, 1.0);
    fill_tile(*B, {0, 0}, 0.5);
    TiledRuntimeTensor<double> C("C", Grid{{2}, {2}});

    cg::Graph graph("dangling_after_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, *A, *B);
    }

    REQUIRE_NOTHROW(graph.execute());

    A.reset();

    // Re-validation is what a pass buys back, and it has to be worth having:
    // the destroyed operand must be reported here, not dereferenced. This is
    // also the only ordering in which a post-replay destruction is checked at
    // all: ``execute()`` validates while ``!_executed``, so a tensor dropped
    // after a successful replay that no pass invalidated is not re-examined.
    // That is the standing contract ("all tensors outlive the graph"), not
    // something this fix narrowed.
    graph.mark_sorted();
    REQUIRE_THROWS_WITH(graph.execute(), Catch::Matchers::ContainsSubstring("appears to have been destroyed"));
}
