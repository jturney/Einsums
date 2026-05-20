//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_DeadNodeElimination.cpp
/// @brief Unit tests for the DeadNodeElimination optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("DeadNodeElimination - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("dne_empty");

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - keeps nodes with user-owned outputs", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("dne_user_owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - eliminates intermediate with no reader", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("dne_intermediate");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
    }

    size_t const nodes_before = graph.num_nodes();
    REQUIRE(nodes_before >= 1);

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK(modified);
    CHECK(pass.num_eliminated() >= 1);
    CHECK(graph.num_nodes() < nodes_before);
}

TEST_CASE("DeadNodeElimination - keeps intermediate with reader", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("dne_live_intermediate");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
        cg::einsum("ik;kj->ij", &C, T, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - rank-3 BatchedGemm intermediate is eliminated", "[ComputeGraph][Passes][HigherRank]") {
    // Strided-batched pattern → BatchedGemm node whose output nobody reads.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);

    cg::Graph graph("dne_rank3");
    auto     &T = graph.create_zero_tensor<double, 3>("T", 3, 6, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &T, A, B);
    }

    // Verify the fast path actually fired.
    bool has_batched = false;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            has_batched = true;
    REQUIRE(has_batched);

    size_t const nodes_before = graph.num_nodes();
    REQUIRE(nodes_before >= 1);

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK(modified);
    CHECK(pass.num_eliminated() >= 1);
}

// ── Loop-aware behavior (need-care groundwork) ───────────────────────────

TEST_CASE("DeadNodeElimination - eliminates a dead intermediate inside a loop body", "[ComputeGraph][Passes][Loop]") {
    // A body intermediate written but never read should be eliminated when
    // DNE recurses into the body (via PassManager).
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("dne_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &dead = body.create_zero_tensor<double, 2>("dead", 4, 4);
        cg::einsum("ik;kj->ij", 0.0, &dead, 1.0, A, A); // written, never read
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, A);    // real work
    }

    size_t const body_nodes_before = body.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    bool const modified = pm.run(g);

    CHECK(modified);
    CHECK(body.num_nodes() == body_nodes_before - 1); // the dead einsum is gone
}

TEST_CASE("DeadNodeElimination - keeps a producer feeding only a nested loop", "[ComputeGraph][Passes][Loop]") {
    // THE groundwork case: `shared` is produced in the outer body but read
    // ONLY by a node inside a nested loop. A Loop node doesn't list its
    // body's reads as inputs, so without collect_subtree_referenced_ptrs
    // DNE would wrongly eliminate `shared`'s producer. It must be kept.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("dne_nested");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });

    // Produce `shared` in the outer body. It's an outer-body intermediate.
    auto &shared = outer.create_zero_tensor<double, 2>("shared", 4, 4);
    {
        cg::CaptureGuard const guard(outer);
        cg::einsum("ik;kj->ij", 0.0, &shared, 1.0, A, A); // sole producer of `shared`
    }
    // Nested loop whose body reads `shared` (its only consumer).
    auto &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, shared, A); // reads `shared`
    }

    size_t const outer_nodes_before = outer.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    pm.run(g);

    // The producer of `shared` must survive — it feeds the nested loop.
    CHECK(outer.num_nodes() == outer_nodes_before);
    bool found_shared_producer = false;
    for (auto const &nd : outer.nodes()) {
        for (auto tid : nd.outputs) {
            if (outer.tensor(tid).tensor_ptr == static_cast<void *>(&shared)) {
                found_shared_producer = true;
            }
        }
    }
    CHECK(found_shared_producer);
}
