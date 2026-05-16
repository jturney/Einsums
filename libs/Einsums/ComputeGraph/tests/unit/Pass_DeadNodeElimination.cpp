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
