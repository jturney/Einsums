//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_LoopInvariantHoisting.cpp
/// @brief Unit tests for the LoopInvariantHoisting optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("LoopInvariantHoisting - empty loop body", "[ComputeGraph][Passes]") {
    cg::Graph graph("lih_empty");
    graph.add_loop("loop", 3, [](size_t iter) { return iter < 2; });

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();
    CHECK_FALSE(modified);
    CHECK(pass.num_hoisted() == 0);
}

TEST_CASE("LoopInvariantHoisting - nothing to hoist", "[ComputeGraph][Passes]") {
    auto value = Tensor<double, 1>("value", 1);
    value(0)   = 100.0;

    cg::Graph graph("no_hoist");

    auto &body = graph.add_loop("loop", 5, [](size_t iter) { return iter < 4; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &value);
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();
    REQUIRE_FALSE(modified);
}

TEST_CASE("LoopInvariantHoisting - hoists invariant node", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("hoist_test");

    auto &body = graph.add_loop("loop", 5, [](size_t iter) { return iter < 4; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(0.9, &C);
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    REQUIRE(modified);
    REQUIRE(pass.num_hoisted() == 1);

    cg::LoopDescriptor const *loop_desc = nullptr;
    for (auto const &node : graph.nodes()) {
        loop_desc = std::get_if<cg::LoopDescriptor>(&node.op_data);
        if (loop_desc)
            break;
    }
    REQUIRE(loop_desc != nullptr);
    REQUIRE(loop_desc->body->num_nodes() == 1);
}

TEST_CASE("LoopInvariantHoisting - dependency chain partially hoists", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_random_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("lih_dep_chain");

    auto &body = graph.add_loop("loop", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
        cg::scale(0.5, &C);
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    CHECK(modified);
    CHECK(pass.num_hoisted() == 1);
}

TEST_CASE("LoopInvariantHoisting - all nodes invariant", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("lih_all_invariant");

    auto &body = graph.add_loop("loop", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    CHECK(modified);
    CHECK(pass.num_hoisted() == 2);
}

TEST_CASE("LoopInvariantHoisting - rank-3 BatchedGemm hoists", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern → BatchedGemm node that's invariant.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);
    auto D = create_random_tensor<double>("D", 3, 6, 4);

    cg::Graph graph("lih_rank3");

    auto &body = graph.add_loop("loop", 4, [](size_t iter) { return iter < 3; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ikb;kjb->ijb", &C, A, B); // invariant — A, B never change
        cg::scale(0.9, &D);                   // not invariant — writes D
    }

    // Confirm the einsum is actually captured as BatchedGemm inside the body.
    auto const *loop_desc = [&]() -> cg::LoopDescriptor const * {
        for (auto const &n : graph.nodes())
            if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
                return d;
        return nullptr;
    }();
    REQUIRE(loop_desc != nullptr);
    bool body_has_batched = false;
    for (auto const &n : loop_desc->body->nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            body_has_batched = true;
    REQUIRE(body_has_batched);

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    CHECK(modified);
    CHECK(pass.num_hoisted() == 1);
}
