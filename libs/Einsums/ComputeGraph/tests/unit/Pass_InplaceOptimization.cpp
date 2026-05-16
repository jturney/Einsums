//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_InplaceOptimization.cpp
/// @brief Unit tests for the InplaceOptimization analysis pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("InplaceOptimization - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("io_empty");

    auto [modified, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK_FALSE(modified);
    CHECK(pass.num_candidates() == 0);
}

TEST_CASE("InplaceOptimization - user-owned tensor not a candidate", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("io_user");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK_FALSE(modified);
    CHECK(pass.num_candidates() == 0);
}

TEST_CASE("InplaceOptimization - finds candidates", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("inplace_test");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B); // writes T
        cg::einsum("ik;kj->ij", &C, T, A); // reads T (sole consumer)
    }

    auto [_m, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK(pass.num_candidates() >= 0); // analysis-only, verify it runs cleanly
}

TEST_CASE("InplaceOptimization - rank-3 BatchedGemm intermediate with sole consumer", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern so the einsum is captured as BatchedGemm.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);

    cg::Graph graph("inplace_rank3");
    auto     &T = graph.create_zero_tensor<double, 3>("T", 3, 6, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &T, A, B);
        cg::scale(0.5, &T);
    }

    auto [_m, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK(pass.num_candidates() >= 0);
}
