//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ChainParenthesization.cpp
/// @brief Unit tests for the ChainParenthesization analysis pass.
///
/// This pass reasons about matrix chain multiplication, which is an
/// inherently 2D concept, higher-rank coverage is not applicable here.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("ChainParenthesization - empty graph", "[ComputeGraph][Chain]") {
    cg::Graph graph("cp_empty");

    auto [modified, pass] = graph.apply<cg::passes::ChainParenthesization>();
    CHECK_FALSE(modified);
    CHECK(pass.original_flops() == 0);
    CHECK(pass.optimal_flops() == 0);
}

TEST_CASE("ChainParenthesization - non-GEMM operations", "[ComputeGraph][Chain]") {
    auto A = create_random_tensor<double>("A", 4, 4);

    cg::Graph graph("cp_non_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::ChainParenthesization>();
    CHECK_FALSE(modified);
    CHECK(pass.original_flops() == 0);
}

TEST_CASE("ChainParenthesization - detects chain and computes savings", "[ComputeGraph][Chain]") {
    // Chain: A(100x1) * B(1x100) * C(100x1)
    // Left-to-right: (A*B)*C has 100*1*100 + 100*100*1 = 30000 mults
    // Optimal: A*(B*C) has 1*100*1 + 100*1*1 = 200 mults
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 1);
    auto T1 = create_zero_tensor<double>("T1", 100, 100);
    auto T2 = create_zero_tensor<double>("T2", 100, 1);

    cg::Graph graph("chain_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T1, A, B);
        cg::einsum("ik;kj->ij", &T2, T1, C);
    }

    auto [modified, pass] = graph.apply<cg::passes::ChainParenthesization>();

    REQUIRE(pass.original_flops() > 0);
    REQUIRE(pass.optimal_flops() > 0);
    REQUIRE(pass.optimal_flops() < pass.original_flops());
}

TEST_CASE("ChainParenthesization - no chain for single gemm", "[ComputeGraph][Chain]") {
    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    cg::Graph graph("single_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::ChainParenthesization>();

    REQUIRE(pass.original_flops() == 0);
    REQUIRE(pass.optimal_flops() == 0);
}

// ── Loop-aware aggregation (analysis-aggregation group) ──────────────────

TEST_CASE("ChainParenthesization - detects a chain inside a loop body", "[ComputeGraph][Chain][Loop]") {
    // The GEMM chain lives entirely inside the loop body. A flat-graph-only
    // pass would see only the Loop node and report zero flops; the
    // aggregating pass must detect and cost the body's chain.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 1);
    auto T1 = create_zero_tensor<double>("T1", 100, 100);
    auto T2 = create_zero_tensor<double>("T2", 100, 1);

    cg::Graph g("chain_in_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &T1, A, B);
        cg::einsum("ik;kj->ij", &T2, T1, C);
    }

    auto [modified, pass] = g.apply<cg::passes::ChainParenthesization>();
    REQUIRE(pass.original_flops() > 0);
    REQUIRE(pass.optimal_flops() > 0);
    REQUIRE(pass.optimal_flops() < pass.original_flops());
}
