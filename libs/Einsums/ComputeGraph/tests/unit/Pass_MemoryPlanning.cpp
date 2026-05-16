//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_MemoryPlanning.cpp
/// @brief Unit tests for the MemoryPlanning analysis pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <sstream>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("MemoryPlanning - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("mp_empty");

    auto [modified, pass] = graph.apply<cg::passes::MemoryPlanning>();
    CHECK_FALSE(modified);
    CHECK(pass.total_memory() == 0);
    CHECK(pass.peak_memory() == 0);
}

TEST_CASE("MemoryPlanning - basic analysis", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 10, 10);
    auto B = create_random_tensor<double>("B", 10, 10);
    auto C = create_zero_tensor<double>("C", 10, 10);

    cg::Graph graph("memory_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    REQUIRE(pass.total_memory() == static_cast<long>(3 * 10 * 10) * sizeof(double));
    REQUIRE(pass.peak_memory() == static_cast<long>(3 * 10 * 10) * sizeof(double));

    std::ostringstream report;
    pass.print_report(report);
    REQUIRE(report.str().find("Total tensor memory") != std::string::npos);
}

TEST_CASE("MemoryPlanning - chain shows lower peak than total", "[ComputeGraph][Passes]") {
    auto A  = create_random_tensor<double>("A", 10, 10);
    auto B  = create_random_tensor<double>("B", 10, 10);
    auto T1 = create_zero_tensor<double>("T1", 10, 10);
    auto T2 = create_zero_tensor<double>("T2", 10, 10);

    cg::Graph graph("chain_memory");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T1, A, B);
        cg::einsum("ik;kj->ij", &T2, T1, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    REQUIRE(pass.total_memory() == static_cast<long>(4 * 10 * 10) * sizeof(double));
    REQUIRE(pass.peak_memory() < pass.total_memory());
}

TEST_CASE("MemoryPlanning - rank-3 BatchedGemm tensor liveness", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern so each einsum becomes a BatchedGemm.
    // Sizes: A(I=3,K=5,B=4), B(K=5,J=6,B=4), C(I=3,J=6,B=4), D(same as C).
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);
    auto D = create_zero_tensor<double>("D", 3, 6, 4);

    cg::Graph graph("mp_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
    size_t const expected_total = (size_t{3 * 5 * 4} + size_t{5 * 6 * 4} + 2 * size_t{3 * 6 * 4}) * sizeof(double);
    CHECK(pass.total_memory() == expected_total);
    CHECK(pass.peak_memory() > 0);
}
