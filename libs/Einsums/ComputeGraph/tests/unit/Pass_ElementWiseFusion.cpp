//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ElementWiseFusion.cpp
/// @brief Unit tests for the ElementWiseFusion optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("ElementWiseFusion - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("ewf_empty");

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    CHECK_FALSE(modified);
    CHECK(ewf.num_fused() == 0);
}

TEST_CASE("ElementWiseFusion - single node", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("ewf_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    CHECK_FALSE(modified);
}

TEST_CASE("ElementWiseFusion - fuses consecutive scales", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);
    linear_algebra::scale(3.0, &A_ref);

    cg::Graph graph("ewf_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(ewf.num_fused() == 1);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(A(ii, jj) - A_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("ElementWiseFusion - three consecutive scales fuse to one", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);
    linear_algebra::scale(3.0, &A_ref);
    linear_algebra::scale(4.0, &A_ref);

    cg::Graph graph("ewf_triple");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
        cg::scale(4.0, &A);
    }

    REQUIRE(graph.num_nodes() == 3);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();

    CHECK(modified);
    CHECK(ewf.num_fused() == 2);
    CHECK(graph.num_nodes() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            CHECK(A(ii, jj) == Catch::Approx(A_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("ElementWiseFusion - no fusion for different tensors", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("ewf_no_fuse");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &B);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE_FALSE(modified);
}

TEST_CASE("ElementWiseFusion - scale separated by einsum does not fuse", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("ewf_barrier");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::einsum("ik;kj->ij", 0.0, &A, 1.0, A, B); // NOLINT
        cg::scale(3.0, &A);
    }

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();

    CHECK_FALSE(modified);
    CHECK(ewf.num_fused() == 0);
    CHECK(graph.num_nodes() == 3);
}

TEST_CASE("ElementWiseFusion - fuses consecutive rank-3 scales", "[ComputeGraph][Passes][HigherRank]") {
    auto A = create_random_tensor<double>("A", 4, 3, 5);

    auto A_ref = Tensor<double, 3>(A);
    linear_algebra::scale(2.0, &A_ref);
    linear_algebra::scale(3.0, &A_ref);

    cg::Graph graph("ewf_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(ewf.num_fused() == 1);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();

    for (size_t bb = 0; bb < 4; bb++) {
        for (size_t ii = 0; ii < 3; ii++) {
            for (size_t jj = 0; jj < 5; jj++) {
                REQUIRE(std::abs(A(bb, ii, jj) - A_ref(bb, ii, jj)) < 1e-12);
            }
        }
    }
}

TEST_CASE("ElementWiseFusion - fused output feeding a downstream consumer stays correct", "[ComputeGraph][Passes]") {
    // Consumer-bearing topology (bug-1012 class): the fused node's POSITION
    // matters because a later gemm reads the scaled tensor. The fused
    // replacement must stay ahead of that reader.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_random_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(2.0, &C_ref);
    linear_algebra::scale(3.0, &C_ref);

    cg::Graph graph("ewf_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
        cg::scale(3.0, &C);
        cg::gemm<false, false>(1.0, C, A, 0.0, &D); // consumer of the fused scales
    }
    REQUIRE(graph.num_nodes() == 3);

    auto [modified, ewf] = graph.apply<cg::passes::ElementWiseFusion>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 2);

    graph.execute();

    Tensor<double, 2> D_ref("D_ref", 4, 4);
    D_ref.zero();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t kk = 0; kk < 4; kk++) {
                D_ref(ii, jj) += C_ref(ii, kk) * A(kk, jj);
            }
        }
    }
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-11);
        }
    }
}
