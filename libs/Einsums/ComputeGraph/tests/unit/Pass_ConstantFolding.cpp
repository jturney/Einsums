//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ConstantFolding.cpp
/// @brief Unit tests for the ConstantFolding optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("ConstantFolding - user-owned tensors are not assumed constant", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cf_user_owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::ConstantFolding>();

    CHECK_FALSE(modified);
    CHECK(pass.num_folded() == 0);
}

TEST_CASE("ConstantFolding - written intermediate is not constant", "[ComputeGraph][Passes]") {
    cg::Graph graph("cf_intermediate");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);
    for (size_t ii = 0; ii < 3; ii++)
        T(ii, ii) = 1.0;

    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &T);
    }

    // T is written by scale, so it's not constant.
    auto [modified, pass] = graph.apply<cg::passes::ConstantFolding>();
    CHECK_FALSE(modified);
}

TEST_CASE("ConstantFolding - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("cf_empty");

    auto [modified, pass] = graph.apply<cg::passes::ConstantFolding>();

    CHECK_FALSE(modified);
    CHECK(pass.num_folded() == 0);
}

TEST_CASE("ConstantFolding - skips control flow nodes", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("cf_loop");
    auto     &body = graph.add_loop("loop", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::ConstantFolding>();
    CHECK(pass.num_folded() == 0);
}

TEST_CASE("ConstantFolding - safe with Pipeline loop body", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    for (int iter = 0; iter < 3; iter++) {
        tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
        linear_algebra::scale(0.9, &C_ref);
    }

    cg::Pipeline pipeline("cf_pipeline");
    {
        auto                  &loop = pipeline.add_loop("iter", 3, [](size_t it) { return it < 2; });
        cg::CaptureGuard const guard(loop);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::scale(0.9, &C);
    }

    auto pm = cg::PassManager::create_default();
    pipeline.apply(pm);
    pipeline.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("ConstantFolding - rank-3 user-owned tensors are not folded", "[ComputeGraph][Passes][HigherRank]") {
    // Strided-batched pattern (col-major default): batch axis at the LAST
    // position so capture takes the BatchedGemm fast path. Shapes: A(I,K,B),
    // B(K,J,B), C(I,J,B). All inputs user-owned → nothing folds.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);

    cg::Graph graph("cf_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
    }

    REQUIRE(graph.nodes()[0].kind == cg::OpKind::BatchedGemm);

    auto [modified, pass] = graph.apply<cg::passes::ConstantFolding>();

    CHECK_FALSE(modified);
    CHECK(pass.num_folded() == 0);
}
