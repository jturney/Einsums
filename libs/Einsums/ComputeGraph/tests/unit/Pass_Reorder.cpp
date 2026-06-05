//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_Reorder.cpp
/// @brief Unit tests for the Reorder (memory-aware topological sort) pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Reorder - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("reorder_empty");

    auto [modified, pass] = graph.apply<cg::passes::Reorder>();
    CHECK_FALSE(modified);
}

TEST_CASE("Reorder - single node", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("reorder_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::Reorder>();
    CHECK_FALSE(modified);
}

TEST_CASE("Reorder - produces valid topological order", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);
    auto D = create_zero_tensor<double>("D", 5, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 5, 5);
    auto D_ref = create_zero_tensor<double>("Dref", 5, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &D_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("reorder_indep");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    graph.apply<cg::passes::Reorder>();
    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
            CHECK(D(ii, jj) == Catch::Approx(D_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("Reorder - preserves data dependencies", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);
    auto D = create_zero_tensor<double>("D", 5, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 5, 5);
    auto D_ref = create_zero_tensor<double>("Dref", 5, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &D_ref, Indices{i, k}, C_ref, Indices{k, j}, B);

    cg::Graph graph("reorder_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, C, B);
    }

    graph.apply<cg::passes::Reorder>();
    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            CHECK(C(ii, jj) == Catch::Approx(C_ref(ii, jj)).margin(1e-10));
            CHECK(D(ii, jj) == Catch::Approx(D_ref(ii, jj)).margin(1e-10));
        }
    }
}

TEST_CASE("Reorder - memory-aware: frees large tensor early", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 128, 128);
    auto B = create_random_tensor<double>("B", 128, 128);
    auto C = create_zero_tensor<double>("C", 128, 128);
    auto D = create_random_tensor<double>("D", 4, 4);

    cg::Graph graph("reorder_memory");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &D);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    graph.apply<cg::passes::Reorder>();
    graph.execute();
}

TEST_CASE("Reorder - preserves rank-3 BatchedGemm chain (row-major)", "[ComputeGraph][Passes][HigherRank]") {
    // Row-major tensors + batch-prefix pattern → row_mode fast path. The
    // chain still needs to execute correctly after Reorder touches it.
    auto A = create_random_tensor<double>(true, "A", 4, 3, 3);
    auto B = create_random_tensor<double>(true, "B", 4, 3, 3);
    auto C = create_zero_tensor<double>(true, "C", 4, 3, 3);
    auto D = create_zero_tensor<double>(true, "D", 4, 3, 3);

    auto C_ref = create_zero_tensor<double>(true, "Cref", 4, 3, 3);
    auto D_ref = create_zero_tensor<double>(true, "Dref", 4, 3, 3);
    {
        using namespace einsums::index;
        tensor_algebra::einsum(Indices{b, i, j}, &C_ref, Indices{b, i, k}, A, Indices{b, k, j}, B);
        tensor_algebra::einsum(Indices{b, i, j}, &D_ref, Indices{b, i, k}, C_ref, Indices{b, k, j}, B);
    }

    cg::Graph graph("reorder_rank3_row");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bik;bkj->bij", &C, A, B);
        cg::einsum("bik;bkj->bij", &D, C, B);
    }

    size_t batched = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched;
    REQUIRE(batched == 2);

    graph.apply<cg::passes::Reorder>();
    graph.execute();

    for (size_t bb = 0; bb < 4; bb++) {
        for (size_t ii = 0; ii < 3; ii++) {
            for (size_t jj = 0; jj < 3; jj++) {
                CHECK(C(bb, ii, jj) == Catch::Approx(C_ref(bb, ii, jj)).margin(1e-10));
                CHECK(D(bb, ii, jj) == Catch::Approx(D_ref(bb, ii, jj)).margin(1e-10));
            }
        }
    }
}

TEST_CASE("Reorder - preserves rank-3 BatchedGemm dependency chain", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern so each stage becomes a BatchedGemm.
    // Square-ish shapes keep the chain trivially valid: (I=K=J=3, B=4).
    auto A = create_random_tensor<double>("A", 3, 3, 4);
    auto B = create_random_tensor<double>("B", 3, 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 3, 4);
    auto D = create_zero_tensor<double>("D", 3, 3, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 3, 3, 4);
    auto D_ref = create_zero_tensor<double>("Dref", 3, 3, 4);
    {
        using namespace einsums::index;
        tensor_algebra::einsum(Indices{i, j, b}, &C_ref, Indices{i, k, b}, A, Indices{k, j, b}, B);
        tensor_algebra::einsum(Indices{i, j, b}, &D_ref, Indices{i, k, b}, C_ref, Indices{k, j, b}, B);
    }

    cg::Graph graph("reorder_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, C, B);
    }

    // Verify both stages were captured as BatchedGemm.
    size_t batched = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched;
    REQUIRE(batched == 2);

    graph.apply<cg::passes::Reorder>();
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            for (size_t bb = 0; bb < 4; bb++) {
                CHECK(C(ii, jj, bb) == Catch::Approx(C_ref(ii, jj, bb)).margin(1e-10));
                CHECK(D(ii, jj, bb) == Catch::Approx(D_ref(ii, jj, bb)).margin(1e-10));
            }
        }
    }
}
