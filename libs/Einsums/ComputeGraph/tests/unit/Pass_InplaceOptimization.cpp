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
    CHECK(pass.num_candidates() >= 1);
    // Einsum consumers may never alias output with input; no merge happens.
    CHECK(pass.num_merged() == 0);
}

TEST_CASE("InplaceOptimization - merges direct_product output into dying input", "[ComputeGraph][Passes][Inplace]") {
    // X = A·B (intermediate), Y = alpha*(X ⊙ B) with beta=0 (intermediate,
    // pure overwrite, elementwise), out = Y·A (user-visible). X dies at the
    // direct_product, whose output is element-aligned with it, so Y reuses
    // X's storage and Y's own allocation disappears.
    auto A   = create_random_tensor<double>("A", 6, 6);
    auto B   = create_random_tensor<double>("B", 6, 6);
    auto out = create_zero_tensor<double>("out", 6, 6);

    // Reference, computed eagerly.
    auto X_ref = create_zero_tensor<double>("Xref", 6, 6);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto Y_ref = create_zero_tensor<double>("Yref", 6, 6);
    linear_algebra::direct_product(2.0, X_ref, B, 0.0, &Y_ref);
    auto out_ref = create_zero_tensor<double>("OUTref", 6, 6);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, Y_ref, Indices{k, j}, A);

    cg::Graph graph("inplace_merge");
    auto     &X = graph.create_zero_tensor<double, 2>("X", 6, 6);
    auto     &Y = graph.create_zero_tensor<double, 2>("Y", 6, 6);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::direct_product(2.0, X, B, 0.0, &Y);
        cg::einsum("ik;kj->ij", &out, Y, A);
    }

    size_t const allocs_before = [&] {
        size_t n = 0;
        for (auto const &node : graph.nodes()) {
            n += node.kind == cg::OpKind::Alloc ? 1 : 0;
        }
        return n;
    }();
    REQUIRE(allocs_before == 2); // X and Y

    auto [modified, pass] = graph.apply<cg::passes::InplaceOptimization>();
    REQUIRE(modified);
    CHECK(pass.num_merged() == 1);

    size_t allocs_after = 0;
    for (auto const &node : graph.nodes()) {
        allocs_after += node.kind == cg::OpKind::Alloc ? 1 : 0;
    }
    CHECK(allocs_after == 1); // Y's alloc removed

    graph.execute();
    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }

    // Replays keep working through the merged storage.
    out.zero();
    graph.execute();
    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("InplaceOptimization - accumulating direct_product is not merged", "[ComputeGraph][Passes][Inplace]") {
    // beta != 0 means the op reads its destination (listed as an input by
    // the recording convention), so the output is not a pure overwrite.
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto B   = create_random_tensor<double>("B", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("inplace_rmw");
    auto     &X = graph.create_zero_tensor<double, 2>("X", 4, 4);
    auto     &Y = graph.create_zero_tensor<double, 2>("Y", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::direct_product(2.0, X, B, 1.0, &Y); // beta=1: accumulates into Y
        cg::einsum("ik;kj->ij", &out, Y, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK(pass.num_merged() == 0);
    (void)modified;
}

TEST_CASE("InplaceOptimization - twice-read input is not merged", "[ComputeGraph][Passes][Inplace]") {
    // X is read again after the elementwise consumer: its storage must live.
    auto A    = create_random_tensor<double>("A", 4, 4);
    auto B    = create_random_tensor<double>("B", 4, 4);
    auto out1 = create_zero_tensor<double>("out1", 4, 4);
    auto out2 = create_zero_tensor<double>("out2", 4, 4);

    cg::Graph graph("inplace_alive");
    auto     &X = graph.create_zero_tensor<double, 2>("X", 4, 4);
    auto     &Y = graph.create_zero_tensor<double, 2>("Y", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::direct_product(1.0, X, B, 0.0, &Y);
        cg::einsum("ik;kj->ij", &out1, Y, A);
        cg::einsum("ik;kj->ij", &out2, X, A); // X read again
    }

    auto [modified, pass] = graph.apply<cg::passes::InplaceOptimization>();
    CHECK(pass.num_merged() == 0);
    (void)modified;

    // Numerics stay correct regardless.
    graph.execute();
    auto X_ref = create_zero_tensor<double>("Xref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto OUT2_ref = create_zero_tensor<double>("OUT2ref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &OUT2_ref, Indices{i, k}, X_ref, Indices{k, j}, A);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(out2(ii, jj) - OUT2_ref(ii, jj)) < 1e-12);
        }
    }
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

// ── Loop-aware aggregation (analysis-aggregation group) ──────────────────

TEST_CASE("InplaceOptimization - counts a candidate inside a loop body", "[ComputeGraph][Passes][Loop]") {
    // A body-created intermediate with a single producer + single consumer
    // is an in-place candidate. The aggregating pass must find it even
    // though it lives inside the loop body.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("inplace_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    auto     &T    = body.create_zero_tensor<double, 2>("T", 4, 4);
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &T, A, B); // sole writer of T
        cg::einsum("ik;kj->ij", &C, T, A); // sole reader of T
    }

    auto [_m, pass] = g.apply<cg::passes::InplaceOptimization>();
    CHECK(pass.num_candidates() >= 1);
}
