//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_CSE.cpp
/// @brief Unit tests for Common Subexpression Elimination.

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

TEST_CASE("CSE - empty graph", "[ComputeGraph][CSE]") {
    cg::Graph graph("cse_empty");

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
}

TEST_CASE("CSE - single node graph", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cse_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 1);
}

TEST_CASE("CSE - eliminates duplicate einsum", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);

    cg::Graph graph("cse_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("CSE - surviving consumer of an eliminated duplicate reads the survivor", "[ComputeGraph][CSE]") {
    // Regression for the CSE soundness bug where folding a duplicate producer
    // corrupted a *downstream* consumer of that duplicate. Executor lambdas
    // resolve operands through their captured TensorSlot, not Node::inputs, so
    // CSE's TensorId metadata redirect alone is invisible at run time: a node
    // reading the eliminated duplicate's output kept reading its (now
    // never-written) buffer and silently produced zeros. The earlier tests only
    // check the survivor's value or the node count, so they missed this.
    //
    // Diamond shape: C and D are identical products (D is eliminated); OUT reads
    // D. After CSE, D's producer is gone and OUT must resolve to C's buffer via
    // Graph::redirect_slot.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 5);
    auto F   = create_random_tensor<double>("F", 5, 2);
    auto C   = create_zero_tensor<double>("C", 4, 5);
    auto D   = create_zero_tensor<double>("D", 4, 5);
    auto OUT = create_zero_tensor<double>("OUT", 4, 2);

    cg::Graph graph("cse_surviving_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);   // survivor
        cg::einsum("ik;kj->ij", &D, A, B);   // duplicate (eliminated)
        cg::einsum("ik;kj->ij", &OUT, D, F); // consumer of the eliminated duplicate
    }

    REQUIRE(graph.num_nodes() == 3);

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 2); // D's producer folded away

    graph.execute();

    // Reference: OUT = (A·B)·F
    auto AB = create_zero_tensor<double>("AB", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    auto OUT_ref = create_zero_tensor<double>("OUTref", 4, 2);
    tensor_algebra::einsum(Indices{i, j}, &OUT_ref, Indices{i, k}, AB, Indices{k, j}, F);

    double max_abs = 0.0;
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 2; jj++) {
            max_abs = std::max(max_abs, std::abs(OUT(ii, jj)));
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-12);
        }
    }
    // Guard against the failure mode being masked by an all-zero reference.
    REQUIRE(max_abs > 1e-10);
}

TEST_CASE("CSE - three identical einsums reduces to one", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);
    auto E = create_zero_tensor<double>("E", 4, 5);

    cg::Graph graph("cse_triple");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
        cg::einsum("ik;kj->ij", &E, A, B);
    }

    REQUIRE(graph.num_nodes() == 3);

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    CHECK(graph.num_nodes() == 1);
}

TEST_CASE("CSE - does not eliminate different operations", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("cse_no_match");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 2.0, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("CSE - does not eliminate different inputs", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("cse_diff_inputs");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, B, A); // swapped
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("CSE - does not merge scale with different factors", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("cse_diff_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::CSE>();
    CHECK_FALSE(modified);
    CHECK(graph.num_nodes() == 2);
}

TEST_CASE("CSE + DeadNodeElimination composition", "[ComputeGraph][CSE]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("cse_dne");
    auto     &D = graph.create_zero_tensor<double, 2>("D", 4, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }

    size_t const n_before = graph.num_nodes();
    REQUIRE(n_before >= 2);

    graph.apply<cg::passes::CSE>();
    CHECK(graph.num_nodes() < n_before);

    // DNE may or may not find further dead nodes depending on Alloc handling;
    // just verify it runs without crashing.
    auto [modified, dne] = graph.apply<cg::passes::DeadNodeElimination>();
    (void)modified;
}

TEST_CASE("CSE - deduplicates rank-3 BatchedGemm nodes (col-major)", "[ComputeGraph][CSE][HigherRank]") {
    // Two identical rank-3 strided-batched contractions → each captures as
    // OpKind::BatchedGemm. Exercises the batched_gemm_desc_equal path in
    // CSE::op_data_equal (added to handle BatchedGemm comparison).
    // Col-major default + batch-suffix pattern triggers the fast path's col_mode.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);
    auto D = create_zero_tensor<double>("D", 3, 6, 4);

    cg::Graph graph("cse_rank3_col");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, A, B);
    }

    size_t batched_before = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_before;
    REQUIRE(batched_before == 2);

    size_t const nodes_before = graph.num_nodes();

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    CHECK(graph.num_nodes() < nodes_before);
    size_t batched_after = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_after;
    CHECK(batched_after == 1);
}

TEST_CASE("CSE - deduplicates rank-3 BatchedGemm nodes (row-major)", "[ComputeGraph][CSE][HigherRank]") {
    // Same contraction, row-major tensors + batch-prefix pattern → triggers
    // the fast path's row_mode branch. Verifies CSE works across both layout
    // modes of the strided-batched capture.
    auto A = create_random_tensor<double>(/*row_major=*/true, "A", 4, 3, 5);
    auto B = create_random_tensor<double>(/*row_major=*/true, "B", 4, 5, 6);
    auto C = create_zero_tensor<double>(/*row_major=*/true, "C", 4, 3, 6);
    auto D = create_zero_tensor<double>(/*row_major=*/true, "D", 4, 3, 6);

    cg::Graph graph("cse_rank3_row");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bik;bkj->bij", &C, A, B);
        cg::einsum("bik;bkj->bij", &D, A, B);
    }

    size_t batched_before = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_before;
    REQUIRE(batched_before == 2);

    auto [modified, pass] = graph.apply<cg::passes::CSE>();

    CHECK(modified);
    size_t batched_after = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::BatchedGemm)
            ++batched_after;
    CHECK(batched_after == 1);
}
