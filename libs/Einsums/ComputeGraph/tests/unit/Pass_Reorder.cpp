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
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
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
    reference_einsum("ij <- ik ; kj", &C_ref, A, B);
    reference_einsum("ij <- ik ; kj", &D_ref, A, B);

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
    reference_einsum("ij <- ik ; kj", &C_ref, A, B);
    reference_einsum("ij <- ik ; kj", &D_ref, C_ref, B);

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

TEST_CASE("Reorder - preserves rank-3 batched einsum chain (row-major)", "[ComputeGraph][Passes][HigherRank]") {
    // Row-major tensors + batch-prefix pattern → row_mode fast path. The
    // chain still needs to execute correctly after Reorder touches it.
    auto A = create_random_tensor<double>(true, "A", 4, 3, 3);
    auto B = create_random_tensor<double>(true, "B", 4, 3, 3);
    auto C = create_zero_tensor<double>(true, "C", 4, 3, 3);
    auto D = create_zero_tensor<double>(true, "D", 4, 3, 3);

    auto C_ref = create_zero_tensor<double>(true, "Cref", 4, 3, 3);
    auto D_ref = create_zero_tensor<double>(true, "Dref", 4, 3, 3);
    {
        reference_einsum("bij <- bik ; bkj", &C_ref, A, B);
        reference_einsum("bij <- bik ; bkj", &D_ref, C_ref, B);
    }

    cg::Graph graph("reorder_rank3_row");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("bik;bkj->bij", &C, A, B);
        cg::einsum("bik;bkj->bij", &D, C, B);
    }

    size_t batched = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::Einsum)
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

TEST_CASE("Reorder - preserves rank-3 batched einsum dependency chain", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern so each stage takes the strided-batch route.
    // Square-ish shapes keep the chain trivially valid: (I=K=J=3, B=4).
    auto A = create_random_tensor<double>("A", 3, 3, 4);
    auto B = create_random_tensor<double>("B", 3, 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 3, 4);
    auto D = create_zero_tensor<double>("D", 3, 3, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 3, 3, 4);
    auto D_ref = create_zero_tensor<double>("Dref", 3, 3, 4);
    {
        reference_einsum("ijb <- ikb ; kjb", &C_ref, A, B);
        reference_einsum("ijb <- ikb ; kjb", &D_ref, C_ref, B);
    }

    cg::Graph graph("reorder_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, C, B);
    }

    // Both stages are einsum nodes.
    size_t batched = 0;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::Einsum)
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

namespace {

/// Where in the node list the node of kind @p kind writing the tensor at @p ptr sits.
size_t position_of_writer(cg::Graph const &graph, void const *ptr, cg::OpKind kind = cg::OpKind::Einsum) {
    cg::TensorId const id = graph.find_tensor_id_by_ptr(ptr);
    REQUIRE(id != 0);
    auto const &nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].kind != kind) {
            continue;
        }
        for (auto const out : nodes[i].outputs) {
            if (out == id) {
                return i;
            }
        }
    }
    FAIL("no node of that kind writes the tensor");
    return nodes.size();
}

} // namespace

// Reorder used to score a node by every tensor it was the last user of, the caller's included, and
// to take equal scores in whatever order its heap returned them. Here the first producer and both
// allocations score the same, and the heap ran the whole second chain, its allocation first, ahead of
// the first. Now equal scores keep program order, and an allocation scores against the bytes it
// holds, so it waits until its producer is next.
TEST_CASE("Reorder - keeps program order among equal scores and holds an allocation back", "[ComputeGraph][Passes]") {
    constexpr size_t n  = 32;
    auto             A  = create_random_tensor<double>("A", n, n);
    auto             Z  = create_random_tensor<double>("Z", n, 4 * n);
    auto             Y  = create_random_tensor<double>("Y", 4 * n, n);
    auto             W  = create_random_tensor<double>("W", n, 2 * n); // last read by the second producer
    auto             V  = create_random_tensor<double>("V", 2 * n, n);
    auto             r1 = create_zero_tensor<double>("r1", n, n);
    auto             r2 = create_zero_tensor<double>("r2", n, n);
    auto             r3 = create_zero_tensor<double>("r3", n, n);

    cg::Graph   graph("reorder_ties");
    void const *x2 = nullptr;
    {
        cg::CaptureGuard const guard(graph);
        auto                  &X1 = graph.create_zero_tensor<double, 2>("X1", n, n);
        auto                  &X2 = graph.create_zero_tensor<double, 2>("X2", n, n);
        x2                        = &X2;
        cg::einsum("ij <- ik ; kj", 0.0, &X1, 1.0, Z, Y);
        cg::einsum("ij <- ik ; kj", 0.0, &r1, 1.0, X1, A);
        cg::einsum("ij <- ik ; kj", 0.0, &X2, 1.0, W, V);
        cg::einsum("ij <- ik ; kj", 0.0, &r2, 1.0, X2, A);
        cg::einsum("ij <- ik ; kj", 0.0, &r3, 1.0, Z, Y); // the first producer is not the last reader of Z and Y
    }

    graph.apply<cg::passes::Reorder>();
    size_t const first_consumer = position_of_writer(graph, &r1);
    CHECK(first_consumer < position_of_writer(graph, x2));
    CHECK(first_consumer < position_of_writer(graph, x2, cg::OpKind::Alloc));

    graph.execute();
    auto X1_ref = create_zero_tensor<double>("X1ref", n, n);
    auto X2_ref = create_zero_tensor<double>("X2ref", n, n);
    auto r1_ref = create_zero_tensor<double>("r1ref", n, n);
    auto r2_ref = create_zero_tensor<double>("r2ref", n, n);
    auto r3_ref = create_zero_tensor<double>("r3ref", n, n);
    reference_einsum("ij <- ik ; kj", &X1_ref, Z, Y);
    reference_einsum("ij <- ik ; kj", &r1_ref, X1_ref, A);
    reference_einsum("ij <- ik ; kj", &X2_ref, W, V);
    reference_einsum("ij <- ik ; kj", &r2_ref, X2_ref, A);
    reference_einsum("ij <- ik ; kj", &r3_ref, Z, Y);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            CHECK(r1(i, j) == Catch::Approx(r1_ref(i, j)).margin(1e-9));
            CHECK(r2(i, j) == Catch::Approx(r2_ref(i, j)).margin(1e-9));
            CHECK(r3(i, j) == Catch::Approx(r3_ref(i, j)).margin(1e-9));
        }
    }
}

// The case Reorder exists for, which the fixes above must not lose: both producers captured before
// both consumers hold two intermediates at once, and the reschedule runs each consumer right after
// its producer.
TEST_CASE("Reorder - runs a consumer before the next producer when that lowers the peak", "[ComputeGraph][Passes]") {
    constexpr size_t n  = 32;
    auto             A  = create_random_tensor<double>("A", n, n);
    auto             B  = create_random_tensor<double>("B", n, n);
    auto             r1 = create_zero_tensor<double>("r1", n, n);
    auto             r2 = create_zero_tensor<double>("r2", n, n);

    cg::Graph   graph("reorder_interleave");
    void const *x2 = nullptr;
    {
        cg::CaptureGuard const guard(graph);
        auto                  &X1 = graph.create_zero_tensor<double, 2>("X1", n, n);
        auto                  &X2 = graph.create_zero_tensor<double, 2>("X2", n, n);
        x2                        = &X2;
        cg::einsum("ij <- ik ; kj", 0.0, &X1, 1.0, A, B);
        cg::einsum("ij <- ik ; kj", 0.0, &X2, 1.0, B, A);
        cg::einsum("ij <- ik ; kj", 0.0, &r1, 1.0, X1, A);
        cg::einsum("ij <- ik ; kj", 0.0, &r2, 1.0, X2, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::Reorder>();
    CHECK(modified);
    CHECK(position_of_writer(graph, &r1) < position_of_writer(graph, x2));

    graph.execute();
    auto X1_ref = create_zero_tensor<double>("X1ref", n, n);
    auto X2_ref = create_zero_tensor<double>("X2ref", n, n);
    auto r1_ref = create_zero_tensor<double>("r1ref", n, n);
    auto r2_ref = create_zero_tensor<double>("r2ref", n, n);
    reference_einsum("ij <- ik ; kj", &X1_ref, A, B);
    reference_einsum("ij <- ik ; kj", &X2_ref, B, A);
    reference_einsum("ij <- ik ; kj", &r1_ref, X1_ref, A);
    reference_einsum("ij <- ik ; kj", &r2_ref, X2_ref, B);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            CHECK(r1(i, j) == Catch::Approx(r1_ref(i, j)).margin(1e-9));
            CHECK(r2(i, j) == Catch::Approx(r2_ref(i, j)).margin(1e-9));
        }
    }
}
