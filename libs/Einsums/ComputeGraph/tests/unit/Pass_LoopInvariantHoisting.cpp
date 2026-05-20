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
    // C = A·B is invariant (A, B never written in the loop) and C is its only
    // writer, so it's safe to compute once before the loop; the body then
    // accumulates the invariant C into acc each iteration.
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto C   = create_zero_tensor<double>("C", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    cg::Graph graph("hoist_test");

    auto &body = graph.add_loop("loop", 5, [](size_t iter) { return iter < 4; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C, A, B); // invariant, single-writer of C
        cg::axpy(1.0, C, &acc);            // acc += C (reads C; self-modifying on acc)
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    REQUIRE(modified);
    REQUIRE(pass.num_hoisted() == 1); // the einsum

    cg::LoopDescriptor const *loop_desc = nullptr;
    for (auto const &node : graph.nodes()) {
        loop_desc = std::get_if<cg::LoopDescriptor>(&node.op_data);
        if (loop_desc)
            break;
    }
    REQUIRE(loop_desc != nullptr);
    REQUIRE(loop_desc->body->num_nodes() == 1); // only the axpy remains
}

TEST_CASE("LoopInvariantHoisting - does NOT hoist a producer whose output is overwritten in-place", "[ComputeGraph][Passes]") {
    // C = A·B (invariant inputs) but C is then scaled in place every
    // iteration. Hoisting the einsum out would remove the per-iteration
    // reset, so the scale would compound: C, 0.9C, 0.81C, … instead of
    // 0.9·(A·B) every iteration. The single-writer guard must refuse the
    // hoist, and the executed result must be 0.9·(A·B).
    constexpr size_t N = 5;
    auto             A = create_random_tensor<double>("A", 3, 3);
    auto             B = create_random_tensor<double>("B", 3, 3);
    auto             C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("no_hoist_overwrite");
    auto     &body = graph.add_loop("loop", N, [](size_t iter) { return iter + 1 < N; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C, A, B); // C = A·B
        cg::scale(0.9, &C);                // C *= 0.9  (second writer of C)
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();
    CHECK_FALSE(modified);
    CHECK(pass.num_hoisted() == 0); // C has two writers — not hoisted

    graph.execute();

    // Correct result: C = 0.9 * (A·B) (the reset runs every iteration).
    auto C_ref = create_zero_tensor<double>("C_ref", 3, 3);
    einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    for (size_t k = 0; k < C.size(); ++k) {
        CHECK(C.data()[k] == Catch::Approx(0.9 * C_ref.data()[k]));
    }
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

TEST_CASE("LoopInvariantHoisting - hoisted node with a deferred output stays materialized", "[ComputeGraph][Passes][Loop]") {
    // LIH runs before MaterializationPass. If it hoists an invariant einsum
    // whose output is a workspace (deferred) tensor, the full pipeline must
    // still place the Materialize before the hoisted node — otherwise the
    // hoisted einsum would write unallocated storage. Verify via the full
    // default pipeline + a correctness check.
    constexpr size_t N = 4;
    auto             A = create_random_tensor<double>("A", 3, 3); // eager, invariant
    auto             B = create_random_tensor<double>("B", 3, 3);

    cg::Workspace ws("ws");
    auto         &W   = ws.declare_zero_tensor<double, 2>("W", 3, 3); // deferred output
    auto         &acc = ws.declare_zero_tensor<double, 2>("acc", 3, 3);

    cg::Graph g("lih_deferred");
    auto     &body = g.add_loop("loop", N, [](size_t it) { return it + 1 < N; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &W, A, B); // W = A·B (invariant, single-writer) → hoistable
        cg::axpy(1.0, W, &acc);            // acc += W
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);
    ws.materialize_all();
    REQUIRE_NOTHROW(g.execute());

    // acc = N * (A·B).
    auto AB = create_zero_tensor<double>("AB", 3, 3);
    einsum(Indices{i, j}, &AB, Indices{i, k}, A, Indices{k, j}, B);
    for (size_t k = 0; k < acc.size(); ++k) {
        CHECK(acc.data()[k] == Catch::Approx(static_cast<double>(N) * AB.data()[k]));
    }
}
