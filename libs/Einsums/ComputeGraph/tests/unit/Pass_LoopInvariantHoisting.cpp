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
    CHECK(pass.num_hoisted() == 0); // C has two writers, not hoisted

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
        cg::einsum("ikb;kjb->ijb", &C, A, B); // invariant: A, B never change
        cg::scale(0.9, &D);                   // not invariant, writes D
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
    // still place the Materialize before the hoisted node, otherwise the
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

TEST_CASE("LoopInvariantHoisting - inner-loop invariant hoists all the way to parent", "[ComputeGraph][Passes][Loop][Nested]") {
    // parent → outer(2 iters) → inner(2 iters). The inner body computes
    // W = A·A (A external, invariant w.r.t. BOTH loops) and acc += W·B.
    // Innermost-first hoisting lifts W's producer inner→outer body on the
    // outer-body sweep, then outer-body→parent on the parent sweep (a single
    // run() call). The producer must end up in the PARENT graph; neither loop
    // body may still contain it. acc = 4·(A·A)·B (4 = 2 outer × 2 inner).
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto W   = create_zero_tensor<double>("W", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    cg::Graph graph("lih_nested");
    auto     &outer_body = graph.add_loop("outer", 2, [](size_t it) { return it + 1 < 2; });
    auto     &inner_body = outer_body.add_loop("inner", 2, [](size_t it) { return it + 1 < 2; });
    {
        cg::CaptureGuard const guard(inner_body);
        cg::einsum("ik;kj->ij", &W, A, A);             // W = A·A (invariant everywhere)
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, B); // acc += W·B (reads acc → self-modifying)
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    REQUIRE(modified);
    REQUIRE(pass.num_hoisted() == 2); // two composed single-level hoists: inner→outer, then outer→parent

    // Locate the loop descriptors after hoisting.
    cg::LoopDescriptor const *outer_desc = nullptr;
    for (auto const &n : graph.nodes())
        if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
            outer_desc = d;
    REQUIRE(outer_desc != nullptr);
    cg::LoopDescriptor const *inner_desc = nullptr;
    for (auto const &n : outer_desc->body->nodes())
        if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
            inner_desc = d;
    REQUIRE(inner_desc != nullptr);

    // The W producer landed at the parent; the bodies no longer contain it.
    CHECK(graph.num_nodes() == 2);             // W producer + outer loop
    CHECK(outer_desc->body->num_nodes() == 1); // just the inner loop
    CHECK(inner_desc->body->num_nodes() == 1); // just acc += W·B

    // Hand reference: acc = 4·(A·A)·B.
    auto AA = create_zero_tensor<double>("AA", 3, 3);
    einsum(Indices{i, j}, &AA, Indices{i, k}, A, Indices{k, j}, A);
    auto WB = create_zero_tensor<double>("WB", 3, 3);
    einsum(Indices{i, j}, &WB, Indices{i, k}, AA, Indices{k, j}, B);

    acc.zero();
    graph.execute();
    for (size_t idx = 0; idx < acc.size(); ++idx) {
        CHECK(acc.data()[idx] == Catch::Approx(4.0 * WB.data()[idx]));
    }

    // Repeated parallel execution must match the sequential reference.
    for (int rep = 0; rep < 10; ++rep) {
        acc.zero();
        cg::DataflowExecutor df;
        graph.execute(df);
        for (size_t idx = 0; idx < acc.size(); ++idx) {
            CHECK(acc.data()[idx] == Catch::Approx(4.0 * WB.data()[idx]));
        }
    }
}

TEST_CASE("LoopInvariantHoisting - invariant w.r.t. inner loop only hoists exactly one level", "[ComputeGraph][Passes][Loop][Nested]") {
    // parent → outer(2 iters) → inner(2 iters). The inner body computes
    // W = M·M and acc += W·B, where M is a graph-owned tensor that the OUTER
    // body rewrites (M *= 2) after the inner loop. W is invariant w.r.t. the
    // inner loop (M is not touched inside the inner body), so its producer
    // hoists inner→outer body. But M IS written in the outer body, so W is NOT
    // invariant w.r.t. the outer loop: the producer must stop at the outer body
    // and must NOT reach the parent.
    //
    // Hand reference (M0 = initial M, P = M0·M0):
    //   outer 0: W = M0·M0 = P;      acc += 2·(P·B);        M → 2·M0
    //   outer 1: W = (2M0)² = 4·P;   acc += 2·(4·P·B);      M → 4·M0
    //   acc = 10·(M0·M0)·B
    auto M   = create_random_tensor<double>("M", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto W   = create_zero_tensor<double>("W", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    // Reference computed BEFORE execution mutates M and acc.
    auto MM = create_zero_tensor<double>("MM", 3, 3);
    einsum(Indices{i, j}, &MM, Indices{i, k}, M, Indices{k, j}, M);
    auto MMB = create_zero_tensor<double>("MMB", 3, 3);
    einsum(Indices{i, j}, &MMB, Indices{i, k}, MM, Indices{k, j}, B);

    cg::Graph graph("lih_inner_only");
    auto     &outer_body = graph.add_loop("outer", 2, [](size_t it) { return it + 1 < 2; });
    auto     &inner_body = outer_body.add_loop("inner", 2, [](size_t it) { return it + 1 < 2; });
    {
        cg::CaptureGuard const guard(inner_body);
        cg::einsum("ik;kj->ij", &W, M, M);             // W = M·M (invariant w.r.t. inner only)
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, B); // acc += W·B
    }
    {
        cg::CaptureGuard const guard(outer_body);
        cg::scale(2.0, &M); // rewrites M in the outer body → blocks the outer-level hoist
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    REQUIRE(modified);
    REQUIRE(pass.num_hoisted() == 1); // exactly one hoist: inner → outer body

    cg::LoopDescriptor const *outer_desc = nullptr;
    for (auto const &n : graph.nodes())
        if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
            outer_desc = d;
    REQUIRE(outer_desc != nullptr);
    cg::LoopDescriptor const *inner_desc = nullptr;
    for (auto const &n : outer_desc->body->nodes())
        if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
            inner_desc = d;
    REQUIRE(inner_desc != nullptr);

    // W's producer stopped at the outer body; it did NOT reach the parent.
    CHECK(graph.num_nodes() == 1);             // parent holds only the outer loop
    CHECK(outer_desc->body->num_nodes() == 3); // W = M·M, inner loop, M *= 2
    CHECK(inner_desc->body->num_nodes() == 1); // only acc += W·B remains

    acc.zero();
    graph.execute();
    for (size_t idx = 0; idx < acc.size(); ++idx) {
        CHECK(acc.data()[idx] == Catch::Approx(10.0 * MMB.data()[idx]));
    }
}

TEST_CASE("LoopInvariantHoisting - does NOT hoist out of a conditional branch inside a loop", "[ComputeGraph][Passes][Loop][Conditional]") {
    // A computation that is loop-invariant but lives inside a conditional
    // branch inside the loop must NOT be hoisted. Recursion descends into Loop
    // bodies ONLY, never Conditional branches: lifting a node out of a branch
    // would execute it unconditionally and change semantics when the predicate
    // is false. The single-level driver also never treats a Conditional as a
    // hoist candidate, so the producer stays exactly where it was written.
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto W   = create_zero_tensor<double>("W", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    cg::Graph graph("lih_cond_in_loop");
    auto     &body        = graph.add_loop("loop", 2, [](size_t it) { return it + 1 < 2; });
    auto [then_g, else_g] = body.add_conditional("cond", [] { return true; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::einsum("ik;kj->ij", &W, A, A); // W = A·A: invariant, but inside the branch
        cg::axpy(1.0, W, &acc);            // acc += W
    }

    auto [modified, pass] = graph.apply<cg::passes::LoopInvariantHoisting>();

    CHECK_FALSE(modified);
    CHECK(pass.num_hoisted() == 0);

    // Nothing moved: the branch keeps both ops, the loop body keeps the
    // conditional, the parent keeps only the loop.
    cg::LoopDescriptor const *loop_desc = nullptr;
    for (auto const &n : graph.nodes())
        if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data))
            loop_desc = d;
    REQUIRE(loop_desc != nullptr);
    CHECK(graph.num_nodes() == 1);            // only the loop at parent
    CHECK(loop_desc->body->num_nodes() == 1); // only the conditional in the body
    CHECK(then_g.num_nodes() == 2);           // W = A·A and acc += W stay in the branch

    // Predicate is always true → acc = 2·(A·A) over the two loop iterations.
    auto AA = create_zero_tensor<double>("AA", 3, 3);
    einsum(Indices{i, j}, &AA, Indices{i, k}, A, Indices{k, j}, A);

    acc.zero();
    graph.execute();
    for (size_t idx = 0; idx < acc.size(); ++idx) {
        CHECK(acc.data()[idx] == Catch::Approx(2.0 * AA.data()[idx]));
    }
}
