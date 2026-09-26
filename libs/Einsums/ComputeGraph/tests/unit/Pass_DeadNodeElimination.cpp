//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_DeadNodeElimination.cpp
/// @brief Unit tests for the DeadNodeElimination optimization pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <string>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("DeadNodeElimination - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("dne_empty");

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - keeps nodes with user-owned outputs", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("dne_user_owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - eliminates intermediate with no reader", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("dne_intermediate");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
    }

    size_t const nodes_before = graph.num_nodes();
    REQUIRE(nodes_before >= 1);

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK(modified);
    CHECK(pass.num_eliminated() >= 1);
    CHECK(graph.num_nodes() < nodes_before);
}

TEST_CASE("DeadNodeElimination - keeps intermediate with reader", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("dne_live_intermediate");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
        cg::einsum("ik;kj->ij", &C, T, A);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
}

TEST_CASE("DeadNodeElimination - rank-3 batched einsum intermediate is eliminated", "[ComputeGraph][Passes][HigherRank]") {
    // Strided-batched einsum whose output nobody reads.
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);

    cg::Graph graph("dne_rank3");
    auto     &T = graph.create_zero_tensor<double, 3>("T", 3, 6, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &T, A, B);
    }

    // The contraction is one einsum node.
    bool has_batched = false;
    for (auto const &n : graph.nodes())
        if (n.kind == cg::OpKind::Einsum)
            has_batched = true;
    REQUIRE(has_batched);

    size_t const nodes_before = graph.num_nodes();
    REQUIRE(nodes_before >= 1);

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK(modified);
    CHECK(pass.num_eliminated() >= 1);
}

// ── Loop-aware behavior (need-care groundwork) ───────────────────────────

TEST_CASE("DeadNodeElimination - eliminates a dead intermediate inside a loop body", "[ComputeGraph][Passes][Loop]") {
    // A body intermediate written but never read should be eliminated when
    // DNE recurses into the body (via PassManager).
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("dne_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &dead = body.create_zero_tensor<double, 2>("dead", 4, 4);
        cg::einsum("ik;kj->ij", 0.0, &dead, 1.0, A, A); // written, never read
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, A);    // real work
    }

    size_t const body_nodes_before = body.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    bool const modified = pm.run(g);

    CHECK(modified);
    CHECK(body.num_nodes() == body_nodes_before - 1); // the dead einsum is gone
}

TEST_CASE("DeadNodeElimination - keeps a producer feeding only a nested loop", "[ComputeGraph][Passes][Loop]") {
    // THE groundwork case: `shared` is produced in the outer body but read
    // ONLY by a node inside a nested loop. A Loop node doesn't list its
    // body's reads as inputs, so without collect_subtree_referenced_ptrs
    // DNE would wrongly eliminate `shared`'s producer. It must be kept.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("dne_nested");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });

    // Produce `shared` in the outer body. It's an outer-body intermediate.
    auto &shared = outer.create_zero_tensor<double, 2>("shared", 4, 4);
    {
        cg::CaptureGuard const guard(outer);
        cg::einsum("ik;kj->ij", 0.0, &shared, 1.0, A, A); // sole producer of `shared`
    }
    // Nested loop whose body reads `shared` (its only consumer).
    auto &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, shared, A); // reads `shared`
    }

    size_t const outer_nodes_before = outer.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    pm.run(g);

    // The producer of `shared` must survive, it feeds the nested loop.
    CHECK(outer.num_nodes() == outer_nodes_before);
    bool found_shared_producer = false;
    for (auto const &nd : outer.nodes()) {
        for (auto tid : nd.outputs) {
            if (outer.tensor(tid).tensor_ptr == static_cast<void *>(&shared)) {
                found_shared_producer = true;
            }
        }
    }
    CHECK(found_shared_producer);
}

// ── Cross-graph liveness: a body tensor read by an ENCLOSING or SIBLING graph ──
// A tensor produced inside a control-flow child is dead only if nothing outside
// that child reads it. DNE descends itself carrying the enclosing references, so
// a body producer feeding a parent-after-the-loop node, a sibling loop body, or
// a parent-after-the-conditional node is never eliminated.

TEST_CASE("DeadNodeElimination - keeps a body intermediate consumed by the parent after the loop", "[ComputeGraph][Passes][Loop]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph g("dne_body_to_parent");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    auto     &tmp  = body.create_zero_tensor<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A); // tmp = A*A, its only reader is the parent below
    }
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, tmp, A); // out = tmp*A, in the PARENT after the loop
    }

    size_t const body_before = body.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    pm.run(g);

    // The body producer of `tmp` must survive: an outside consumer reads it.
    CHECK(body.num_nodes() == body_before);

    g.execute();

    auto AA = create_zero_tensor<double>("AA", 4, 4);
    reference_einsum("ij <- ik ; kj", &AA, A, A);
    auto ref = create_zero_tensor<double>("ref", 4, 4);
    reference_einsum("ij <- ik ; kj", &ref, AA, A);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            CHECK(std::abs(out(i, j) - ref(i, j)) < 1e-10);
        }
    }
}

TEST_CASE("DeadNodeElimination - keeps a body intermediate consumed by a sibling loop body", "[ComputeGraph][Passes][Loop]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto acc = create_zero_tensor<double>("acc", 4, 4);

    cg::Graph g("dne_sibling_loops");
    auto     &body1 = g.add_loop("loop1", 1, [](size_t) { return false; });
    auto     &tmp   = body1.create_zero_tensor<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(body1);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A); // tmp = A*A, read only by the sibling loop
    }
    auto &body2 = g.add_loop("loop2", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body2);
        cg::einsum("ik;kj->ij", 0.0, &acc, 1.0, tmp, A); // acc = tmp*A, in a SIBLING loop body
    }

    size_t const body1_before = body1.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    pm.run(g);

    // The `tmp` producer in loop1 must survive: loop2's body reads it.
    CHECK(body1.num_nodes() == body1_before);

    g.execute();

    auto AA = create_zero_tensor<double>("AA", 4, 4);
    reference_einsum("ij <- ik ; kj", &AA, A, A);
    auto ref = create_zero_tensor<double>("ref", 4, 4);
    reference_einsum("ij <- ik ; kj", &ref, AA, A);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            CHECK(std::abs(acc(i, j) - ref(i, j)) < 1e-10);
        }
    }
}

TEST_CASE("DeadNodeElimination - keeps a then-branch intermediate consumed by the parent", "[ComputeGraph][Passes][Conditional]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph g("dne_then_to_parent");
    auto [then_g, else_g] = g.add_conditional("cond", [] { return true; });
    auto &tmp             = then_g.create_zero_tensor<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(then_g);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A); // tmp = A*A in the then-branch
    }
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, tmp, A); // parent reads tmp after the conditional
    }

    size_t const then_before = then_g.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    pm.run(g);

    // The then-branch producer of `tmp` must survive: the parent reads it.
    CHECK(then_g.num_nodes() == then_before);

    g.execute();

    auto AA = create_zero_tensor<double>("AA", 4, 4);
    reference_einsum("ij <- ik ; kj", &AA, A, A);
    auto ref = create_zero_tensor<double>("ref", 4, 4);
    reference_einsum("ij <- ik ; kj", &ref, AA, A);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            CHECK(std::abs(out(i, j) - ref(i, j)) < 1e-10);
        }
    }
}

TEST_CASE("DeadNodeElimination - names the tensors it pruned", "[ComputeGraph][Passes]") {
    // The pruning here is the intended kind, but it is indistinguishable from the case that
    // cost a bisect to find: a caller creates a result with the scratch-defaulted creator,
    // optimizes, and reads zeros because the only node writing it was removed. The pass cannot
    // tell those apart, so it names what it dropped and says what to do about it. A report that
    // carried only a node count is what made that failure silent.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("dne_names_pruned");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    REQUIRE(modified);
    REQUIRE(pass.num_eliminated() >= 1);

    // The dropped tensor is named, not merely counted.
    auto const &pruned = pass.pruned_tensors();
    REQUIRE_FALSE(pruned.empty());
    CHECK(std::ranges::find(pruned, std::string{"T"}) != pruned.end());

    // And the explanation carries the name plus the remedy, so a reader who expected 'T' to
    // survive learns why it did not without reading the pass source.
    auto const  lines = pass.explain();
    std::string joined;
    for (auto const &line : lines) {
        joined += line + "\n";
    }
    CHECK(joined.find("'T'") != std::string::npos);
    CHECK(joined.find("intermediate=false") != std::string::npos);
}

TEST_CASE("DeadNodeElimination - a declared result survives the pass", "[ComputeGraph][Passes]") {
    // The other half of the contract above: declare_tensor marks a graph-owned tensor as a
    // result rather than scratch, and the pass must leave its producer alone even though
    // nothing inside the graph reads it. This is the spelling the explain() text recommends,
    // so it has to actually work.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    cg::Graph graph("dne_declared_result");
    auto     &C = graph.declare_zero_tensor<double, 2>("C", 3, 3);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::DeadNodeElimination>();

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
    CHECK(pass.pruned_tensors().empty());
}

// Defends: a Materialize is kept while a surviving node writes its tensor. GEMMBatching folds the
// dead product and the live one into one batched call, which keeps writing the dead product's
// buffer; DeadNodeElimination used to judge that buffer's Materialize by "nothing reads it" alone,
// removed it, and the batch then wrote through a deferred shell, so execute() threw. The default
// order runs this pass before batching, which is why only a reordered pipeline reached it.
TEST_CASE("DeadNodeElimination - keeps the Materialize a surviving batched GEMM writes through", "[ComputeGraph][Passes][GEMMBatching]") {
    auto                    A     = create_random_tensor<double>("A", 2, 3);
    auto                    B     = create_random_tensor<double>("B", 3, 3);
    auto                    R     = create_random_tensor<double>("R", 2, 3);
    Tensor<double, 2> const B_old = B;
    Tensor<double, 2> const R_old = R;

    cg::Graph graph("dne_batched_materialize");
    auto     &dead = graph.scratch<double, 2>("dead", 2, 3);
    auto     &live = graph.create_zero_tensor<double, 2>("live", 2, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.5, &B);
        cg::einsum("ik;kj->ij", &dead, A, B);
        cg::einsum("ik;kj->ij", &live, A, B);
        cg::axpy(1.0, live, &R);
    }

    cg::PassManager pm;
    pm.add<cg::passes::GEMMBatching>();
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::DeadNodeElimination>();
    graph.apply(pm);
    INFO(pm.explain());
    // Not vacuous: the two products must really share one batched call.
    REQUIRE(std::ranges::any_of(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::BatchedGemm; }));

    REQUIRE_NOTHROW(graph.execute());

    auto B_half = create_zero_tensor<double>("B_half", 3, 3);
    auto AB     = create_zero_tensor<double>("AB", 2, 3);
    einsums::testing::reference_permute("ij <- ij", 0.0, &B_half, 0.5, B_old);
    reference_einsum("ij <- ik ; kj", 0.0, &AB, 1.0, A, B_half);
    auto R_ref = R_old;
    einsums::testing::reference_permute("ij <- ij", 1.0, &R_ref, 1.0, AB);
    for (size_t i = 0; i < R.size(); i++) {
        INFO("element " << i);
        CHECK(std::abs(R.data()[i] - R_ref.data()[i]) < 1e-12);
    }
}
