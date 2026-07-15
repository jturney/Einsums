//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_FreeInsertion.cpp
/// @brief Tests for the FreeInsertion pass, including loop-aware behavior
///        (Step 3 of loop_handling_audit.md): a body-resident intermediate
///        must be freed once *after* the owning loop, never inside the body.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

size_t count_nodes(cg::Graph const &g, cg::OpKind kind) {
    size_t n = 0;
    for (auto const &node : g.nodes()) {
        if (node.kind == kind) {
            n++;
        }
    }
    return n;
}

} // namespace

TEST_CASE("FreeInsertion - frees a flat-graph intermediate after last use", "[ComputeGraph][FreeInsertion]") {
    auto A = create_random_tensor<double>("A", 8, 8);
    auto B = create_random_tensor<double>("B", 8, 8);
    auto C = create_zero_tensor<double>("C", 8, 8);

    cg::Graph g("flat");
    {
        cg::CaptureGuard const guard(g);
        // tmp is a graph-owned intermediate (is_intermediate = true).
        auto &tmp = g.create_zero_tensor<double, 2>("tmp", 8, 8);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, B);
    }

    // min_bytes = 0 so the small tmp qualifies.
    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    REQUIRE(fi.run(g));
    CHECK(fi.num_freed() == 1);
    CHECK(count_nodes(g, cg::OpKind::Free) == 1);

    REQUIRE_NOTHROW(g.execute());
}

TEST_CASE("FreeInsertion - is idempotent across runs", "[ComputeGraph][FreeInsertion]") {
    auto A = create_random_tensor<double>("A", 6, 6);
    auto C = create_zero_tensor<double>("C", 6, 6);

    cg::Graph g("idem");
    {
        cg::CaptureGuard const guard(g);
        auto                  &tmp = g.create_zero_tensor<double, 2>("tmp", 6, 6);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, A);
    }

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    REQUIRE(fi.run(g));
    size_t const after_first = count_nodes(g, cg::OpKind::Free);
    CHECK(after_first == 1);

    // Second run should not add another Free for the same tensor.
    fi.run(g);
    CHECK(count_nodes(g, cg::OpKind::Free) == after_first);
}

TEST_CASE("FreeInsertion - hoists a body intermediate to after the loop", "[ComputeGraph][FreeInsertion][Loop]") {
    // A body-created intermediate is live across every iteration. Its Free
    // must land in the parent, *after* the Loop node, never inside the
    // body (which would free-then-reuse each iteration).
    auto A = create_random_tensor<double>("A", 8, 8);
    auto C = create_zero_tensor<double>("C", 8, 8);

    cg::Graph g("loop_free");
    auto     &body = g.add_loop("iter", /*max_iterations=*/1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &tmp = body.create_zero_tensor<double, 2>("body_tmp", 8, 8);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, A);
    }

    // The body should have no Free before the pass, and crucially none after.
    REQUIRE(count_nodes(body, cg::OpKind::Free) == 0);

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    REQUIRE(fi.run(g));
    CHECK(fi.num_freed() == 1);

    // Free landed in the PARENT, not the body.
    CHECK(count_nodes(g, cg::OpKind::Free) == 1);
    CHECK(count_nodes(body, cg::OpKind::Free) == 0);

    // And it sits immediately after the Loop node.
    auto const &nodes           = g.nodes();
    bool        free_after_loop = false;
    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        if (nodes[i].kind == cg::OpKind::Loop && nodes[i + 1].kind == cg::OpKind::Free) {
            free_after_loop = true;
            break;
        }
    }
    CHECK(free_after_loop);

    REQUIRE_NOTHROW(g.execute());
}

TEST_CASE("FreeInsertion - hoists a deeply-nested body intermediate past every loop", "[ComputeGraph][FreeInsertion][Loop]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);

    cg::Graph g("nested_free");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        auto                  &tmp = inner.create_zero_tensor<double, 2>("deep_tmp", 5, 5);
        cg::einsum("ik;kj->ij", 0.0, &tmp, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, tmp, A);
    }

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    REQUIRE(fi.run(g));

    // Free hoisted all the way to the outermost parent; neither body holds one.
    CHECK(count_nodes(g, cg::OpKind::Free) == 1);
    CHECK(count_nodes(outer, cg::OpKind::Free) == 0);
    CHECK(count_nodes(inner, cg::OpKind::Free) == 0);
}

TEST_CASE("FreeInsertion - eager create_tensor intermediate survives replay", "[ComputeGraph][FreeInsertion][Replay]") {
    // Eager graph-owned intermediates (create_tensor) DO carry a release_fn
    // via make_handle, so once they cross the pass's min-bytes threshold a
    // real Free is inserted - but unlike deferred tensors they had no paired
    // Materialize, so the SECOND execute() touched released storage. Every
    // prior test tensor sat under the 1MB threshold, hiding this. 400x400
    // doubles = 1.28MB.
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("fi_eager_replay");
    auto     &tmp = graph.create_tensor<double, 2>("tmp", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);   // writes tmp
        cg::einsum("ik;kj->ij", &out, tmp, B); // last (only) read of tmp
    }

    auto [modified, pass] = graph.apply<cg::passes::FreeInsertion>();
    REQUIRE(modified); // 1.28MB > 1MB threshold
    REQUIRE(count_nodes(graph, cg::OpKind::Free) == 1);
    // The Free must come with a paired Materialize so replays re-allocate.
    REQUIRE(count_nodes(graph, cg::OpKind::Materialize) == 1);

    graph.execute();
    auto const OUT_first = Tensor<double, 2>(out);

    out.zero();
    graph.execute(); // replay: tmp must be re-materialized, not dangling

    for (size_t ii = 0; ii < N; ii += 37) {
        for (size_t jj = 0; jj < N; jj += 41) {
            REQUIRE(std::abs(out(ii, jj) - OUT_first(ii, jj)) < 1e-10);
        }
    }
}

TEST_CASE("FreeInsertion - leaves workspace tensors alive (not freed)", "[ComputeGraph][FreeInsertion][Loop]") {
    // Workspace tensors are is_intermediate = false: they persist across
    // pipelines, so FreeInsertion must never free them even when used
    // inside a loop body.
    auto B = create_random_tensor<double>("B", 6, 6);

    cg::Workspace ws("ws");
    auto         &persist = ws.declare_zero_tensor<double, 2>("persist", 6, 6);

    cg::Graph g("ws_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;jk->ij", 0.0, &persist, 1.0, B, B);
    }

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    bool const                modified = fi.run(g);
    // No freeable intermediates → pass is a no-op.
    CHECK_FALSE(modified);
    CHECK(count_nodes(g, cg::OpKind::Free) == 0);
    CHECK(count_nodes(body, cg::OpKind::Free) == 0);
}
