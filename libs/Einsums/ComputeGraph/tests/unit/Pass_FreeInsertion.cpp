//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_FreeInsertion.cpp
/// @brief Tests for the FreeInsertion pass, including loop-aware behavior
///       : a body-resident intermediate
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

TEST_CASE("FreeInsertion - parent-declared scratch used only in a loop body frees after the loop", "[ComputeGraph][FreeInsertion][Loop]") {
    // Regression: Part A's local-only last_use saw only the Materialize /
    // Initialize writes of PARENT-declared, body-only-used scratch and freed
    // it BEFORE the loop (execute then failed validation: "still deferred").
    // Part B never covers these: the body-map handles of parent-declared
    // tensors carry neither is_intermediate nor release_fn, so its collect
    // filter skips them. Python intermediate=True scratch driving a captured
    // loop (the CCSD example) is exactly this shape.
    constexpr size_t n   = 6;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("parent_scratch_loop");
    auto     &W    = g.scratch<double, 2>("W", n, n);
    auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);   // W = A*A, recomputed per iteration
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A); // acc += W*A
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    g.apply(pm);

    size_t loop_pos = SIZE_MAX, free_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        if (g.nodes()[idx].kind == cg::OpKind::Loop) {
            loop_pos = idx;
        }
        if (g.nodes()[idx].kind == cg::OpKind::Free) {
            free_pos = idx;
        }
    }
    REQUIRE(loop_pos != SIZE_MAX);
    REQUIRE(free_pos != SIZE_MAX);
    REQUIRE(free_pos > loop_pos);

    REQUIRE_NOTHROW(g.execute());

    // Two iterations of acc += (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = 2.0 * s;
        }
    }
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }
}

TEST_CASE("FreeInsertion - Free is ordered after every reader under DataflowExecutor", "[ComputeGraph][FreeInsertion][Dataflow]") {
    // Regression: Free nodes used to list their tensor only as an INPUT, so
    // the dependency builder saw them as fellow READERS - unordered against
    // the real consumers. A concurrent executor could then release the
    // buffer (arena detach included) while a consumer einsum was still
    // reading it; the serial executor was safe only by node position. Free
    // now declares the tensor as an output too, so the WAR scan pins it
    // after every prior reader. The chain below makes t1 live across TWO
    // later readers, and replays under the DataflowExecutor must match the
    // eager reference exactly, every time.
    constexpr size_t n = 24;
    auto             A = create_random_tensor<double>("A", n, n);
    auto             C = create_zero_tensor<double>("C", n, n);

    Tensor<double, 2> t1_ref("t1_ref", n, n), t2_ref("t2_ref", n, n), C_ref("C_ref", n, n);
    einsum(0.0, Indices{i, j}, &t1_ref, 1.0, Indices{i, k}, A, Indices{k, j}, A);
    einsum(0.0, Indices{i, j}, &t2_ref, 1.0, Indices{i, k}, t1_ref, Indices{k, j}, A);
    einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, t2_ref, Indices{k, j}, t1_ref);

    cg::Graph g("free_dataflow");
    auto     &t1 = g.scratch<double, 2>("t1", n, n);
    auto     &t2 = g.scratch<double, 2>("t2", n, n);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &t1, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &t2, 1.0, t1, A);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, t2, t1); // t1's LAST reader, after t2's writer
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    pm.add<cg::passes::MemoryPlanning>();
    g.apply(pm);
    REQUIRE(count_nodes(g, cg::OpKind::Free) == 2);

    for (int rep = 0; rep < 10; rep++) {
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
            }
        }
    }
}

TEST_CASE("FreeInsertion - hoisted body Free is ordered against the loop under DataflowExecutor",
          "[ComputeGraph][FreeInsertion][Loop][Dataflow]") {
    // An eager intermediate CREATED INSIDE a loop body gets a single Free
    // hoisted to the parent after the Loop, plus a paired Materialize before
    // it (so replays re-allocate after the Free reclaims the buffer). Pre-fix
    // both carried no TensorIds (owns_tid=false) and floated as edgeless
    // roots: the DataflowExecutor could release body_tmp while the loop was
    // still executing. They must now carry the parent id so the Free is
    // ordered after the Loop and the Materialize before it.
    constexpr size_t n   = 24;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("body_eager_dataflow");
    auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &body_tmp = body.create_zero_tensor<double, 2>("body_tmp", n, n);
        cg::einsum("ik;kj->ij", 0.0, &body_tmp, 1.0, A, A);   // body_tmp = A*A
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, body_tmp, A); // acc += body_tmp*A
    }

    cg::passes::FreeInsertion fi(/*min_bytes=*/0);
    REQUIRE(fi.run(g));
    REQUIRE(count_nodes(g, cg::OpKind::Free) == 1);
    // The Free is paired with a Materialize so replays reallocate the buffer
    // the Free reclaimed.
    REQUIRE(count_nodes(g, cg::OpKind::Materialize) == 1);

    // The hoisted Free lands after the Loop and its paired Materialize before
    // it; both carry a parent TensorId (the Free as both input and output) so a
    // concurrent executor orders them against the loop instead of releasing the
    // buffer mid-execution or replaying into released storage.
    size_t loop_pos = SIZE_MAX, free_pos = SIZE_MAX, mat_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        auto const &node = g.nodes()[idx];
        if (node.kind == cg::OpKind::Loop) {
            loop_pos = idx;
        } else if (node.kind == cg::OpKind::Free) {
            free_pos = idx;
            CHECK_FALSE(node.inputs.empty());
            CHECK_FALSE(node.outputs.empty());
        } else if (node.kind == cg::OpKind::Materialize) {
            mat_pos = idx;
            CHECK_FALSE(node.outputs.empty());
        }
    }
    REQUIRE(loop_pos != SIZE_MAX);
    REQUIRE(free_pos != SIZE_MAX);
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(mat_pos < loop_pos);
    REQUIRE(free_pos > loop_pos);

    // Hand reference: two iterations of acc += (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = 2.0 * s;
        }
    }

    for (int rep = 0; rep < 20; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-10));
            }
        }
    }
}

TEST_CASE("FreeInsertion - parent scratch used only in an INNER nested loop frees after the OUTER loop",
          "[ComputeGraph][FreeInsertion][Loop]") {
    // Parent-declared scratch consumed only in a doubly-nested body. Part A's
    // subtree-aware last_use must see the inner-body reads through BOTH loop
    // levels and free W after the OUTER loop, while Materialize / Initialize
    // land before it. Freeing before the outer loop would leave W deferred on
    // entry; freeing inside either body would release-then-reuse each pass.
    constexpr size_t n   = 6;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("parent_scratch_nested");
    auto     &W     = g.scratch_zero<double, 2>("W", n, n);
    auto     &outer = g.add_loop("outer", 2, [](size_t it) { return it < 2; });
    auto     &inner = outer.add_loop("inner", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(inner);
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);   // W = A*A, recomputed per inner pass
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A); // acc += W*A
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    g.apply(pm);

    // At the parent level the only Loop node is the outer loop. Lifecycle
    // (Materialize / Initialize) before it, Free after it.
    size_t mat_pos = SIZE_MAX, init_pos = SIZE_MAX, loop_pos = SIZE_MAX, free_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        switch (g.nodes()[idx].kind) {
        case cg::OpKind::Materialize:
            mat_pos = idx;
            break;
        case cg::OpKind::Initialize:
            init_pos = idx;
            break;
        case cg::OpKind::Loop:
            loop_pos = idx;
            break;
        case cg::OpKind::Free:
            free_pos = idx;
            break;
        default:
            break;
        }
    }
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(init_pos != SIZE_MAX);
    REQUIRE(loop_pos != SIZE_MAX);
    REQUIRE(free_pos != SIZE_MAX);
    REQUIRE(mat_pos < loop_pos);
    REQUIRE(init_pos < loop_pos);
    REQUIRE(free_pos > loop_pos);

    // Hand reference: 2 outer x 2 inner = 4 passes of acc += (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = 4.0 * s;
        }
    }

    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }

    // Concurrent replays: Free must stay ordered after the outer loop so the
    // buffer is not reclaimed mid-execution, every rep.
    for (int rep = 0; rep < 15; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
            }
        }
    }
}

TEST_CASE("FreeInsertion - parent scratch used only in a conditional branch frees after the Conditional",
          "[ComputeGraph][FreeInsertion][ControlFlow]") {
    // Parent-declared scratch consumed only inside a conditional then-branch.
    // Part A's subtree-aware last_use must see the branch reads and free W
    // AFTER the Conditional node, with Materialize / Initialize before it.
    constexpr size_t n   = 6;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("parent_scratch_cond");
    auto     &W         = g.scratch_zero<double, 2>("W", n, n);
    bool      take_then = true;

    auto [then_g, else_g] = g.add_conditional("branch", [&]() { return take_then; });
    {
        cg::CaptureGuard const guard(then_g);
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);   // W = A*A
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A); // acc += W*A
    }
    // else_g left empty.

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    g.apply(pm);

    size_t mat_pos = SIZE_MAX, init_pos = SIZE_MAX, cond_pos = SIZE_MAX, free_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        switch (g.nodes()[idx].kind) {
        case cg::OpKind::Materialize:
            mat_pos = idx;
            break;
        case cg::OpKind::Initialize:
            init_pos = idx;
            break;
        case cg::OpKind::Conditional:
            cond_pos = idx;
            break;
        case cg::OpKind::Free:
            free_pos = idx;
            break;
        default:
            break;
        }
    }
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(init_pos != SIZE_MAX);
    REQUIRE(cond_pos != SIZE_MAX);
    REQUIRE(free_pos != SIZE_MAX);
    REQUIRE(mat_pos < cond_pos);
    REQUIRE(init_pos < cond_pos);
    REQUIRE(free_pos > cond_pos);

    // Hand reference: one application of acc = (A*A)*A (then-branch taken).
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = s;
        }
    }

    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }

    for (int rep = 0; rep < 10; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
            }
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
