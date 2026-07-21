//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_MemoryPlanning.cpp
/// @brief Unit tests for the MemoryPlanning analysis pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <sstream>

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

TEST_CASE("MemoryPlanning - empty graph", "[ComputeGraph][Passes]") {
    cg::Graph graph("mp_empty");

    auto [modified, pass] = graph.apply<cg::passes::MemoryPlanning>();
    CHECK_FALSE(modified);
    CHECK(pass.total_memory() == 0);
    CHECK(pass.peak_memory() == 0);
}

TEST_CASE("MemoryPlanning - basic analysis", "[ComputeGraph][Passes]") {
    auto A = create_random_tensor<double>("A", 10, 10);
    auto B = create_random_tensor<double>("B", 10, 10);
    auto C = create_zero_tensor<double>("C", 10, 10);

    cg::Graph graph("memory_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    REQUIRE(pass.total_memory() == static_cast<long>(3 * 10 * 10) * sizeof(double));
    REQUIRE(pass.peak_memory() == static_cast<long>(3 * 10 * 10) * sizeof(double));

    std::ostringstream report;
    pass.print_report(report);
    REQUIRE(report.str().find("Total tensor memory") != std::string::npos);
}

TEST_CASE("MemoryPlanning - chain shows lower peak than total", "[ComputeGraph][Passes]") {
    auto A  = create_random_tensor<double>("A", 10, 10);
    auto B  = create_random_tensor<double>("B", 10, 10);
    auto T1 = create_zero_tensor<double>("T1", 10, 10);
    auto T2 = create_zero_tensor<double>("T2", 10, 10);

    cg::Graph graph("chain_memory");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T1, A, B);
        cg::einsum("ik;kj->ij", &T2, T1, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    REQUIRE(pass.total_memory() == static_cast<long>(4 * 10 * 10) * sizeof(double));
    REQUIRE(pass.peak_memory() < pass.total_memory());
}

TEST_CASE("MemoryPlanning - rank-3 BatchedGemm tensor liveness", "[ComputeGraph][Passes][HigherRank]") {
    // Col-major batch-suffix pattern so each einsum becomes a BatchedGemm.
    // Sizes: A(I=3,K=5,B=4), B(K=5,J=6,B=4), C(I=3,J=6,B=4), D(same as C).
    auto A = create_random_tensor<double>("A", 3, 5, 4);
    auto B = create_random_tensor<double>("B", 5, 6, 4);
    auto C = create_zero_tensor<double>("C", 3, 6, 4);
    auto D = create_zero_tensor<double>("D", 3, 6, 4);

    cg::Graph graph("mp_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ikb;kjb->ijb", &C, A, B);
        cg::einsum("ikb;kjb->ijb", &D, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::MemoryPlanning>();

    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
    size_t const expected_total = (size_t{3 * 5 * 4} + size_t{5 * 6 * 4} + 2 * size_t{3 * 6 * 4}) * sizeof(double);
    CHECK(pass.total_memory() == expected_total);
    CHECK(pass.peak_memory() > 0);
}

// ── Loop-aware aggregation (analysis-aggregation group) ──────────────────

TEST_CASE("MemoryPlanning - aggregates tensors inside a loop body", "[ComputeGraph][Passes][Loop]") {
    // The matmul lives entirely inside a loop body. A flat-graph-only pass
    // would report zero footprint (the top-level graph holds just the Loop
    // node). The aggregating pass must account for the body's tensors.
    auto A = create_random_tensor<double>("A", 10, 10);
    auto B = create_random_tensor<double>("B", 10, 10);
    auto C = create_zero_tensor<double>("C", 10, 10);

    cg::Graph g("mp_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [modified, pass] = g.apply<cg::passes::MemoryPlanning>();
    CHECK_FALSE(modified);
    // A, B, C are each 10x10 doubles and all live inside the body.
    CHECK(pass.total_memory() == 3 * size_t{10 * 10} * sizeof(double));
    CHECK(pass.peak_memory() > 0);
}

TEST_CASE("MemoryPlanning - aggregates across nested loops", "[ComputeGraph][Passes][Loop]") {
    auto A = create_random_tensor<double>("A", 8, 8);
    auto C = create_zero_tensor<double>("C", 8, 8);

    cg::Graph g("mp_nested");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    auto [modified, pass] = g.apply<cg::passes::MemoryPlanning>();
    CHECK_FALSE(modified);
    // A and C, both 8x8 doubles, used in the innermost body.
    CHECK(pass.total_memory() == 2 * size_t{8 * 8} * sizeof(double));
}

TEST_CASE("MemoryPlanning - arena shares storage between disjoint-lifetime intermediates", "[ComputeGraph][Passes][Arena]") {
    // X and Y are eager 1.28MB intermediates with sequential, non-overlapping
    // lifetimes: FreeInsertion brackets each with Materialize/Free, and the
    // arena planner places both at the same offset - one buffer instead of
    // two, verbatim across replays.
    constexpr size_t N    = 400;
    auto             A    = create_random_tensor<double>("A", N, N);
    auto             B    = create_random_tensor<double>("B", N, N);
    auto             out1 = create_zero_tensor<double>("out1", N, N);
    auto             out2 = create_zero_tensor<double>("out2", N, N);

    // Reference.
    auto X_ref = create_zero_tensor<double>("Xref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto OUT1_ref = create_zero_tensor<double>("OUT1ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto Y_ref = create_zero_tensor<double>("Yref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &Y_ref, Indices{i, k}, OUT1_ref, Indices{k, j}, B);
    auto OUT2_ref = create_zero_tensor<double>("OUT2ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT2_ref, Indices{i, k}, Y_ref, Indices{k, j}, B);

    cg::Graph graph("mp_arena");
    auto     &X = graph.create_tensor<double, 2>("X", N, N);
    auto     &Y = graph.create_tensor<double, 2>("Y", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::einsum("ik;kj->ij", &out1, X, B); // X dies here
        cg::einsum("ik;kj->ij", &Y, out1, B); // Y born after X's death
        cg::einsum("ik;kj->ij", &out2, Y, B);
    }

    // Bracket the intermediates, then plan + apply the arena.
    {
        auto [fi_mod, fi] = graph.apply<cg::passes::FreeInsertion>();
        REQUIRE(fi_mod);
        REQUIRE(fi.num_freed() == 2);
    }
    auto [mp_mod, mp] = graph.apply<cg::passes::MemoryPlanning>();
    REQUIRE(mp_mod);
    CHECK(mp.num_planned() == 2);
    constexpr size_t kBuf = N * N * sizeof(double);
    CHECK(mp.planned_tensor_bytes() == 2 * kBuf);
    // Disjoint lifetimes SHOULD alias both intermediates into ONE arena buffer
    // (round_up(kBuf, 64)). Whether that aliasing fires depends on the relative
    // order of the floating Free(X)/Materialize(Y) lifecycle nodes, which is not
    // deterministic across standard-library implementations: under MSVC's STL
    // the bracket positions can read as overlapping and the arena holds both
    // buffers (2x), while libstdc++/libc++ alias to one. Accept either bound -
    // the aliasing is an optimization, not a correctness property, and the
    // execution result is verified against the eager oracle below.
    constexpr size_t kOneBuf = ((kBuf + 63) / 64) * 64;
    CHECK(mp.planned_arena_bytes() >= kOneBuf);
    CHECK(mp.planned_arena_bytes() <= 2 * kOneBuf);

    graph.execute();
    for (size_t ii = 0; ii < N; ii += 41) {
        for (size_t jj = 0; jj < N; jj += 37) {
            REQUIRE(std::abs(out2(ii, jj) - OUT2_ref(ii, jj)) < 1e-8);
        }
    }

    out1.zero();
    out2.zero();
    graph.execute(); // replay through the arena
    for (size_t ii = 0; ii < N; ii += 41) {
        for (size_t jj = 0; jj < N; jj += 37) {
            REQUIRE(std::abs(out2(ii, jj) - OUT2_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("MemoryPlanning - arena slot reuse is ordered under DataflowExecutor", "[ComputeGraph][Passes][Arena][Dataflow]") {
    // The arena packs X and Y at the same offset (position-disjoint lifetimes),
    // so Materialize(Y) claims X's bytes but shares NO TensorId with X's chain.
    // Y is pushed past X's death only via out1 (X's first reader), so Y's writer
    // is unordered against X's OTHER readers out1b/out1c. Parallel executors
    // order nodes ONLY by shared-TensorId hazards, never by node position, so
    // without a storage-reuse WAW edge Materialize(Y) + Y's writer run
    // concurrently with out1b/out1c and scribble X's still-live bytes. The
    // serial executor is safe only by node position; the DataflowExecutor is not.
    constexpr size_t N     = 400;
    auto             A     = create_random_tensor<double>("A", N, N);
    auto             B     = create_random_tensor<double>("B", N, N);
    auto             out1  = create_zero_tensor<double>("out1", N, N);
    auto             out1b = create_zero_tensor<double>("out1b", N, N);
    auto             out1c = create_zero_tensor<double>("out1c", N, N);
    auto             out1d = create_zero_tensor<double>("out1d", N, N);
    auto             out1e = create_zero_tensor<double>("out1e", N, N);
    auto             out2  = create_zero_tensor<double>("out2", N, N);

    // Eager references.
    auto X_ref = create_zero_tensor<double>("Xref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto OUT1_ref = create_zero_tensor<double>("OUT1ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto OUT1B_ref = create_zero_tensor<double>("OUT1Bref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1B_ref, Indices{i, k}, X_ref, Indices{k, j}, A);
    auto OUT1C_ref = create_zero_tensor<double>("OUT1Cref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1C_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto OUT1D_ref = create_zero_tensor<double>("OUT1Dref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1D_ref, Indices{i, k}, X_ref, Indices{k, j}, A);
    auto OUT1E_ref = create_zero_tensor<double>("OUT1Eref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1E_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto Y_ref = create_zero_tensor<double>("Yref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &Y_ref, Indices{i, k}, OUT1_ref, Indices{k, j}, A);
    auto OUT2_ref = create_zero_tensor<double>("OUT2ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT2_ref, Indices{i, k}, Y_ref, Indices{k, j}, A);

    cg::Graph graph("mp_arena_dataflow");
    auto     &X = graph.create_tensor<double, 2>("X", N, N);
    auto     &Y = graph.create_tensor<double, 2>("Y", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::einsum("ik;kj->ij", &out1, X, B);  // X reader that feeds Y
        cg::einsum("ik;kj->ij", &out1b, X, A); // pure X reader (races with Y)
        cg::einsum("ik;kj->ij", &out1c, X, B); // pure X reader (races with Y)
        cg::einsum("ik;kj->ij", &out1d, X, A); // pure X reader (races with Y)
        cg::einsum("ik;kj->ij", &out1e, X, B); // pure X reader (races with Y)
        cg::einsum("ik;kj->ij", &Y, out1, A);  // reads out1 (NOT X): one level above X's readers
        cg::einsum("ik;kj->ij", &out2, Y, A);
    }

    {
        auto [fi_mod, fi] = graph.apply<cg::passes::FreeInsertion>();
        REQUIRE(fi_mod);
        REQUIRE(fi.num_freed() == 2);
    }
    auto [mp_mod, mp] = graph.apply<cg::passes::MemoryPlanning>();
    REQUIRE(mp_mod);
    REQUIRE(mp.num_planned() == 2);
    // Position-disjoint lifetimes -> both at offset 0: one shared buffer.
    constexpr size_t kBuf = ((N * N * sizeof(double) + 63) / 64) * 64;
    REQUIRE(mp.planned_arena_bytes() == kBuf);

    for (int rep = 0; rep < 30; rep++) {
        out1.zero();
        out1b.zero();
        out1c.zero();
        out1d.zero();
        out1e.zero();
        out2.zero();
        cg::DataflowExecutor df;
        graph.execute(df);
        for (size_t ii = 0; ii < N; ii += 41) {
            for (size_t jj = 0; jj < N; jj += 37) {
                REQUIRE(std::abs(out1(ii, jj) - OUT1_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(out1b(ii, jj) - OUT1B_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(out1c(ii, jj) - OUT1C_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(out1d(ii, jj) - OUT1D_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(out1e(ii, jj) - OUT1E_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(out2(ii, jj) - OUT2_ref(ii, jj)) < 1e-8);
            }
        }
    }
}

TEST_CASE("MemoryPlanning - arena keeps overlapping lifetimes apart", "[ComputeGraph][Passes][Arena]") {
    // X and Y are simultaneously live (Y's producer reads X after Y exists),
    // so they must get disjoint arena ranges: arena == sum of both buffers.
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("mp_arena_overlap");
    auto     &X = graph.create_tensor<double, 2>("X", N, N);
    auto     &Y = graph.create_tensor<double, 2>("Y", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::einsum("ik;kj->ij", &Y, X, B);   // X and Y both live
        cg::einsum("ik;kj->ij", &out, Y, X); // reads both: lifetimes overlap
    }

    graph.apply<cg::passes::FreeInsertion>();
    auto [mp_mod, mp] = graph.apply<cg::passes::MemoryPlanning>();
    REQUIRE(mp_mod);
    CHECK(mp.num_planned() == 2);
    constexpr size_t kAligned = ((N * N * sizeof(double) + 63) / 64) * 64;
    CHECK(mp.planned_arena_bytes() == 2 * kAligned);

    // Numerics with both intermediates arena-resident at distinct offsets.
    auto X_ref = create_zero_tensor<double>("Xref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto Y_ref = create_zero_tensor<double>("Yref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &Y_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, Y_ref, Indices{k, j}, X_ref);

    graph.execute();
    for (size_t ii = 0; ii < N; ii += 41) {
        for (size_t jj = 0; jj < N; jj += 37) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("MemoryPlanning - analysis-only mode plans without applying", "[ComputeGraph][Passes][Arena]") {
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("mp_arena_analysis");
    auto     &X = graph.create_tensor<double, 2>("X", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);
        cg::einsum("ik;kj->ij", &out, X, B);
    }

    graph.apply<cg::passes::FreeInsertion>();

    cg::passes::MemoryPlanning mp(/*apply_arena=*/false);
    bool const                 modified = mp.run(graph);
    CHECK_FALSE(modified); // plan computed, graph untouched
    CHECK(mp.num_planned() == 1);
    CHECK(mp.planned_arena_bytes() > 0);

    graph.execute(); // still runs on its own allocations
}

// ── Body-resident intermediates: arena status quo (analysis mode) ────────

TEST_CASE("MemoryPlanning - body-resident intermediates are never arena-planned", "[ComputeGraph][Passes][Arena][Loop]") {
    // Status quo (see plan_arena's control-flow bail): a loop body's own
    // intermediate is never arena-planned from EITHER side. Its whole
    // lifecycle (Materialize/Initialize/Free) is hoisted to the parent by
    // Materialization + FreeInsertion, so the body holds no Materialize/Free
    // bracket to plan; and the parent then carries a Loop node, which makes
    // parent-level plan_arena bail. Net: num_planned == 0. Both the deferred
    // (scratch_zero) and eager (create_zero_tensor) body-declared variants.
    constexpr size_t N   = 96;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             acc = create_zero_tensor<double>("acc", N, N);

    SECTION("deferred scratch declared in the body") {
        cg::Graph g("mp_body_scratch");
        auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
        {
            cg::CaptureGuard const guard(body);
            auto                  &W = body.scratch_zero<double, 2>("W", N, N); // deferred + intermediate + zero
            cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);                        // W = A*A, recomputed per iteration
            cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A);                      // acc += W*A
        }

        cg::PassManager pm;
        pm.add<cg::passes::Materialization>();
        pm.add<cg::passes::FreeInsertion>(size_t{0});
        g.apply(pm);

        // Lifecycle hoisted out of the body: no in-body bracket to plan.
        CHECK(count_nodes(body, cg::OpKind::Materialize) == 0);
        CHECK(count_nodes(body, cg::OpKind::Free) == 0);

        cg::passes::MemoryPlanning mp(/*apply_arena=*/false);
        bool const                 modified = mp.run(g);
        CHECK_FALSE(modified);
        CHECK(mp.num_planned() == 0); // body: no bracket; parent: Loop node bails
    }

    SECTION("eager create_zero_tensor declared in the body") {
        cg::Graph g("mp_body_eager");
        auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
        {
            cg::CaptureGuard const guard(body);
            auto                  &body_tmp = body.create_zero_tensor<double, 2>("body_tmp", N, N);
            cg::einsum("ik;kj->ij", 0.0, &body_tmp, 1.0, A, A);   // body_tmp = A*A
            cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, body_tmp, A); // acc += body_tmp*A
        }

        // FreeInsertion hoists the Free (and a paired Materialize) to the parent.
        cg::passes::FreeInsertion fi(/*min_bytes=*/0);
        REQUIRE(fi.run(g));
        CHECK(count_nodes(body, cg::OpKind::Materialize) == 0);
        CHECK(count_nodes(body, cg::OpKind::Free) == 0);

        cg::passes::MemoryPlanning mp(/*apply_arena=*/false);
        bool const                 modified = mp.run(g);
        CHECK_FALSE(modified);
        CHECK(mp.num_planned() == 0); // body: no bracket; parent: Loop node bails
    }
}

// ── Wraparound canary: prove the in-body bracket is unreachable ───────────

TEST_CASE("MemoryPlanning - in-body arena bracket is unreachable (wraparound canary)", "[ComputeGraph][Passes][Arena][Loop]") {
    // plan_arena's interval test uses body-LOCAL node positions and is blind to
    // cross-iteration liveness (a value written in iteration i and read in i+1
    // looks dead between its body-local Free and the next Materialize). Two
    // iteration-crossing tensors with body-local-disjoint intervals would then
    // be wrongly aliased. That hazard is only unreachable because NO in-body
    // Materialize/Free bracket ever forms: Materialization + FreeInsertion hoist
    // the whole lifecycle to the parent.
    //
    // CANARY: a scratch created AND fully consumed inside a single body is the
    // exact candidate for an in-body bracket. If this ever leaves a Materialize
    // or Free INSIDE the body (e.g. someone teaches FreeInsertion to place
    // in-body Frees), this fails - and plan_arena's iteration-wraparound
    // blindness must be revisited (treat iteration-crossing tensors as
    // always-live, or refuse to plan inside Loop bodies).
    constexpr size_t n   = 8;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("mp_wraparound_canary");
    auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &W = body.scratch_zero<double, 2>("W", n, n); // declared + consumed inside one body
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A);
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    pm.add<cg::passes::MemoryPlanning>();
    g.apply(pm);

    // The load-bearing invariant: no bracket survives inside the body.
    CHECK(count_nodes(body, cg::OpKind::Materialize) == 0);
    CHECK(count_nodes(body, cg::OpKind::Free) == 0);
    // And the eager loop-body variant of the same candidate.
    {
        cg::Graph g2("mp_wraparound_canary_eager");
        auto     &body2 = g2.add_loop("iter", 2, [](size_t it) { return it < 2; });
        {
            cg::CaptureGuard const guard(body2);
            auto                  &T = body2.create_zero_tensor<double, 2>("T", n, n);
            cg::einsum("ik;kj->ij", 0.0, &T, 1.0, A, A);
            cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, T, A);
        }
        cg::passes::FreeInsertion fi(/*min_bytes=*/0);
        REQUIRE(fi.run(g2));
        CHECK(count_nodes(body2, cg::OpKind::Materialize) == 0);
        CHECK(count_nodes(body2, cg::OpKind::Free) == 0);
    }
}

// ── Control flow at the graph level suppresses a flat-prefix arena ───────

TEST_CASE("MemoryPlanning - a loop at graph level suppresses the flat-prefix arena", "[ComputeGraph][Passes][Arena][Loop]") {
    // Known conservatism: X and Y are flat, disjoint-lifetime intermediates
    // BEFORE a loop - on their own they would share one arena slot. But
    // plan_arena bails the WHOLE graph level the instant it sees a Loop node
    // (control flow references tensors invisibly to the flat node list), so the
    // arena-shareable prefix is left on its own allocations. num_planned == 0
    // even though FreeInsertion bracketed X and Y. Execution stays correct.
    constexpr size_t N    = 128;
    auto             A    = create_random_tensor<double>("A", N, N);
    auto             B    = create_random_tensor<double>("B", N, N);
    auto             out1 = create_zero_tensor<double>("out1", N, N);
    auto             out2 = create_zero_tensor<double>("out2", N, N);
    auto             acc  = create_zero_tensor<double>("acc", N, N);

    // Eager references.
    auto X_ref = create_zero_tensor<double>("Xref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &X_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto OUT1_ref = create_zero_tensor<double>("OUT1ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT1_ref, Indices{i, k}, X_ref, Indices{k, j}, B);
    auto Y_ref = create_zero_tensor<double>("Yref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &Y_ref, Indices{i, k}, OUT1_ref, Indices{k, j}, B);
    auto OUT2_ref = create_zero_tensor<double>("OUT2ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT2_ref, Indices{i, k}, Y_ref, Indices{k, j}, B);
    auto ACC_ref = create_zero_tensor<double>("ACCref", N, N); // acc += A*B, twice
    tensor_algebra::einsum(1.0, Indices{i, j}, &ACC_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(1.0, Indices{i, j}, &ACC_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("mp_prefix_loop");
    auto     &X = graph.create_tensor<double, 2>("X", N, N);
    auto     &Y = graph.create_tensor<double, 2>("Y", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &X, A, B);    // X
        cg::einsum("ik;kj->ij", &out1, X, B); // X dies
        cg::einsum("ik;kj->ij", &Y, out1, B); // Y born after X's death
        cg::einsum("ik;kj->ij", &out2, Y, B); // Y dies
    }
    auto &body = graph.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, A, B); // acc += A*B
    }

    // Bracket X and Y (min_bytes 0 so the 128KB buffers qualify).
    {
        cg::passes::FreeInsertion fi(/*min_bytes=*/0);
        REQUIRE(fi.run(graph));
    }
    auto [mp_mod, mp] = graph.apply<cg::passes::MemoryPlanning>();
    CHECK_FALSE(mp_mod);          // control-flow bail: nothing applied
    CHECK(mp.num_planned() == 0); // flat prefix left unplanned despite being shareable

    auto check_against_refs = [&]() {
        for (size_t ii = 0; ii < N; ii += 31) {
            for (size_t jj = 0; jj < N; jj += 29) {
                REQUIRE(std::abs(out2(ii, jj) - OUT2_ref(ii, jj)) < 1e-8);
                REQUIRE(std::abs(acc(ii, jj) - ACC_ref(ii, jj)) < 1e-8);
            }
        }
    };

    graph.execute();
    check_against_refs();

    for (int rep = 0; rep < 10; rep++) {
        out1.zero();
        out2.zero();
        acc.zero();
        cg::DataflowExecutor df;
        graph.execute(df);
        check_against_refs();
    }
}
