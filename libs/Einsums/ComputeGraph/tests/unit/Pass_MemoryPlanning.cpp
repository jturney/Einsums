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
    // Disjoint lifetimes -> both at offset 0: the arena is ONE buffer.
    CHECK(mp.planned_arena_bytes() == ((kBuf + 63) / 64) * 64);

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
