//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ContractionPlanning.cpp
/// @brief Dedicated unit coverage for the ContractionPlanning pass. The cost
///        model and residency behavior live in HardwareProfile.cpp; this file
///        pins the graph-restructuring contract: which chains get folded into a
///        cheaper parenthesization, which stay analysis-only, and that a folded
///        chain stays numerically identical to the eager reference. The
///        headline case is the leaf-orientation gate: a chain whose interior
///        leaf is captured TRANSPOSED must not be restructured, because the
///        rank-2 fold reads leaves with gemm<false, false> on their physical
///        layout and would silently corrupt the result.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {
/// A profile that makes restructuring profitable and is machine-independent:
/// a fast CPU whose GEMM time is dominated by FLOP count, so the DP prefers
/// the cheap parenthesization. Constructing the pass with an explicit profile
/// mirrors how populate_default drives it.
cg::HardwareProfile skewed_profile() {
    cg::HardwareProfile p;
    p.cpu.peak_gflops_fp64          = 100.0;
    p.cpu.mem_bandwidth_gbps        = 40.0;
    p.cpu.kernel_launch_overhead_us = 0.1;
    p.cpu.name                      = "TestCPU";
    return p;
}

size_t count_kind(cg::Graph const &g, cg::OpKind kind) {
    size_t n = 0;
    for (auto const &node : g.nodes())
        if (node.kind == kind)
            n++;
    return n;
}
} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Leaf-orientation gate (the transposed-operand bug)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - transposed interior leaf is NOT restructured", "[ComputeGraph][Passes][CP]") {
    // Regression for the leaf-orientation defect. The restructured chain emits
    // fixed (m,k)x(k,n) GEMM nodes and runs them through the rank-2 fast path,
    // which calls gemm<false, false> on each leaf's PHYSICAL layout - it
    // ignores the captured index order. When an interior leaf is captured
    // TRANSPOSED ("ik;jk->ij", link index LAST), folding it into a plain GEMM
    // reads it in the wrong orientation and silently produces garbage (this
    // chain gave max abs error ~38 before the gate was added). The pass must
    // decline: analysis-only, original order preserved, numerics exact.
    auto A   = create_random_tensor<double>("A", 100, 1);
    auto B   = create_random_tensor<double>("B", 1, 100);
    auto Csq = create_random_tensor<double>("Csq", 100, 100); // square, so the wrong-orientation GEMM would still *run*
    auto D   = create_random_tensor<double>("D", 100, 1);
    auto E   = create_random_tensor<double>("E", 1, 100);
    auto T4  = create_zero_tensor<double>("T4", 100, 100);

    // Eager reference: tensor_algebra::einsum honors the transposed spec.
    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 100);
    auto T3r = create_zero_tensor<double>("T3r", 100, 1);
    auto T4r = create_zero_tensor<double>("T4r", 100, 100);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{j, k}, Csq); // transposed
    tensor_algebra::einsum(0.0, Indices{i, j}, &T3r, 1.0, Indices{i, k}, T2r, Indices{k, j}, D);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T4r, 1.0, Indices{i, k}, T3r, Indices{k, j}, E);

    cg::Graph graph("cp_transposed");
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    auto     &T2 = graph.create_zero_tensor<double, 2>("T2", 100, 100);
    auto     &T3 = graph.create_zero_tensor<double, 2>("T3", 100, 1);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;jk->ij", 0.0, &T2, 1.0, T1, Csq); // interior leaf captured transposed
        cg::einsum("ik;kj->ij", 0.0, &T3, 1.0, T2, D);
        cg::einsum("ik;kj->ij", 0.0, &T4, 1.0, T3, E);
    }

    cg::passes::ContractionPlanning pass(skewed_profile());
    pass.run(graph);

    // The chain is detected (a cost report exists) but declined for folding.
    CHECK(pass.chain_reports().size() == 1);
    CHECK(pass.chains_restructured() == 0);
    CHECK(pass.intermediates_created() == 0);
    // No fold means no emitted Gemm nodes; the original Einsum chain survives.
    CHECK(count_kind(graph, cg::OpKind::Gemm) == 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        for (size_t jj = 0; jj < 100; jj++)
            CHECK(T4(ii, jj) == Catch::Approx(T4r(ii, jj)).margin(1e-8));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Straight-line canonical chain: restructured and correct
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - straight-line 4-GEMM chain restructured and correct", "[ComputeGraph][Passes][CP]") {
    // Canonical "ik;kj->ij" chain with thin (dim-1) links: left-to-right blows
    // up into 100x100 intermediates, so the DP finds a strictly cheaper
    // parenthesization and folds. The pass emits its own Materialize nodes for
    // the deferred intermediates it declares, so a standalone apply stays
    // executable.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 100);
    auto D  = create_random_tensor<double>("D", 100, 1);
    auto E  = create_random_tensor<double>("E", 1, 100);
    auto T4 = create_zero_tensor<double>("T4", 100, 100);

    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 100);
    auto T3r = create_zero_tensor<double>("T3r", 100, 1);
    auto T4r = create_zero_tensor<double>("T4r", 100, 100);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T3r, 1.0, Indices{i, k}, T2r, Indices{k, j}, D);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T4r, 1.0, Indices{i, k}, T3r, Indices{k, j}, E);

    cg::Graph graph("cp_straight");
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    auto     &T2 = graph.create_zero_tensor<double, 2>("T2", 100, 100);
    auto     &T3 = graph.create_zero_tensor<double, 2>("T3", 100, 1);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
        cg::einsum("ik;kj->ij", 0.0, &T3, 1.0, T2, D);
        cg::einsum("ik;kj->ij", 0.0, &T4, 1.0, T3, E);
    }

    cg::passes::ContractionPlanning pass(skewed_profile());
    bool const                      modified = pass.run(graph);

    CHECK(modified);
    CHECK(pass.chains_restructured() >= 1);
    CHECK(pass.intermediates_created() > 0);
    // The pass emits Materialize nodes for its own deferred intermediates.
    CHECK(count_kind(graph, cg::OpKind::Materialize) > 0);
    CHECK(count_kind(graph, cg::OpKind::Gemm) > 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        for (size_t jj = 0; jj < 100; jj++)
            CHECK(T4(ii, jj) == Catch::Approx(T4r(ii, jj)).margin(1e-8));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Contiguity conservatism: an interleaved node breaks chain recognition
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - interleaved node prevents chain recognition", "[ComputeGraph][Passes][CP]") {
    // find_contraction_chains extends a chain only while the NEXT node in
    // topological order is the einsum reading the previous output; the first
    // non-einsum node stops the walk. A truly unrelated node would just float
    // away under the pass's topological_sort, so to force a node BETWEEN the
    // two einsums we scale the chain link in place: the ordering
    // einsum -> scale(T1) -> einsum is dependency-forced. A dependency-based
    // chain finder could still fold the surrounding contractions, so this is a
    // known conservatism, not a correctness requirement - the chain simply is
    // not recognized and nothing is restructured. Pinned so the limitation
    // stays intentional.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 1);
    auto T1 = create_zero_tensor<double>("T1", 100, 100);
    auto T2 = create_zero_tensor<double>("T2", 100, 1);

    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(0.5, &T1r);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C);

    cg::Graph graph("cp_interleaved");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::scale(0.5, &T1); // interleaved writer, dependency-forced between the einsums
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_profile());
    bool const                      modified = pass.run(graph);

    CHECK_FALSE(modified);
    CHECK(pass.chain_reports().empty()); // no chain of length >= 2 recognized
    CHECK(pass.chains_restructured() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        CHECK(T2(ii, 0) == Catch::Approx(T2r(ii, 0)).margin(1e-8));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Observable interior: gate refuses (light version; full regression in
// HardwareProfile.cpp - "user-visible interior blocks restructuring" and
// "outside reader of interior blocks restructuring").
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - user-visible interior blocks restructuring", "[ComputeGraph][Passes][CP]") {
    // Re-parenthesizing elides the writes to interior tensors; that is only
    // legal when the interior is a graph-owned intermediate no one else
    // observes. A user-visible interior (created eagerly, not via the graph)
    // keeps an observable write, so the pass must fall back to analysis-only.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 1);
    auto T1 = create_zero_tensor<double>("T1", 100, 100); // user-visible interior
    auto T2 = create_zero_tensor<double>("T2", 100, 1);

    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C);

    cg::Graph graph("cp_user_interior");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_profile());
    pass.run(graph);
    CHECK(pass.chains_restructured() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++) {
        CHECK(T1(ii, 0) == Catch::Approx(T1r(ii, 0)).margin(1e-8)); // the interior write survived
        CHECK(T2(ii, 0) == Catch::Approx(T2r(ii, 0)).margin(1e-8));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime-tensor chain stays analysis-only (bug-1015 class)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - runtime-tensor chain stays analysis-only", "[ComputeGraph][Passes][CP]") {
    // The rank-2 fold casts each operand to Tensor<T, 2>*. A runtime-rank
    // tensor in the chain would be type confusion at execute (garbage rank),
    // the same hazard LCCF grew a gate for. all_rank2 checks is_runtime and
    // must fall back to analysis-only even when the shape would otherwise be
    // profitable to fold.
    auto A_t = create_random_tensor<double>("A", 100, 1);
    auto B_t = create_random_tensor<double>("B", 1, 100);
    auto C_t = create_random_tensor<double>("C", 100, 1);

    // Runtime-rank captures.
    RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
    RuntimeTensor<double> T1("T1", std::vector<size_t>{100, 100});
    RuntimeTensor<double> T2("T2", std::vector<size_t>{100, 1});
    T1.zero();
    T2.zero();

    cg::Graph graph("cp_runtime");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_profile());
    pass.run(graph);

    // Detected but declined: no fold, no emitted Gemm nodes.
    CHECK(pass.chains_restructured() == 0);
    CHECK(pass.intermediates_created() == 0);
    CHECK(count_kind(graph, cg::OpKind::Gemm) == 0);

    graph.execute();

    // Reference on the fixed-rank originals (compile-time Indices einsum needs
    // static rank, so it can't take a RuntimeTensor directly).
    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A_t, Indices{k, j}, B_t);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C_t);

    for (size_t ii = 0; ii < 100; ii++) {
        std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), 0};
        CHECK(T2(idx) == Catch::Approx(T2r(ii, 0)).margin(1e-8));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Idempotency: a second run adds nothing
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - second run is a no-op", "[ComputeGraph][Passes][CP]") {
    // After restructuring, the chain is Gemm nodes, not an Einsum chain, so a
    // second run finds nothing to fold: no new intermediates, no new
    // Materialize nodes, and the result is unchanged.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 100);
    auto D  = create_random_tensor<double>("D", 100, 1);
    auto E  = create_random_tensor<double>("E", 1, 100);
    auto T4 = create_zero_tensor<double>("T4", 100, 100);

    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 100);
    auto T3r = create_zero_tensor<double>("T3r", 100, 1);
    auto T4r = create_zero_tensor<double>("T4r", 100, 100);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T3r, 1.0, Indices{i, k}, T2r, Indices{k, j}, D);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T4r, 1.0, Indices{i, k}, T3r, Indices{k, j}, E);

    cg::Graph graph("cp_idempotent");
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    auto     &T2 = graph.create_zero_tensor<double, 2>("T2", 100, 100);
    auto     &T3 = graph.create_zero_tensor<double, 2>("T3", 100, 1);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
        cg::einsum("ik;kj->ij", 0.0, &T3, 1.0, T2, D);
        cg::einsum("ik;kj->ij", 0.0, &T4, 1.0, T3, E);
    }

    cg::passes::ContractionPlanning first(skewed_profile());
    REQUIRE(first.run(graph));
    REQUIRE(first.chains_restructured() >= 1);

    size_t const tensors_after_first     = graph.tensors_map().size();
    size_t const gemms_after_first       = count_kind(graph, cg::OpKind::Gemm);
    size_t const materialize_after_first = count_kind(graph, cg::OpKind::Materialize);

    cg::passes::ContractionPlanning second(skewed_profile());
    bool const                      modified_second = second.run(graph);

    CHECK_FALSE(modified_second);
    CHECK(second.chains_restructured() == 0);
    CHECK(second.intermediates_created() == 0);
    CHECK(graph.tensors_map().size() == tensors_after_first); // no duplicate intermediates
    CHECK(count_kind(graph, cg::OpKind::Gemm) == gemms_after_first);
    CHECK(count_kind(graph, cg::OpKind::Materialize) == materialize_after_first);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        for (size_t jj = 0; jj < 100; jj++)
            CHECK(T4(ii, jj) == Catch::Approx(T4r(ii, jj)).margin(1e-8));
}
