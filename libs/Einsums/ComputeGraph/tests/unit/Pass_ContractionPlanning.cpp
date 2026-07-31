//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_ContractionPlanning.cpp
/// @brief Dedicated unit coverage for the ContractionPlanning pass. The cost
///        model and residency behavior live in CostModel.cpp; this file
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
/// A cost_model that makes restructuring profitable and is machine-independent:
/// a fast CPU whose GEMM time is dominated by FLOP count, so the DP prefers
/// the cheap parenthesization. Constructing the pass with an explicit cost_model
/// mirrors how populate_default drives it.
cg::CostModel skewed_model() {
    cg::CostModel p;
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

TEST_CASE("ContractionPlanning - transposed interior leaf is restructured via a transpose flag", "[ComputeGraph][Passes][CP]") {
    // A leaf captured TRANSPOSED ("ik;jk->ij", link index LAST) is stored (N,K)
    // rather than (K,N). The emitted GEMM now carries that as a transpose flag,
    // which BLAS absorbs in its packing step, so the chain folds like any other
    // and no physical permute is inserted.
    //
    // This case used to be declined outright: the fold read every leaf with
    // gemm<false, false> on its physical layout and silently produced garbage
    // (max abs error ~38 on this chain), so the pass refused any non-canonical
    // orientation. The numeric check below is what that gate was protecting.
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

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    CHECK(pass.chains_restructured() == 1);
    CHECK(pass.intermediates_created() > 0);
    CHECK(count_kind(graph, cg::OpKind::Gemm) > 0);
    // The transposed leaf is handled by a flag on the GEMM, not by inserting a
    // permute of its own.
    CHECK(count_kind(graph, cg::OpKind::Permute) == 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        for (size_t jj = 0; jj < 100; jj++)
            CHECK(T4(ii, jj) == Catch::Approx(T4r(ii, jj)).margin(1e-8));
}

TEST_CASE("ContractionPlanning - every operand orientation folds to the same values", "[ComputeGraph][Passes][CP]") {
    // Each of the three leaves can be captured either way round, so the emitted
    // GEMMs have to cover all four transA/transB combinations. The reference is
    // the SAME graph left unfolded: what is being pinned is that restructuring
    // preserves semantics, whatever orientation the chain was captured in.
    //
    // Shapes are chosen so left-to-right (100x1x100 then 100x100x2) is far
    // worse than the right-first association, guaranteeing a fold.
    // The specs must be literals (EinsumFormatString is consteval), so the
    // orientations are template parameters rather than generated values.
    auto run_case = []<bool AT, bool BT, bool CT>() {
        CAPTURE(AT, BT, CT);

        // Mathematical shapes: L0 is 100x1, L1 is 1x100, L2 is 100x2. A
        // transposed leaf is stored with its axes swapped.
        auto L0 = AT ? create_random_tensor<double>("L0", 1, 100) : create_random_tensor<double>("L0", 100, 1);
        auto L1 = BT ? create_random_tensor<double>("L1", 100, 1) : create_random_tensor<double>("L1", 1, 100);
        auto L2 = CT ? create_random_tensor<double>("L2", 2, 100) : create_random_tensor<double>("L2", 100, 2);

        auto build = [&](cg::Graph &g, Tensor<double, 2> &out) {
            auto                  &mid = g.create_zero_tensor<double, 2>("mid", 100, 100);
            cg::CaptureGuard const guard(g);
            if constexpr (!AT && !BT) {
                cg::einsum("ik;kj->ij", 0.0, &mid, 1.0, L0, L1);
            } else if constexpr (!AT && BT) {
                cg::einsum("ik;jk->ij", 0.0, &mid, 1.0, L0, L1);
            } else if constexpr (AT && !BT) {
                cg::einsum("ki;kj->ij", 0.0, &mid, 1.0, L0, L1);
            } else {
                cg::einsum("ki;jk->ij", 0.0, &mid, 1.0, L0, L1);
            }
            if constexpr (!CT) {
                cg::einsum("ij;jl->il", 0.0, &out, 1.0, mid, L2);
            } else {
                cg::einsum("ij;lj->il", 0.0, &out, 1.0, mid, L2);
            }
        };

        // Reference: the same graph, left unfolded. What is pinned is that
        // restructuring preserves semantics in every capture orientation.
        auto      ref = create_zero_tensor<double>("ref", 100, 2);
        cg::Graph ref_graph("cp_orient_ref");
        build(ref_graph, ref);
        ref_graph.execute();

        auto      got = create_zero_tensor<double>("got", 100, 2);
        cg::Graph graph("cp_orient");
        build(graph, got);

        cg::passes::ContractionPlanning pass(skewed_model());
        pass.run(graph);

        CHECK(pass.chains_restructured() == 1);
        CHECK(count_kind(graph, cg::OpKind::Permute) == 0);

        graph.execute();

        double ref_norm = 0.0;
        for (size_t ii = 0; ii < 100; ii++) {
            for (size_t jj = 0; jj < 2; jj++) {
                ref_norm += std::abs(ref(ii, jj));
                CHECK(got(ii, jj) == Catch::Approx(ref(ii, jj)).margin(1e-10));
            }
        }
        // Guard against a comparison of two all-zero results.
        REQUIRE(ref_norm > 1e-8);
    };

    run_case.template operator()<false, false, false>();
    run_case.template operator()<false, false, true>();
    run_case.template operator()<false, true, false>();
    run_case.template operator()<false, true, true>();
    run_case.template operator()<true, false, false>();
    run_case.template operator()<true, false, true>();
    run_case.template operator()<true, true, false>();
    run_case.template operator()<true, true, true>();
}

TEST_CASE("ContractionPlanning - a permuted output keeps the chain analysis-only", "[ComputeGraph][Passes][CP]") {
    // The operand transpose flags cannot express a permuted RESULT: the fold
    // computes an M x N block and there is no flag that writes it as N x M. A
    // member whose output reverses its axes has to stay unfolded.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 100);
    auto T2 = create_zero_tensor<double>("T2", 100, 100);

    cg::Graph graph("cp_permuted_output_declined");
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ji", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    CHECK(pass.chain_reports().size() == 1); // detected and priced
    CHECK(pass.chains_restructured() == 0);  // but not folded
    CHECK(count_kind(graph, cg::OpKind::Gemm) == 0);
}

TEST_CASE("ContractionPlanning - transposed OUTPUT is not folded into a straight GEMM", "[ComputeGraph][Passes][CP]") {
    // The leaf-orientation gate looks only at the operands. A member whose
    // OUTPUT index order is reversed ("ji <- ik ; kj") passes every gate, but
    // the fold emits C[m,n] = A[m,k] * B[k,n] and writes an M x N result into
    // a tensor whose axes are N x M. Square dimensions make that a silent
    // transpose rather than a shape error.
    auto A  = create_random_tensor<double>("A", 100, 1);
    auto B  = create_random_tensor<double>("B", 1, 100);
    auto C  = create_random_tensor<double>("C", 100, 100);
    auto T2 = create_zero_tensor<double>("T2", 100, 100);

    // Eager reference: T2[j,i] = sum_k T1[i,k] * C[k,j].
    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 100);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{j, i}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C);

    cg::Graph graph("cp_transposed_output");
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ji", 0.0, &T2, 1.0, T1, C); // output axes reversed
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        for (size_t jj = 0; jj < 100; jj++)
            CHECK(T2(ii, jj) == Catch::Approx(T2r(ii, jj)).margin(1e-8));
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

    cg::passes::ContractionPlanning pass(skewed_model());
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

TEST_CASE("ContractionPlanning - two dependent chains both restructure correctly", "[ComputeGraph][Passes][CP]") {
    // Regression for STALE NODE INDICES across multiple restructured chains.
    // find_contraction_chains records absolute node positions once; restructuring
    // the first chain rebuilds the node vector (nodes = std::move(result)),
    // shifting every later position. The second chain then removed the wrong
    // nodes and/or dropped its own emitted intermediates, silently corrupting
    // its result. Two INDEPENDENT chains get interleaved by topological_sort and
    // are never both detected, so the trigger is two chains separated by a
    // non-einsum node (here a scale of the first chain's output X): topo order
    // keeps them contiguous-but-separate, find returns two chains, and the first
    // restructure invalidates the second's indices. Both must fold + stay exact.
    auto A1 = create_random_tensor<double>("A1", 100, 1);
    auto B1 = create_random_tensor<double>("B1", 1, 100);
    auto C1 = create_random_tensor<double>("C1", 100, 100);
    auto D1 = create_random_tensor<double>("D1", 100, 1);

    auto G2 = create_random_tensor<double>("G2", 1, 100);
    auto H2 = create_random_tensor<double>("H2", 100, 100);
    auto I2 = create_random_tensor<double>("I2", 100, 1);
    auto Y  = create_zero_tensor<double>("Y", 100, 1);

    // Eager reference: chain 1 (A1.B1.C1.D1) -> X (100x1), scale(X) by 2, then
    // chain 2 (X.G2.H2.I2) -> Y (100x1). Both left-to-right orders build a 100x100
    // intermediate and hit a 100^3 contraction that a cheaper parenthesization
    // avoids, so the DP folds BOTH.
    auto X1r = create_zero_tensor<double>("X1r", 100, 100);
    auto X2r = create_zero_tensor<double>("X2r", 100, 100);
    auto Xr  = create_zero_tensor<double>("Xr", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &X1r, 1.0, Indices{i, k}, A1, Indices{k, j}, B1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &X2r, 1.0, Indices{i, k}, X1r, Indices{k, j}, C1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &Xr, 1.0, Indices{i, k}, X2r, Indices{k, j}, D1);
    linear_algebra::scale(2.0, &Xr);
    auto Y1r = create_zero_tensor<double>("Y1r", 100, 100);
    auto Y2r = create_zero_tensor<double>("Y2r", 100, 100);
    auto Yr  = create_zero_tensor<double>("Yr", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &Y1r, 1.0, Indices{i, k}, Xr, Indices{k, j}, G2);
    tensor_algebra::einsum(0.0, Indices{i, j}, &Y2r, 1.0, Indices{i, k}, Y1r, Indices{k, j}, H2);
    tensor_algebra::einsum(0.0, Indices{i, j}, &Yr, 1.0, Indices{i, k}, Y2r, Indices{k, j}, I2);

    cg::Graph graph("cp_two_dep_chains");
    auto     &X1 = graph.create_zero_tensor<double, 2>("X1", 100, 100);
    auto     &X2 = graph.create_zero_tensor<double, 2>("X2", 100, 100);
    auto     &X  = graph.create_zero_tensor<double, 2>("X", 100, 1);
    auto     &Y1 = graph.create_zero_tensor<double, 2>("Y1", 100, 100);
    auto     &Y2 = graph.create_zero_tensor<double, 2>("Y2", 100, 100);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &X1, 1.0, A1, B1);
        cg::einsum("ik;kj->ij", 0.0, &X2, 1.0, X1, C1);
        cg::einsum("ik;kj->ij", 0.0, &X, 1.0, X2, D1);
        cg::scale(2.0, &X); // non-einsum separator: keeps the two chains ordered but distinct
        cg::einsum("ik;kj->ij", 0.0, &Y1, 1.0, X, G2);
        cg::einsum("ik;kj->ij", 0.0, &Y2, 1.0, Y1, H2);
        cg::einsum("ik;kj->ij", 0.0, &Y, 1.0, Y2, I2);
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    bool const                      modified = pass.run(graph);

    CHECK(modified);
    CHECK(pass.chains_restructured() >= 2); // BOTH chains fold

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++)
        CHECK(Y(ii, 0) == Catch::Approx(Yr(ii, 0)).margin(1e-8)); // corrupted by the stale-index bug
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

    cg::passes::ContractionPlanning pass(skewed_model());
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
// CostModel.cpp - "user-visible interior blocks restructuring" and
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

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);
    CHECK(pass.chains_restructured() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 100; ii++) {
        CHECK(T1(ii, 0) == Catch::Approx(T1r(ii, 0)).margin(1e-8)); // the interior write survived
        CHECK(T2(ii, 0) == Catch::Approx(T2r(ii, 0)).margin(1e-8));
    }
}

TEST_CASE("ContractionPlanning - a leaf that is an earlier interior blocks the fold", "[ComputeGraph][Passes][CP]") {
    // Found by test_fuzz_free_lifecycle_parallel_stress. The last member's
    // fresh operand is T0, which is also the FIRST member's output. The DP is
    // handed T0 as an independent leaf, but re-parenthesizing elides the write
    // that produces it, so the restructured tree reads a buffer nothing
    // computes any more - and once FreeInsertion is in the pipeline, one that
    // has been released. The shape is a DAG, not a chain.
    //
    // The observable-interior scan cannot catch this on its own: it skips
    // readers that are chain members, which is right for the chain LINK.
    auto A   = create_random_tensor<double>("A", 100, 1);
    auto B   = create_random_tensor<double>("B", 1, 100);
    auto C   = create_random_tensor<double>("C", 100, 100);
    auto out = create_zero_tensor<double>("out", 100, 100);

    auto T0r  = create_zero_tensor<double>("T0r", 100, 100);
    auto T1r  = create_zero_tensor<double>("T1r", 100, 100);
    auto outr = create_zero_tensor<double>("outr", 100, 100);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T0r, 1.0, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, T0r, Indices{k, j}, C);
    tensor_algebra::einsum(0.0, Indices{i, j}, &outr, 1.0, Indices{i, k}, T1r, Indices{k, j}, T0r);

    cg::Graph graph("cp_leaf_is_interior");
    auto     &T0 = graph.create_zero_tensor<double, 2>("T0", 100, 100);
    auto     &T1 = graph.create_zero_tensor<double, 2>("T1", 100, 100);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T0, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, T0, C);
        cg::einsum("ik;kj->ij", 0.0, &out, 1.0, T1, T0); // leaf is the first member's output
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    CHECK(pass.chains_restructured() == 0);
    CHECK(count_kind(graph, cg::OpKind::Gemm) == 0);

    graph.execute();

    double ref_norm = 0.0;
    for (size_t ii = 0; ii < 100; ii++) {
        for (size_t jj = 0; jj < 100; jj++) {
            ref_norm += std::abs(outr(ii, jj));
            CHECK(out(ii, jj) == Catch::Approx(outr(ii, jj)).margin(1e-8));
        }
    }
    REQUIRE(ref_norm > 1e-8);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime-rank chains
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContractionPlanning - runtime-rank chain is restructured", "[ComputeGraph][Passes][CP]") {
    // RuntimeTensor is the Python-facing type, so this is the shape any graph
    // captured from Python has. It used to be declined outright: the fold cast
    // every operand to Tensor<T,2>*, which for a runtime tensor is type
    // confusion. The emitted executor now goes through each tensor's impl and
    // the dynamic-rank gemm overload, so the chain folds like a typed one.
    auto A_t = create_random_tensor<double>("A", 100, 1);
    auto B_t = create_random_tensor<double>("B", 1, 100);
    auto C_t = create_random_tensor<double>("C", 100, 1);

    RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
    RuntimeTensor<double> T2("T2", std::vector<size_t>{100, 1});
    T2.zero();

    cg::Graph graph("cp_runtime");
    // Graph-owned interior, so the observable-interior gate does not fire.
    auto &T1 = graph.create_runtime_tensor<double>("T1", {100, 100}, /*intermediate=*/true);
    T1.zero();
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    CHECK(pass.chains_restructured() == 1);
    CHECK(count_kind(graph, cg::OpKind::Gemm) > 0);

    graph.execute();

    // Reference on the fixed-rank originals (compile-time Indices einsum needs
    // static rank, so it can't take a RuntimeTensor directly).
    auto T1r = create_zero_tensor<double>("T1r", 100, 100);
    auto T2r = create_zero_tensor<double>("T2r", 100, 1);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T1r, 1.0, Indices{i, k}, A_t, Indices{k, j}, B_t);
    tensor_algebra::einsum(0.0, Indices{i, j}, &T2r, 1.0, Indices{i, k}, T1r, Indices{k, j}, C_t);

    double ref_norm = 0.0;
    for (size_t ii = 0; ii < 100; ii++) {
        std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), 0};
        ref_norm += std::abs(T2r(ii, 0));
        CHECK(T2(idx) == Catch::Approx(T2r(ii, 0)).margin(1e-8));
    }
    REQUIRE(ref_norm > 1e-8);
}

TEST_CASE("ContractionPlanning - user-visible runtime interior still blocks the fold", "[ComputeGraph][Passes][CP]") {
    // Runtime-ness is no longer the blocker, but the observable-interior rule
    // is unchanged: a user-created interior keeps a write the fold would elide.
    auto A_t = create_random_tensor<double>("A", 100, 1);
    auto B_t = create_random_tensor<double>("B", 1, 100);
    auto C_t = create_random_tensor<double>("C", 100, 1);

    RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
    RuntimeTensor<double> T1("T1", std::vector<size_t>{100, 100}); // user-visible interior
    RuntimeTensor<double> T2("T2", std::vector<size_t>{100, 1});
    T1.zero();
    T2.zero();

    cg::Graph graph("cp_runtime_user_interior");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &T2, 1.0, T1, C);
    }

    cg::passes::ContractionPlanning pass(skewed_model());
    pass.run(graph);

    CHECK(pass.chains_restructured() == 0);
    CHECK(count_kind(graph, cg::OpKind::Gemm) == 0);

    graph.execute();

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

    cg::passes::ContractionPlanning first(skewed_model());
    REQUIRE(first.run(graph));
    REQUIRE(first.chains_restructured() >= 1);

    size_t const tensors_after_first     = graph.tensors_map().size();
    size_t const gemms_after_first       = count_kind(graph, cg::OpKind::Gemm);
    size_t const materialize_after_first = count_kind(graph, cg::OpKind::Materialize);

    cg::passes::ContractionPlanning second(skewed_model());
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
