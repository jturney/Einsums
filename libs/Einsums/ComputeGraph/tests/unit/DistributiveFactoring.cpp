//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

/// A machine profile on which factoring always pays: contractions are slow and
/// memory is fast, so the axpy chain can never outweigh the saved contractions.
///
/// Most cases below are about the REWRITE, not the pricing, and they use shapes
/// small enough that the real profile would rightly call factoring pointless.
/// Pricing gets its own two cases at the end, against the detected profile.
static cg::CostModel favors_factoring() {
    cg::CostModel cm;
    cm.cpu.peak_gflops_fp64          = 1e-3; // contractions are ruinously slow
    cm.cpu.mem_bandwidth_gbps        = 1e6;  // memory traffic is free
    cm.cpu.kernel_launch_overhead_us = 0.0;
    cm.cpu.alloc_overhead_us         = 0.0;
    cm.cpu.gemm_efficiency.clear();
    return cm;
}

TEST_CASE("DistributiveFactoring - rewrites 2 terms sharing operand A", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);

    // Reference: R_ref = A*B1 + A*B2
    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);

    // Graph version
    auto      R = create_zero_tensor<double>("R", 4, 5);
    cg::Graph graph("factor2");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());

    REQUIRE(modified);
    REQUIRE(pass.num_groups() >= 1);
    REQUIRE(pass.num_eliminated() >= 1);

    // Execute factored graph and verify correctness
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
        }
    }
}

TEST_CASE("DistributiveFactoring - rewrites 3 terms", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto B3 = create_random_tensor<double>("B3", 3, 5);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B3);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    cg::Graph graph("factor3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B3);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() >= 1);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
        }
    }
}

TEST_CASE("DistributiveFactoring - no rewrite when shapes differ", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 7);
    auto R1 = create_zero_tensor<double>("R1", 4, 5);
    auto R2 = create_zero_tensor<double>("R2", 4, 7);

    cg::Graph graph("no_factor");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R1, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R2, 1.0, A, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_groups() == 0);
}

TEST_CASE("DistributiveFactoring - no rewrite for non-accumulating", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto R  = create_zero_tensor<double>("R", 4, 5);

    cg::Graph graph("no_accum");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, A, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE_FALSE(modified);
}

TEST_CASE("DistributiveFactoring - shared operand on B side", "[ComputeGraph][DistributiveFactoring]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A1, Indices{k, j}, B);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A2, Indices{k, j}, B);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    cg::Graph graph("factor_b_shared");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A1, B);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A2, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
        }
    }
}

TEST_CASE("DistributiveFactoring - rank-4 contraction", "[ComputeGraph][DistributiveFactoring]") {
    auto g  = create_random_tensor<double>("g", 3, 3, 2, 2);
    auto T1 = create_random_tensor<double>("T1", 2, 2, 4, 4);
    auto T2 = create_random_tensor<double>("T2", 2, 2, 4, 4);

    auto R_ref = create_zero_tensor<double>("R_ref", 3, 3, 4, 4);
    tensor_algebra::einsum(1.0, Indices{i, j, a, b}, &R_ref, 1.0, Indices{i, j, k, l}, g, Indices{k, l, a, b}, T1);
    tensor_algebra::einsum(1.0, Indices{i, j, a, b}, &R_ref, 1.0, Indices{i, j, k, l}, g, Indices{k, l, a, b}, T2);

    auto      R = create_zero_tensor<double>("R", 3, 3, 4, 4);
    cg::Graph graph("factor_rank4");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijkl;klab->ijab", 1.0, &R, 1.0, g, T1);
        cg::einsum("ijkl;klab->ijab", 1.0, &R, 1.0, g, T2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            for (size_t aa = 0; aa < 4; aa++)
                for (size_t bb = 0; bb < 4; bb++)
                    REQUIRE_THAT(R(ii, jj, aa, bb), Catch::Matchers::WithinRel(R_ref(ii, jj, aa, bb), 1e-10));
}

// ── Edge case tests ──────────────────────────────────────────────────────────

TEST_CASE("DistributiveFactoring - different ab_prefactors", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);

    // Reference: R = 0.5*A*B1 + 2.0*A*B2
    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 0.5, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 2.0, Indices{i, k}, A, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    cg::Graph graph("mixed_prefactors");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 0.5, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 2.0, A, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - downstream reader keeps program order", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto E  = create_random_tensor<double>("E", 5, 2);

    // Reference: R = A*B1 + A*B2 ; S = R*E (S reads the factored output R)
    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    auto S_ref = create_zero_tensor<double>("S_ref", 4, 2);
    tensor_algebra::einsum(0.0, Indices{i, l}, &S_ref, 1.0, Indices{i, j}, R_ref, Indices{j, l}, E);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 4, 2);
    cg::Graph graph("factor_downstream");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("ij;jl->il", 0.0, &S, 1.0, R, E); // downstream reader of the factored R
    }

    // Run through a PassManager so the program-order verifier (check_observed_writes)
    // runs. Appending the combined node would trip it (the later S-reader would then
    // observe R's initial contents); first-member-slot placement keeps it ahead.
    cg::PassManager pm;
    pm.add<cg::passes::DistributiveFactoring>();
    REQUIRE_NOTHROW(pm.run(graph));

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t ll = 0; ll < 2; ll++)
            REQUIRE_THAT(S(ii, ll), Catch::Matchers::WithinRel(S_ref(ii, ll), 1e-10));
}

TEST_CASE("DistributiveFactoring - idempotent", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    cg::Graph graph("factor_idempotent");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
    }

    // First pass factors the group.
    auto [m1, p1] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(m1);
    REQUIRE(p1.num_groups() == 1);

    // Second pass finds no einsum group left (the combined node is a Custom op).
    auto [m2, p2] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE_FALSE(m2);
    REQUIRE(p2.num_groups() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - replay factored graph", "[ComputeGraph][DistributiveFactoring]") {
    auto A  = create_random_tensor<double>("A", 3, 2);
    auto B1 = create_random_tensor<double>("B1", 2, 4);
    auto B2 = create_random_tensor<double>("B2", 2, 4);

    auto R_ref = create_zero_tensor<double>("R_ref", 3, 4);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 3, 4);
    cg::Graph graph("replay_factored");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
    }

    graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());

    // First execute
    graph.execute();
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));

    // Replay
    R.zero();
    graph.execute();
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - declines a member accumulating with c_pf != 1", "[ComputeGraph][DistributiveFactoring]") {
    // The factored form applies the output prefactor once, so a member whose
    // c_pf is not 1 rescales the partial sum its predecessors wrote. Folding it
    // silently produced a wrong answer before this was gated.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto R0 = create_random_tensor<double>("R0", 4, 5);

    // R_ref = 2*(R0 + A*B1) + A*B2, the sequential meaning of the two nodes.
    auto R_ref = Tensor<double, 2>(R0);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(2.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);

    auto      R = Tensor<double, 2>(R0);
    cg::Graph graph("non_unit_accumulate");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 2.0, &R, 1.0, A, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_groups() == 0);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
        }
    }
}

TEST_CASE("DistributiveFactoring - emits ordinary nodes, not one opaque node", "[ComputeGraph][DistributiveFactoring]") {
    // The rewrite used to fuse the sum and the contraction into a single
    // OpKind::Custom node whose executor swapped a slot pointer. That hid the
    // intermediate from every other pass. Keep the lowering visible: one zeroing
    // Scale, one Axpy per summed operand, one Einsum.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto B3 = create_random_tensor<double>("B3", 3, 5);
    auto R  = create_zero_tensor<double>("R", 4, 5);

    cg::Graph graph("explicit_nodes");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B3);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    size_t num_custom = 0, num_scale = 0, num_axpy = 0, num_einsum = 0;
    for (auto const &node : graph.nodes()) {
        switch (node.kind) {
        case cg::OpKind::Custom:
            num_custom++;
            break;
        case cg::OpKind::Scale:
            num_scale++;
            break;
        case cg::OpKind::Axpby:
            num_axpy++;
            break;
        case cg::OpKind::Einsum:
            num_einsum++;
            break;
        default:
            break;
        }
    }
    REQUIRE(num_custom == 0);
    REQUIRE(num_scale == 1);
    REQUIRE(num_axpy == 3);
    REQUIRE(num_einsum == 1);
}

TEST_CASE("DistributiveFactoring - two consumers share one summed intermediate", "[ComputeGraph][DistributiveFactoring]") {
    // The CCSD tau shape: several intermediates contract the SAME sum of operands
    // against the same shared operand. Building it once is what makes tau a named
    // quantity rather than one buffer per consumer.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto C  = create_random_tensor<double>("C", 6, 3);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    auto S_ref = create_zero_tensor<double>("S_ref", 6, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 1.0, Indices{l, k}, C, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 1.0, Indices{l, k}, C, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 6, 5);
    cg::Graph graph("shared_sum");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("lk;kj->lj", 1.0, &S, 1.0, C, B1);
        cg::einsum("lk;kj->lj", 1.0, &S, 1.0, C, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 2);

    // One build (a zeroing Scale plus one Axpy per operand) feeding two einsums.
    size_t num_scale = 0, num_axpy = 0, num_einsum = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Scale)
            num_scale++;
        else if (node.kind == cg::OpKind::Axpby)
            num_axpy++;
        else if (node.kind == cg::OpKind::Einsum)
            num_einsum++;
    }
    REQUIRE(num_scale == 1);
    REQUIRE(num_axpy == 2);
    REQUIRE(num_einsum == 2);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ll = 0; ll < 6; ll++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(S(ll, jj), Catch::Matchers::WithinRel(S_ref(ll, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - sums differing only in a prefactor are not shared", "[ComputeGraph][DistributiveFactoring]") {
    // CCSD's tau and tau-tilde sum the same operands and differ by one
    // coefficient, so they are different tensors. Sharing them would be wrong.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto C  = create_random_tensor<double>("C", 6, 3);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    auto S_ref = create_zero_tensor<double>("S_ref", 6, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 1.0, Indices{l, k}, C, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 0.5, Indices{l, k}, C, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 6, 5);
    cg::Graph graph("tau_vs_tau_tilde");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("lk;kj->lj", 1.0, &S, 1.0, C, B1);
        cg::einsum("lk;kj->lj", 1.0, &S, 0.5, C, B2); // half the second term
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 2);

    // Two distinct sums, so two builds.
    size_t num_scale = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Scale)
            num_scale++;
    }
    REQUIRE(num_scale == 2);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ll = 0; ll < 6; ll++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(S(ll, jj), Catch::Matchers::WithinRel(S_ref(ll, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - a rewritten operand blocks the second group", "[ComputeGraph][DistributiveFactoring]") {
    // A summed operand overwritten between two consumers of the same sum must not
    // let the second read the first's intermediate, which was built from the old
    // value.
    //
    // In practice the interference gate gets there first: topological_sort floats
    // independent writes early, so the overwrite lands inside the second group's
    // span and that group declines outright rather than declining only the reuse.
    // Either way the answer is right, which is what this pins. The reuse pass has
    // its own staleness scan for the case where a write falls between the build
    // and a later span; that is defense in depth and not reachable from here.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto Bs = create_random_tensor<double>("Bs", 3, 5);
    auto C  = create_random_tensor<double>("C", 6, 3);
    auto Cs = create_random_tensor<double>("Cs", 6, 3);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    auto S_ref = create_zero_tensor<double>("S_ref", 6, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    // B2 = Bs and C = Cs happen in between, so S sees the new values.
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 1.0, Indices{l, k}, Cs, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 1.0, Indices{l, k}, Cs, Indices{k, j}, Bs);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 6, 5);
    cg::Graph graph("stale_operand");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::axpby(1.0, Bs, 0.0, &B2); // overwrite a summed operand
        cg::axpby(1.0, Cs, 0.0, &C);  // and the shared one, to pin the order
        cg::einsum("lk;kj->lj", 1.0, &S, 1.0, C, B1);
        cg::einsum("lk;kj->lj", 1.0, &S, 1.0, C, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1); // only the group before the overwrite

    // One build, and the second pair of contractions is left alone rather than
    // pointed at an intermediate holding the pre-overwrite operands.
    size_t num_scale = 0, num_einsum = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Scale)
            num_scale++;
        else if (node.kind == cg::OpKind::Einsum)
            num_einsum++;
    }
    REQUIRE(num_scale == 1);
    REQUIRE(num_einsum == 3); // one factored, two untouched

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ll = 0; ll < 6; ll++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(S(ll, jj), Catch::Matchers::WithinRel(S_ref(ll, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - a proportional sum reuses the build and scales the contraction",
          "[ComputeGraph][DistributiveFactoring]") {
    // The second consumer wants the same sum at half strength. Building it twice
    // would cost a whole extra buffer and axpy chain; the ratio rides on the
    // contraction's ab_pf instead. CCSD consumes tau with 1/4 in W_mnij and
    // W_abef and with 1/2 in the T2 equation, so this is the ordinary case.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto C  = create_random_tensor<double>("C", 6, 3);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    auto S_ref = create_zero_tensor<double>("S_ref", 6, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 0.5, Indices{l, k}, C, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 0.5, Indices{l, k}, C, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 6, 5);
    cg::Graph graph("proportional_sum");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("lk;kj->lj", 1.0, &S, 0.5, C, B1);
        cg::einsum("lk;kj->lj", 1.0, &S, 0.5, C, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 2);

    size_t num_scale = 0, num_axpy = 0, num_einsum = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Scale)
            num_scale++;
        else if (node.kind == cg::OpKind::Axpby)
            num_axpy++;
        else if (node.kind == cg::OpKind::Einsum)
            num_einsum++;
    }
    REQUIRE(num_scale == 1); // one build, not two
    REQUIRE(num_axpy == 2);
    REQUIRE(num_einsum == 2);

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ll = 0; ll < 6; ll++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(S(ll, jj), Catch::Matchers::WithinRel(S_ref(ll, jj), 1e-10));
}

TEST_CASE("DistributiveFactoring - a ratio that is not a power of two is not shared", "[ComputeGraph][DistributiveFactoring]") {
    // Scaling the assembled sum only equals scaling each term when the factor is
    // a power of two, so a ratio of three declines and builds its own sum. The
    // answer is right either way; this pins the conservative choice.
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B1 = create_random_tensor<double>("B1", 3, 5);
    auto B2 = create_random_tensor<double>("B2", 3, 5);
    auto C  = create_random_tensor<double>("C", 6, 3);

    auto R_ref = create_zero_tensor<double>("R_ref", 4, 5);
    auto S_ref = create_zero_tensor<double>("S_ref", 6, 5);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 3.0, Indices{l, k}, C, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{l, j}, &S_ref, 3.0, Indices{l, k}, C, Indices{k, j}, B2);

    auto      R = create_zero_tensor<double>("R", 4, 5);
    auto      S = create_zero_tensor<double>("S", 6, 5);
    cg::Graph graph("odd_ratio");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, A, B2);
        cg::einsum("lk;kj->lj", 1.0, &S, 3.0, C, B1);
        cg::einsum("lk;kj->lj", 1.0, &S, 3.0, C, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>(favors_factoring());
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 2);

    size_t num_scale = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == cg::OpKind::Scale)
            num_scale++;
    }
    REQUIRE(num_scale == 2); // two builds

    graph.execute();
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(R(ii, jj), Catch::Matchers::WithinRel(R_ref(ii, jj), 1e-10));
    for (size_t ll = 0; ll < 6; ll++)
        for (size_t jj = 0; jj < 5; jj++)
            REQUIRE_THAT(S(ll, jj), Catch::Matchers::WithinRel(S_ref(ll, jj), 1e-10));
}

// ── Profitability, priced against the DETECTED machine profile ───────────────

TEST_CASE("DistributiveFactoring - factors a flop-bound contraction", "[ComputeGraph][DistributiveFactoring]") {
    // Square-ish contraction with real reuse: three of them cost far more than
    // one plus an axpy chain over the operands.
    constexpr size_t D  = 120;
    auto             S  = create_random_tensor<double>("S", D, D);
    auto             B1 = create_random_tensor<double>("B1", D, D);
    auto             B2 = create_random_tensor<double>("B2", D, D);
    auto             B3 = create_random_tensor<double>("B3", D, D);
    auto             R  = create_zero_tensor<double>("R", D, D);

    cg::Graph graph("flop_bound");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B2);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B3);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);
    REQUIRE(pass.num_unprofitable() == 0);
}

TEST_CASE("DistributiveFactoring - declines a bandwidth-bound contraction", "[ComputeGraph][DistributiveFactoring]") {
    // Tiny output, huge operands: each contraction is a single streaming pass, so
    // assembling the sum touches the same bytes three times to save passes that
    // were already memory-bound. Factoring loses, and would also cost a buffer.
    constexpr size_t I = 2, K = 2, J = 300000;
    auto             S  = create_random_tensor<double>("S", I, K);
    auto             B1 = create_random_tensor<double>("B1", K, J);
    auto             B2 = create_random_tensor<double>("B2", K, J);
    auto             B3 = create_random_tensor<double>("B3", K, J);
    auto             R  = create_zero_tensor<double>("R", I, J);

    // Reference: the unfactored meaning, which must survive the decline.
    auto R_ref = create_zero_tensor<double>("R_ref", I, J);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, S, Indices{k, j}, B1);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, S, Indices{k, j}, B2);
    tensor_algebra::einsum(1.0, Indices{i, j}, &R_ref, 1.0, Indices{i, k}, S, Indices{k, j}, B3);

    cg::Graph graph("bandwidth_bound");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B1);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B2);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, S, B3);
    }

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_groups() == 0);
    REQUIRE(pass.num_unprofitable() == 1);

    // Untouched graphs still have to compute the right answer.
    graph.execute();
    for (size_t ii = 0; ii < I; ii++) {
        REQUIRE_THAT(R(ii, 0), Catch::Matchers::WithinRel(R_ref(ii, 0), 1e-10));
        REQUIRE_THAT(R(ii, J - 1), Catch::Matchers::WithinRel(R_ref(ii, J - 1), 1e-10));
    }
}
