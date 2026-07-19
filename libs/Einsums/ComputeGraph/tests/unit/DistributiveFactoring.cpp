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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();

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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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

    auto [modified, pass] = graph.apply<cg::passes::DistributiveFactoring>();
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
    auto [m1, p1] = graph.apply<cg::passes::DistributiveFactoring>();
    REQUIRE(m1);
    REQUIRE(p1.num_groups() == 1);

    // Second pass finds no einsum group left (the combined node is a Custom op).
    auto [m2, p2] = graph.apply<cg::passes::DistributiveFactoring>();
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

    graph.apply<cg::passes::DistributiveFactoring>();

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
