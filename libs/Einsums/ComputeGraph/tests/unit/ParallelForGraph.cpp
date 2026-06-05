//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Tests for cg::parallel_for and cg::parallel_reduce graph-capturable wrappers.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("cg::parallel_for - basic capture and execute", "[ComputeGraph][ParallelFor]") {
    auto A = create_zero_tensor<double>("A", 100);

    cg::Graph graph("pf_basic");
    {
        cg::CaptureGuard const guard(graph);
        cg::parallel_for(
            "fill", 0, 100, [&A](size_t i) { A(i) = static_cast<double>(i); }, &A);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t i = 0; i < 100; i++) {
        REQUIRE_THAT(A(i), Catch::Matchers::WithinRel(static_cast<double>(i), 1e-12));
    }
}

TEST_CASE("cg::parallel_for - ordering with einsum", "[ComputeGraph][ParallelFor]") {
    // parallel_for fills J_mat, then einsum reads J_mat.
    // The graph's dependency tracking must ensure parallel_for runs first.
    constexpr size_t N = 10;
    auto             J = create_zero_tensor<double>("J", N, N);
    auto             D = create_random_tensor<double>("D", N, N);
    auto             F = create_zero_tensor<double>("F", N, N);

    // Reference: fill J with D*D element-wise, then F = J * D
    auto J_ref = create_zero_tensor<double>("J_ref", N, N);
    for (size_t ii = 0; ii < N; ii++)
        for (size_t jj = 0; jj < N; jj++)
            J_ref(ii, jj) = D(ii, jj) * D(ii, jj);
    auto F_ref = create_zero_tensor<double>("F_ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &F_ref, Indices{i, k}, J_ref, Indices{k, j}, D);

    // Graph: parallel_for to fill J, then einsum F = J * D
    cg::Graph graph("pf_then_einsum");
    {
        cg::CaptureGuard const guard(graph);
        cg::parallel_for(
            "fill_J", 0, N * N,
            [&J, &D, N](size_t flat) {
                size_t const row = flat / N;
                size_t const col = flat % N;
                J(row, col)      = D(row, col) * D(row, col);
            },
            &J);
        cg::einsum("ik;kj->ij", &F, J, D);
    }

    REQUIRE(graph.num_nodes() == 2);
    graph.execute();

    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE_THAT(F(ii, jj), Catch::Matchers::WithinRel(F_ref(ii, jj), 1e-10));
        }
    }
}

TEST_CASE("cg::parallel_for - replay", "[ComputeGraph][ParallelFor]") {
    auto A = create_zero_tensor<double>("A", 50);

    cg::Graph graph("pf_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::parallel_for(
            "fill", 0, 50, [&A](size_t i) { A(i) = static_cast<double>(i * i); }, &A);
    }

    graph.execute();
    REQUIRE_THAT(A(7), Catch::Matchers::WithinRel(49.0, 1e-12));

    // Replay should produce same result
    A.zero();
    graph.execute();
    REQUIRE_THAT(A(7), Catch::Matchers::WithinRel(49.0, 1e-12));
}

TEST_CASE("cg::parallel_reduce - basic capture and execute", "[ComputeGraph][ParallelReduce]") {
    auto A = create_random_tensor<double>("A", 100);

    // Reference: sum of squares
    double ref = 0.0;
    for (size_t i = 0; i < 100; i++) {
        ref += A(i) * A(i);
    }

    double    result = 0.0;
    cg::Graph graph("pr_basic");
    {
        cg::CaptureGuard const guard(graph);
        cg::parallel_reduce<double>(
            "sum_sq", 0, 100, &result, []() { return 0.0; }, [&A](size_t i, double &acc) { acc += A(i) * A(i); },
            [](double &g, double const &l) { g += l; }, &A);
    }

    graph.execute();
    REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref, 1e-10));
}

TEST_CASE("cg::parallel_for + einsum + parallel_reduce in one graph", "[ComputeGraph][ParallelFor]") {
    // Full pipeline:
    // 1. parallel_for: fill J from D
    // 2. einsum: F = J * D
    // 3. parallel_reduce: energy = sum(D * F)
    constexpr size_t N = 8;
    auto             D = create_random_tensor<double>("D", N, N);
    auto             J = create_zero_tensor<double>("J", N, N);
    auto             F = create_zero_tensor<double>("F", N, N);

    // Reference
    auto J_ref = create_zero_tensor<double>("J_ref", N, N);
    for (size_t ii = 0; ii < N; ii++)
        for (size_t jj = 0; jj < N; jj++)
            J_ref(ii, jj) = D(ii, jj) * 2.0;
    auto F_ref = create_zero_tensor<double>("F_ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &F_ref, Indices{i, k}, J_ref, Indices{k, j}, D);
    double energy_ref = 0.0;
    for (size_t ii = 0; ii < N; ii++)
        for (size_t jj = 0; jj < N; jj++)
            energy_ref += D(ii, jj) * F_ref(ii, jj);

    double    energy = 0.0;
    cg::Graph graph("full_pipeline");
    {
        cg::CaptureGuard const guard(graph);
        // Step 1: parallel_for fills J
        cg::parallel_for(
            "fill_J", 0, N * N, [&J, &D, N](size_t flat) { J(flat / N, flat % N) = D(flat / N, flat % N) * 2.0; }, &J);
        // Step 2: einsum F = J * D
        cg::einsum("ik;kj->ij", &F, J, D);
        // Step 3: parallel_reduce energy = sum(D * F)
        cg::parallel_reduce<double>(
            "energy", 0, N * N, &energy, []() { return 0.0; },
            [&D, &F, N](size_t flat, double &acc) { acc += D(flat / N, flat % N) * F(flat / N, flat % N); },
            [](double &g, double const &l) { g += l; }, &D, &F);
    }

    REQUIRE(graph.num_nodes() == 3);
    graph.execute();

    REQUIRE_THAT(energy, Catch::Matchers::WithinRel(energy_ref, 1e-10));
}

TEST_CASE("cg::parallel_for - outside capture executes immediately", "[ComputeGraph][ParallelFor]") {
    auto A = create_zero_tensor<double>("A", 20);

    // No capture guard — should execute immediately via TaskPool
    cg::parallel_for(
        "fill", 0, 20, [&A](size_t i) { A(i) = static_cast<double>(i); }, &A);

    for (size_t i = 0; i < 20; i++) {
        REQUIRE_THAT(A(i), Catch::Matchers::WithinRel(static_cast<double>(i), 1e-12));
    }
}
