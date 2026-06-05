//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GEMMBatching.cpp
/// @brief Tests for the GEMMBatching optimization pass (collapse groups of
///        independent GEMM-pattern einsums into a single `blas::gemm_batch` call).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr double kTol = 1e-10;

template <typename T>
void require_close(Tensor<T, 2> const &got, Tensor<T, 2> const &ref) {
    REQUIRE(got.size() == ref.size());
    T const *g = got.data();
    T const *r = ref.data();
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(std::abs(g[i] - r[i]) < kTol);
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Basic batching
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GEMMBatching: two independent double GEMMs collapse into one BatchedGemm", "[ComputeGraph][Optimizer][GEMMBatching]") {
    constexpr size_t M = 5, K = 4, N = 3;
    auto             A1 = create_random_tensor<double>("A1", M, K);
    auto             B1 = create_random_tensor<double>("B1", K, N);
    auto             A2 = create_random_tensor<double>("A2", M, K);
    auto             B2 = create_random_tensor<double>("B2", K, N);

    auto C1_ref = create_zero_tensor<double>("C1_ref", M, N);
    auto C2_ref = create_zero_tensor<double>("C2_ref", M, N);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", &C1_ref, A1, B1);
        cg::einsum("ik;kj->ij", &C2_ref, A2, B2);
        g.execute();
    }

    auto      C1 = create_zero_tensor<double>("C1", M, N);
    auto      C2 = create_zero_tensor<double>("C2", M, N);
    cg::Graph graph("batch2");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }
    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 1);
    REQUIRE(pass.total_batched() == 2);
    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(graph.nodes()[0].kind == cg::OpKind::BatchedGemm);

    auto const *d = std::get_if<cg::BatchedGemmDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(d != nullptr);
    REQUIRE(d->batch_count == 2);
    REQUIRE(std::cmp_equal(d->m, M));
    REQUIRE(std::cmp_equal(d->n, N));
    REQUIRE(std::cmp_equal(d->k, K));
    REQUIRE(d->scalar == cg::BlasScalar::Double);

    graph.execute();
    require_close(C1, C1_ref);
    require_close(C2, C2_ref);
}

TEST_CASE("GEMMBatching: five GEMMs all batch together", "[ComputeGraph][Optimizer][GEMMBatching]") {
    constexpr size_t M = 4, K = 6, N = 5;
    constexpr size_t BATCH = 5;

    std::vector<Tensor<double, 2>> As, Bs, Cs, Refs;
    As.reserve(BATCH);
    Bs.reserve(BATCH);
    Cs.reserve(BATCH);
    Refs.reserve(BATCH);
    for (size_t i = 0; i < BATCH; i++) {
        As.push_back(create_random_tensor<double>(fmt::format("A{}", i), M, K));
        Bs.push_back(create_random_tensor<double>(fmt::format("B{}", i), K, N));
        Cs.push_back(create_zero_tensor<double>(fmt::format("C{}", i), M, N));
        Refs.push_back(create_zero_tensor<double>(fmt::format("R{}", i), M, N));
    }
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        for (size_t i = 0; i < BATCH; i++)
            cg::einsum("ik;kj->ij", &Refs[i], As[i], Bs[i]);
        g.execute();
    }

    cg::Graph graph("batch5");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < BATCH; i++)
            cg::einsum("ik;kj->ij", &Cs[i], As[i], Bs[i]);
    }
    REQUIRE(graph.num_nodes() == BATCH);

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 1);
    REQUIRE(pass.total_batched() == BATCH);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    for (size_t i = 0; i < BATCH; i++)
        require_close(Cs[i], Refs[i]);
}

TEST_CASE("GEMMBatching: single-precision floats batch separately from doubles", "[ComputeGraph][Optimizer][GEMMBatching]") {
    // Two doubles at the same level + two floats — pass should create
    // two batches (one per type).
    constexpr size_t M = 3, K = 3, N = 3;
    auto             Ad1 = create_random_tensor<double>("Ad1", M, K);
    auto             Bd1 = create_random_tensor<double>("Bd1", K, N);
    auto             Cd1 = create_zero_tensor<double>("Cd1", M, N);
    auto             Ad2 = create_random_tensor<double>("Ad2", M, K);
    auto             Bd2 = create_random_tensor<double>("Bd2", K, N);
    auto             Cd2 = create_zero_tensor<double>("Cd2", M, N);

    auto Af1 = create_random_tensor<float>("Af1", M, K);
    auto Bf1 = create_random_tensor<float>("Bf1", K, N);
    auto Cf1 = create_zero_tensor<float>("Cf1", M, N);
    auto Af2 = create_random_tensor<float>("Af2", M, K);
    auto Bf2 = create_random_tensor<float>("Bf2", K, N);
    auto Cf2 = create_zero_tensor<float>("Cf2", M, N);

    cg::Graph graph("mixed_types");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &Cd1, Ad1, Bd1);
        cg::einsum("ik;kj->ij", &Cd2, Ad2, Bd2);
        cg::einsum("ik;kj->ij", &Cf1, Af1, Bf1);
        cg::einsum("ik;kj->ij", &Cf2, Af2, Bf2);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 2); // doubles and floats each form a batch
    REQUIRE(pass.total_batched() == 4);
    REQUIRE(graph.num_nodes() == 2); // two BatchedGemm nodes

    graph.execute();
    // Quick sanity: first element is nonzero
    REQUIRE(Cd1(0, 0) != 0.0);
    REQUIRE(Cf1(0, 0) != 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Safety: skip when conditions don't match
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GEMMBatching: different dims do not batch", "[ComputeGraph][Optimizer][GEMMBatching]") {
    auto A1 = create_random_tensor<double>("A1", 5, 4);
    auto B1 = create_random_tensor<double>("B1", 4, 3);
    auto C1 = create_zero_tensor<double>("C1", 5, 3);
    auto A2 = create_random_tensor<double>("A2", 6, 4); // different M
    auto B2 = create_random_tensor<double>("B2", 4, 3);
    auto C2 = create_zero_tensor<double>("C2", 6, 3);

    cg::Graph graph("diff_dims");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_batches() == 0);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("GEMMBatching: different alpha/beta do not batch", "[ComputeGraph][Optimizer][GEMMBatching]") {
    constexpr size_t M = 4, K = 4, N = 4;
    auto             A1 = create_random_tensor<double>("A1", M, K);
    auto             B1 = create_random_tensor<double>("B1", K, N);
    auto             C1 = create_zero_tensor<double>("C1", M, N);
    auto             A2 = create_random_tensor<double>("A2", M, K);
    auto             B2 = create_random_tensor<double>("B2", K, N);
    auto             C2 = create_zero_tensor<double>("C2", M, N);

    cg::Graph graph("diff_alpha");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C1, 1.0, A1, B1); // alpha=1, beta=0
        cg::einsum("ik;kj->ij", 0.0, &C2, 2.5, A2, B2); // alpha=2.5
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("GEMMBatching: different trans flags do not batch", "[ComputeGraph][Optimizer][GEMMBatching]") {
    // First einsum: "ik;kj->ij" — trans_a=N, trans_b=N
    // Second einsum: "ki;kj->ij" — trans_a=T (link k is first index of A)
    auto A1 = create_random_tensor<double>("A1", 4, 5);
    auto B1 = create_random_tensor<double>("B1", 5, 3);
    auto C1 = create_zero_tensor<double>("C1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 5, 4); // transposed shape
    auto B2 = create_random_tensor<double>("B2", 5, 3);
    auto C2 = create_zero_tensor<double>("C2", 4, 3);

    cg::Graph graph("diff_trans");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ki;kj->ij", &C2, A2, B2);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("GEMMBatching: dependent GEMMs (chain) do not batch", "[ComputeGraph][Optimizer][GEMMBatching]") {
    // C1 feeds into the second einsum via its A input, so they're at
    // different levels — shouldn't batch (and can't, semantically).
    constexpr size_t M = 3, K = 3, N = 3;
    auto             A  = create_random_tensor<double>("A", M, K);
    auto             B  = create_random_tensor<double>("B", K, N);
    auto             C1 = create_zero_tensor<double>("C1", M, N);
    auto             B2 = create_random_tensor<double>("B2", N, N);
    auto             C2 = create_zero_tensor<double>("C2", M, N);

    cg::Graph graph("chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A, B);
        cg::einsum("ik;kj->ij", &C2, C1, B2); // depends on C1
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_batches() == 0);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("GEMMBatching: 3D einsums don't have a gemm_hint and are skipped", "[ComputeGraph][Optimizer][GEMMBatching]") {
    auto T1 = create_random_tensor<double>("T1", 2, 3, 4);
    auto M1 = create_random_tensor<double>("M1", 4, 5);
    auto C1 = create_zero_tensor<double>("C1", 2, 3, 5);
    auto T2 = create_random_tensor<double>("T2", 2, 3, 4);
    auto M2 = create_random_tensor<double>("M2", 4, 5);
    auto C2 = create_zero_tensor<double>("C2", 2, 3, 5);

    cg::Graph graph("rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pqr;rs->pqs", &C1, T1, M1);
        cg::einsum("pqr;rs->pqs", &C2, T2, M2);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_batches() == 0);
    REQUIRE(graph.num_nodes() == 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Correctness invariants
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GEMMBatching: batched result matches N-separate-GEMMs with nonzero beta", "[ComputeGraph][Optimizer][GEMMBatching]") {
    // beta != 0 exercises the accumulation path, which has caused bugs
    // historically in batched BLAS implementations that zero C
    // incorrectly.
    constexpr size_t M = 4, K = 3, N = 2;
    constexpr size_t BATCH = 3;

    std::vector<Tensor<double, 2>> As, Bs, Cs, Refs;
    for (size_t i = 0; i < BATCH; i++) {
        As.push_back(create_random_tensor<double>(fmt::format("A{}", i), M, K));
        Bs.push_back(create_random_tensor<double>(fmt::format("B{}", i), K, N));
        Cs.push_back(create_random_tensor<double>(fmt::format("C{}", i), M, N)); // pre-filled
        Refs.emplace_back(Cs[i]);                                                // copy of initial C
    }

    // Reference: N-separate GEMMs with same prefactors.
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        for (size_t i = 0; i < BATCH; i++)
            cg::einsum("ik;kj->ij", 0.5, &Refs[i], 2.0, As[i], Bs[i]);
        g.execute();
    }

    cg::Graph graph("beta");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < BATCH; i++)
            cg::einsum("ik;kj->ij", 0.5, &Cs[i], 2.0, As[i], Bs[i]);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 1);

    graph.execute();
    for (size_t i = 0; i < BATCH; i++)
        require_close(Cs[i], Refs[i]);
}

TEST_CASE("GEMMBatching: replay produces consistent results across execute() calls", "[ComputeGraph][Optimizer][GEMMBatching]") {
    constexpr size_t M = 4, K = 4, N = 4;
    auto             A1 = create_random_tensor<double>("A1", M, K);
    auto             B1 = create_random_tensor<double>("B1", K, N);
    auto             A2 = create_random_tensor<double>("A2", M, K);
    auto             B2 = create_random_tensor<double>("B2", K, N);
    auto             C1 = create_zero_tensor<double>("C1", M, N);
    auto             C2 = create_zero_tensor<double>("C2", M, N);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }
    auto [modified, _p] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);

    graph.execute();
    auto snap1 = Tensor<double, 2>(C1);
    auto snap2 = Tensor<double, 2>(C2);

    C1.zero();
    C2.zero();
    graph.execute();
    require_close(C1, snap1);
    require_close(C2, snap2);
}
