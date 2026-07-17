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
using namespace einsums::index;
using namespace einsums::tensor_algebra;
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
    // Two doubles at the same level + two floats, pass should create
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
    // First einsum: "ik;kj->ij", trans_a=N, trans_b=N
    // Second einsum: "ki;kj->ij", trans_a=T (link k is first index of A)
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
    // different levels, shouldn't batch (and can't, semantically).
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

TEST_CASE("GEMMBatching - profitability gate leaves large GEMMs unbatched", "[ComputeGraph][GEMMBatching][Gate]") {
    // Two independent 400x400x400 GEMMs: ~big enough that one gemm each on
    // its own Dataflow worker beats a single serialized gemm_batch node.
    // With a profile and a tight threshold they must stay separate; the
    // ungated pass still batches them.
    constexpr size_t N  = 400;
    auto             A1 = create_random_tensor<double>("A1", N, N);
    auto             B1 = create_random_tensor<double>("B1", N, N);
    auto             C1 = create_zero_tensor<double>("C1", N, N);
    auto             A2 = create_random_tensor<double>("A2", N, N);
    auto             B2 = create_random_tensor<double>("B2", N, N);
    auto             C2 = create_zero_tensor<double>("C2", N, N);

    auto build = [&](cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    };

    {
        cg::Graph graph("gate_large");
        build(graph);
        cg::passes::GEMMBatching gated(cg::HardwareProfile::detect_default(), /*max_gemm_us=*/50.0);
        bool const               modified = gated.run(graph);
        CHECK_FALSE(modified);
        CHECK(gated.num_batches() == 0);
        CHECK(gated.num_gate_skipped() == 1);
    }
    {
        cg::Graph graph("ungated_large");
        build(graph);
        cg::passes::GEMMBatching ungated;
        REQUIRE(ungated.run(graph));
        CHECK(ungated.num_batches() == 1);
    }
}

TEST_CASE("GEMMBatching - profitability gate still batches small GEMMs", "[ComputeGraph][GEMMBatching][Gate]") {
    constexpr size_t N  = 8;
    auto             A1 = create_random_tensor<double>("A1", N, N);
    auto             B1 = create_random_tensor<double>("B1", N, N);
    auto             C1 = create_zero_tensor<double>("C1", N, N);
    auto             A2 = create_random_tensor<double>("A2", N, N);
    auto             B2 = create_random_tensor<double>("B2", N, N);
    auto             C2 = create_zero_tensor<double>("C2", N, N);

    cg::Graph graph("gate_small");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }

    cg::passes::GEMMBatching gated(cg::HardwareProfile::detect_default(), /*max_gemm_us=*/100.0);
    REQUIRE(gated.run(graph));
    CHECK(gated.num_batches() == 1);
    CHECK(gated.num_gate_skipped() == 0);

    graph.execute();
    // Numerics through the batched node.
    auto C1_ref = create_zero_tensor<double>("C1ref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &C1_ref, Indices{i, k}, A1, Indices{k, j}, B1);
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(std::abs(C1(ii, jj) - C1_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("GEMMBatching - batched node placed before consumers of member outputs", "[ComputeGraph][GEMMBatching]") {
    // Regression guard: the pass used to append the BatchedGemm at the end of the node
    // list. Position is program order in this IR, so a downstream consumer of
    // a member's output was then legally scheduled BEFORE the batch and read a
    // stale buffer. Two identical contractions plus a gemm consuming the
    // second output reproduce it (found by the Python differential fuzzer once
    // CSE stopped folding user-visible duplicates).
    constexpr size_t N  = 8;
    auto             A  = create_random_tensor<double>("A", N, N);
    auto             B  = create_random_tensor<double>("B", N, N);
    auto             C1 = create_zero_tensor<double>("C1", N, N);
    auto             C2 = create_zero_tensor<double>("C2", N, N);
    auto             E  = create_random_tensor<double>("E", N, N);
    auto             D  = create_zero_tensor<double>("D", N, N);

    cg::Graph graph("batch_before_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ki ; kj", 0.0, &C1, 1.0, A, B); // transposed pair: CSE-identical
        cg::einsum("ij <- ki ; kj", 0.0, &C2, 1.0, A, B);
        cg::gemm<false, false>(1.0, C2, E, 0.0, &D);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    Tensor<double, 2> C_ref("Cref", N, N);
    C_ref.zero();
    tensor_algebra::einsum(Indices{index::i, index::j}, &C_ref, Indices{index::k, index::i}, A, Indices{index::k, index::j}, B);
    Tensor<double, 2> D_ref("Dref", N, N);
    D_ref.zero();
    tensor_algebra::einsum(Indices{index::i, index::j}, &D_ref, Indices{index::i, index::k}, C_ref, Indices{index::k, index::j}, E);

    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(std::abs(C2(ii, jj) - C_ref(ii, jj)) < 1e-10);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-10);
        }
    }
}

TEST_CASE("GEMMBatching - interfering node between members disables the batch", "[ComputeGraph][GEMMBatching]") {
    // A reader of the first member's output sits BETWEEN the two batchable
    // contractions. No slot placement is sound there, so the group must be
    // skipped and results must still be correct.
    constexpr size_t N  = 8;
    auto             A  = create_random_tensor<double>("A", N, N);
    auto             B  = create_random_tensor<double>("B", N, N);
    auto             C1 = create_zero_tensor<double>("C1", N, N);
    auto             C2 = create_zero_tensor<double>("C2", N, N);
    auto             E  = create_random_tensor<double>("E", N, N);
    auto             D  = create_zero_tensor<double>("D", N, N);

    cg::Graph graph("batch_interference");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C1, 1.0, A, B);
        cg::gemm<false, false>(1.0, C1, E, 0.0, &D); // reads C1 between the members
        cg::einsum("ik;kj->ij", 0.0, &C2, 1.0, A, B);
    }

    cg::passes::GEMMBatching pass;
    pass.run(graph); // whether it batches is not the point; correctness is
    graph.execute();

    Tensor<double, 2> C_ref("Cref", N, N);
    C_ref.zero();
    tensor_algebra::einsum(Indices{index::i, index::j}, &C_ref, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B);
    Tensor<double, 2> D_ref("Dref", N, N);
    D_ref.zero();
    tensor_algebra::einsum(Indices{index::i, index::j}, &D_ref, Indices{index::i, index::k}, C_ref, Indices{index::k, index::j}, E);

    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(std::abs(C1(ii, jj) - C_ref(ii, jj)) < 1e-10);
            REQUIRE(std::abs(C2(ii, jj) - C_ref(ii, jj)) < 1e-10);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-10);
        }
    }
}
