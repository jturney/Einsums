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
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// A few ulps of the element type per term: the members here sum at most eight order-one
// products, so this is well clear of rounding and far below a wrong term.
template <typename T>
constexpr double tol() {
    return 1000.0 * std::numeric_limits<RemoveComplexT<T>>::epsilon();
}

template <typename T>
void require_close(Tensor<T, 2> const &got, Tensor<T, 2> const &ref) {
    REQUIRE(got.size() == ref.size());
    T const *g = got.data();
    T const *r = ref.data();
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(std::abs(g[i] - r[i]) <= tol<T>() * (1.0 + std::abs(r[i])));
    }
}

template <typename T>
T pf(double re, double im) {
    return testing::prefactor<T>(re, im);
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Basic batching
// ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("GEMMBatching: two independent GEMMs collapse into one BatchedGemm", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t M = 5, K = 4, N = 3;
    auto             A1 = create_random_tensor<T>("A1", M, K);
    auto             B1 = create_random_tensor<T>("B1", K, N);
    auto             A2 = create_random_tensor<T>("A2", M, K);
    auto             B2 = create_random_tensor<T>("B2", K, N);

    auto C1_ref = create_zero_tensor<T>("C1_ref", M, N);
    auto C2_ref = create_zero_tensor<T>("C2_ref", M, N);
    reference_einsum("ij <- ik ; kj", &C1_ref, A1, B1);
    reference_einsum("ij <- ik ; kj", &C2_ref, A2, B2);

    auto      C1 = create_zero_tensor<T>("C1", M, N);
    auto      C2 = create_zero_tensor<T>("C2", M, N);
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

    auto const *d = graph.nodes()[0].op_data.get_if<cg::BatchedGemmDescriptor>();
    REQUIRE(d != nullptr);
    REQUIRE(d->batch_count == 2);
    REQUIRE(std::cmp_equal(d->m, M));
    REQUIRE(std::cmp_equal(d->n, N));
    REQUIRE(std::cmp_equal(d->k, K));
    REQUIRE(d->scalar == cg::blas_scalar_of<T>());

    graph.execute();
    require_close(C1, C1_ref);
    require_close(C2, C2_ref);
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching: five GEMMs all batch together", "[ComputeGraph][Optimizer][GEMMBatching]", testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t M = 4, K = 6, N = 5;
    constexpr size_t BATCH = 5;

    std::vector<Tensor<T, 2>> As, Bs, Cs, Refs;
    As.reserve(BATCH);
    Bs.reserve(BATCH);
    Cs.reserve(BATCH);
    Refs.reserve(BATCH);
    for (size_t i = 0; i < BATCH; i++) {
        As.push_back(create_random_tensor<T>(fmt::format("A{}", i), M, K));
        Bs.push_back(create_random_tensor<T>(fmt::format("B{}", i), K, N));
        Cs.push_back(create_zero_tensor<T>(fmt::format("C{}", i), M, N));
        Refs.push_back(create_zero_tensor<T>(fmt::format("R{}", i), M, N));
        reference_einsum("ij <- ik ; kj", &Refs[i], As[i], Bs[i]);
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

namespace {

// Two same-shaped GEMMs of one element type, with their references.
template <typename T>
struct GemmPair {
    Tensor<T, 2> A1, B1, C1, A2, B2, C2, R1, R2;

    explicit GemmPair(size_t n)
        : A1(create_random_tensor<T>("A1", n, n)), B1(create_random_tensor<T>("B1", n, n)), C1(create_zero_tensor<T>("C1", n, n)),
          A2(create_random_tensor<T>("A2", n, n)), B2(create_random_tensor<T>("B2", n, n)), C2(create_zero_tensor<T>("C2", n, n)),
          R1(create_zero_tensor<T>("R1", n, n)), R2(create_zero_tensor<T>("R2", n, n)) {
        reference_einsum("ij <- ik ; kj", &R1, A1, B1);
        reference_einsum("ij <- ik ; kj", &R2, A2, B2);
    }

    void capture() {
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }

    void check() const {
        require_close(C1, R1);
        require_close(C2, R2);
    }
};

} // namespace

TEST_CASE("GEMMBatching: each element type batches separately", "[ComputeGraph][Optimizer][GEMMBatching]") {
    // Two GEMMs of each of the four element types at one level: the pass must
    // form four batches, one per type, since one gemm_batch call takes one type.
    constexpr size_t               N = 3;
    GemmPair<float>                f(N);
    GemmPair<double>               d(N);
    GemmPair<std::complex<float>>  c(N);
    GemmPair<std::complex<double>> z(N);

    cg::Graph graph("mixed_types");
    {
        cg::CaptureGuard const guard(graph);
        d.capture();
        f.capture();
        z.capture();
        c.capture();
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 4); // one batch per element type
    REQUIRE(pass.total_batched() == 8);
    REQUIRE(graph.num_nodes() == 4); // four BatchedGemm nodes

    graph.execute();
    f.check();
    d.check();
    c.check();
    z.check();
}

// ═══════════════════════════════════════════════════════════════════════════
// Safety: skip when conditions don't match
// ═══════════════════════════════════════════════════════════════════════════

TEMPLATE_LIST_TEST_CASE("GEMMBatching: different dims do not batch", "[ComputeGraph][Optimizer][GEMMBatching]", testing::AllScalarTypes) {
    using T = TestType;
    auto A1 = create_random_tensor<T>("A1", 5, 4);
    auto B1 = create_random_tensor<T>("B1", 4, 3);
    auto C1 = create_zero_tensor<T>("C1", 5, 3);
    auto A2 = create_random_tensor<T>("A2", 6, 4); // different M
    auto B2 = create_random_tensor<T>("B2", 4, 3);
    auto C2 = create_zero_tensor<T>("C2", 6, 3);

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

TEMPLATE_LIST_TEST_CASE("GEMMBatching: different alpha/beta do not batch", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t M = 4, K = 4, N = 4;
    auto             A1 = create_random_tensor<T>("A1", M, K);
    auto             B1 = create_random_tensor<T>("B1", K, N);
    auto             C1 = create_zero_tensor<T>("C1", M, N);
    auto             A2 = create_random_tensor<T>("A2", M, K);
    auto             B2 = create_random_tensor<T>("B2", K, N);
    auto             C2 = create_zero_tensor<T>("C2", M, N);

    SECTION("different alpha") {
        cg::Graph graph("diff_alpha");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", T{0}, &C1, pf<T>(1.0, 0.0), A1, B1);
            cg::einsum("ik;kj->ij", T{0}, &C2, pf<T>(2.5, 0.3), A2, B2);
        }

        auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
        REQUIRE_FALSE(modified);
        REQUIRE(graph.num_nodes() == 2);
    }

    if constexpr (IsComplexV<T>) {
        // One gemm_batch call applies one alpha to every member, so two alphas
        // that agree in their real part and differ in their imaginary part are
        // two groups. A key built on the real part alone would batch them and
        // give the second member the first one's phase.
        SECTION("alphas differing only in the imaginary part") {
            cg::Graph graph("diff_alpha_imag");
            {
                cg::CaptureGuard const guard(graph);
                cg::einsum("ik;kj->ij", T{0}, &C1, pf<T>(2.5, 0.3), A1, B1);
                cg::einsum("ik;kj->ij", T{0}, &C2, pf<T>(2.5, -0.3), A2, B2);
            }

            auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
            REQUIRE_FALSE(modified);
            REQUIRE(graph.num_nodes() == 2);

            graph.execute();
            auto R1 = create_zero_tensor<T>("R1", M, N);
            auto R2 = create_zero_tensor<T>("R2", M, N);
            reference_einsum("ij <- ik ; kj", T{0}, &R1, pf<T>(2.5, 0.3), A1, B1);
            reference_einsum("ij <- ik ; kj", T{0}, &R2, pf<T>(2.5, -0.3), A2, B2);
            require_close(C1, R1);
            require_close(C2, R2);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching: different trans flags do not batch", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    // First einsum: "ik;kj->ij", trans_a=N, trans_b=N
    // Second einsum: "ki;kj->ij", trans_a=T (link k is first index of A)
    using T = TestType;
    auto A1 = create_random_tensor<T>("A1", 4, 5);
    auto B1 = create_random_tensor<T>("B1", 5, 3);
    auto C1 = create_zero_tensor<T>("C1", 4, 3);
    auto A2 = create_random_tensor<T>("A2", 5, 4); // transposed shape
    auto B2 = create_random_tensor<T>("B2", 5, 3);
    auto C2 = create_zero_tensor<T>("C2", 4, 3);

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

TEMPLATE_LIST_TEST_CASE("GEMMBatching: dependent GEMMs (chain) do not batch", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    // C1 feeds into the second einsum via its A input, so they're at
    // different levels, shouldn't batch (and can't, semantically).
    using T            = TestType;
    constexpr size_t M = 3, K = 3, N = 3;
    auto             A  = create_random_tensor<T>("A", M, K);
    auto             B  = create_random_tensor<T>("B", K, N);
    auto             C1 = create_zero_tensor<T>("C1", M, N);
    auto             B2 = create_random_tensor<T>("B2", N, N);
    auto             C2 = create_zero_tensor<T>("C2", M, N);

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

TEMPLATE_LIST_TEST_CASE("GEMMBatching: 3D einsums don't have a gemm_hint and are skipped", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    using T = TestType;
    auto T1 = create_random_tensor<T>("T1", 2, 3, 4);
    auto M1 = create_random_tensor<T>("M1", 4, 5);
    auto C1 = create_zero_tensor<T>("C1", 2, 3, 5);
    auto T2 = create_random_tensor<T>("T2", 2, 3, 4);
    auto M2 = create_random_tensor<T>("M2", 4, 5);
    auto C2 = create_zero_tensor<T>("C2", 2, 3, 5);

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

TEMPLATE_LIST_TEST_CASE("GEMMBatching: batched result matches N-separate-GEMMs with nonzero beta",
                        "[ComputeGraph][Optimizer][GEMMBatching]", testing::AllScalarTypes) {
    // beta != 0 exercises the accumulation path, which has caused bugs
    // historically in batched BLAS implementations that zero C
    // incorrectly. Complex types carry complex alpha and beta, which the
    // batched node must pass through whole rather than by their real parts.
    using T            = TestType;
    constexpr size_t M = 4, K = 3, N = 2;
    constexpr size_t BATCH = 3;
    T const          beta  = pf<T>(0.5, -0.2);
    T const          alpha = pf<T>(2.0, 0.7);

    std::vector<Tensor<T, 2>> As, Bs, Cs, Refs;
    for (size_t i = 0; i < BATCH; i++) {
        As.push_back(create_random_tensor<T>(fmt::format("A{}", i), M, K));
        Bs.push_back(create_random_tensor<T>(fmt::format("B{}", i), K, N));
        Cs.push_back(create_random_tensor<T>(fmt::format("C{}", i), M, N)); // pre-filled
        Refs.emplace_back(Cs[i]);                                           // copy of initial C
        reference_einsum("ij <- ik ; kj", beta, &Refs[i], alpha, As[i], Bs[i]);
    }

    cg::Graph graph("beta");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < BATCH; i++)
            cg::einsum("ik;kj->ij", beta, &Cs[i], alpha, As[i], Bs[i]);
    }

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);
    REQUIRE(pass.num_batches() == 1);

    graph.execute();
    for (size_t i = 0; i < BATCH; i++)
        require_close(Cs[i], Refs[i]);
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching: replay produces consistent results across execute() calls",
                        "[ComputeGraph][Optimizer][GEMMBatching]", testing::AllScalarTypes) {
    using T            = TestType;
    constexpr size_t M = 4, K = 4, N = 4;
    auto             A1 = create_random_tensor<T>("A1", M, K);
    auto             B1 = create_random_tensor<T>("B1", K, N);
    auto             A2 = create_random_tensor<T>("A2", M, K);
    auto             B2 = create_random_tensor<T>("B2", K, N);
    auto             C1 = create_zero_tensor<T>("C1", M, N);
    auto             C2 = create_zero_tensor<T>("C2", M, N);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }
    auto [modified, _p] = graph.apply<cg::passes::GEMMBatching>();
    REQUIRE(modified);

    auto R1 = create_zero_tensor<T>("R1", M, N);
    auto R2 = create_zero_tensor<T>("R2", M, N);
    reference_einsum("ij <- ik ; kj", &R1, A1, B1);
    reference_einsum("ij <- ik ; kj", &R2, A2, B2);

    graph.execute();
    require_close(C1, R1);
    require_close(C2, R2);

    C1.zero();
    C2.zero();
    graph.execute();
    require_close(C1, R1);
    require_close(C2, R2);
}

TEST_CASE("GEMMBatching - profitability gate leaves large GEMMs unbatched", "[ComputeGraph][GEMMBatching][Gate]") {
    // Two independent 400x400x400 GEMMs: ~big enough that one gemm each on
    // its own Dataflow worker beats a single serialized gemm_batch node.
    // With a cost_model and a tight threshold they must stay separate; the
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
        cg::passes::GEMMBatching gated(cg::CostModel::detect_default(), /*max_gemm_us=*/50.0);
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

TEMPLATE_LIST_TEST_CASE("GEMMBatching - profitability gate still batches small GEMMs", "[ComputeGraph][GEMMBatching][Gate]",
                        testing::AllScalarTypes) {
    using T             = TestType;
    constexpr size_t N  = 8;
    auto             A1 = create_random_tensor<T>("A1", N, N);
    auto             B1 = create_random_tensor<T>("B1", N, N);
    auto             C1 = create_zero_tensor<T>("C1", N, N);
    auto             A2 = create_random_tensor<T>("A2", N, N);
    auto             B2 = create_random_tensor<T>("B2", N, N);
    auto             C2 = create_zero_tensor<T>("C2", N, N);

    cg::Graph graph("gate_small");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, A1, B1);
        cg::einsum("ik;kj->ij", &C2, A2, B2);
    }

    cg::passes::GEMMBatching gated(cg::CostModel::detect_default(), /*max_gemm_us=*/100.0);
    REQUIRE(gated.run(graph));
    CHECK(gated.num_batches() == 1);
    CHECK(gated.num_gate_skipped() == 0);

    graph.execute();
    // Numerics through the batched node.
    auto C1_ref = create_zero_tensor<T>("C1ref", N, N);
    auto C2_ref = create_zero_tensor<T>("C2ref", N, N);
    reference_einsum("ij <- ik ; kj", &C1_ref, A1, B1);
    reference_einsum("ij <- ik ; kj", &C2_ref, A2, B2);
    require_close(C1, C1_ref);
    require_close(C2, C2_ref);
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching - batched node placed before consumers of member outputs", "[ComputeGraph][GEMMBatching]",
                        testing::AllScalarTypes) {
    // Regression guard: the pass used to append the BatchedGemm at the end of the node
    // list. Position is program order in this IR, so a downstream consumer of
    // a member's output was then legally scheduled BEFORE the batch and read a
    // stale buffer. Two identical contractions plus a gemm consuming the
    // second output reproduce it (found by the Python differential fuzzer once
    // CSE stopped folding user-visible duplicates).
    using T             = TestType;
    constexpr size_t N  = 8;
    auto             A  = create_random_tensor<T>("A", N, N);
    auto             B  = create_random_tensor<T>("B", N, N);
    auto             C1 = create_zero_tensor<T>("C1", N, N);
    auto             C2 = create_zero_tensor<T>("C2", N, N);
    auto             E  = create_random_tensor<T>("E", N, N);
    auto             D  = create_zero_tensor<T>("D", N, N);

    cg::Graph graph("batch_before_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ki ; kj", T{0}, &C1, T{1}, A, B); // transposed pair: CSE-identical
        cg::einsum("ij <- ki ; kj", T{0}, &C2, T{1}, A, B);
        cg::gemm<false, false>(T{1}, C2, E, T{0}, &D);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    Tensor<T, 2> C_ref("Cref", N, N);
    C_ref.zero();
    reference_einsum("ij <- ki ; kj", &C_ref, A, B);
    Tensor<T, 2> D_ref("Dref", N, N);
    D_ref.zero();
    reference_einsum("ij <- ik ; kj", &D_ref, C_ref, E);

    require_close(C2, C_ref);
    require_close(D, D_ref);
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching - interfering node between members disables the batch", "[ComputeGraph][GEMMBatching]",
                        testing::AllScalarTypes) {
    // A reader of the first member's output sits BETWEEN the two batchable
    // contractions. No slot placement is sound there, so the group must be
    // skipped and results must still be correct.
    using T             = TestType;
    constexpr size_t N  = 8;
    auto             A  = create_random_tensor<T>("A", N, N);
    auto             B  = create_random_tensor<T>("B", N, N);
    auto             C1 = create_zero_tensor<T>("C1", N, N);
    auto             C2 = create_zero_tensor<T>("C2", N, N);
    auto             E  = create_random_tensor<T>("E", N, N);
    auto             D  = create_zero_tensor<T>("D", N, N);

    cg::Graph graph("batch_interference");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", T{0}, &C1, T{1}, A, B);
        cg::gemm<false, false>(T{1}, C1, E, T{0}, &D); // reads C1 between the members
        cg::einsum("ik;kj->ij", T{0}, &C2, T{1}, A, B);
    }

    cg::passes::GEMMBatching pass;
    pass.run(graph); // whether it batches is not the point; correctness is
    graph.execute();

    Tensor<T, 2> C_ref("Cref", N, N);
    C_ref.zero();
    reference_einsum("ij <- ik ; kj", &C_ref, A, B);
    Tensor<T, 2> D_ref("Dref", N, N);
    D_ref.zero();
    reference_einsum("ij <- ik ; kj", &D_ref, C_ref, E);

    require_close(C1, C_ref);
    require_close(C2, C_ref);
    require_close(D, D_ref);
}

// ─────────────────────────────────────────────────────────────────────────────
// batched_gemm_blocked: destinations described, not enumerated
// ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_LIST_TEST_CASE("batched_gemm_blocked matches the list form", "[ComputeGraph][GEMMBatching]", testing::AllScalarTypes) {
    using namespace einsums;
    namespace cg = einsums::compute_graph;
    using T      = TestType;

    constexpr size_t m = 4, k = 6, n = 3, count = 5;

    auto const near = [](T got, T want) { return std::abs(got - want) <= tol<T>() * (1.0 + std::abs(want)); };

    std::vector<RuntimeTensor<T>> a_store, b_store;
    for (size_t i = 0; i < count; ++i) {
        a_store.push_back(create_random_tensor<T>("a", m, k));
        b_store.push_back(create_random_tensor<T>("b", k, n));
    }
    std::vector<RuntimeTensor<T> const *> a_list, b_list;
    for (size_t i = 0; i < count; ++i) {
        a_list.push_back(&a_store[i]);
        b_list.push_back(&b_store[i]);
    }

    // Reference: each member's product by the naive oracle, written into its column block.
    RuntimeTensor<T> ref("ref", std::vector<size_t>{m, n * count});
    ref.zero();
    for (size_t i = 0; i < count; ++i) {
        RuntimeTensor<T> block("block", std::vector<size_t>{m, n});
        block.zero();
        reference_einsum("ij <- ik ; kj", &block, a_store[i], b_store[i]);
        for (size_t e = 0; e < m * n; ++e) {
            ref.data()[i * n * m + e] = block.data()[e];
        }
    }

    // Column block i of an (m, n*count) column-major tensor starts at i*n*m.
    std::vector<size_t> offsets;
    for (size_t i = 0; i < count; ++i) {
        offsets.push_back(i * n * m);
    }

    SECTION("the list form") {
        RuntimeTensor<T> got("got", std::vector<size_t>{m, n * count});
        got.zero();
        std::vector<RuntimeTensorView<T>> views;
        views.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            views.push_back(got(AllT{}, Range{i * n, (i + 1) * n}));
        }
        std::vector<RuntimeTensorView<T> *> c_list;
        for (auto &v : views) {
            c_list.push_back(&v);
        }
        cg::batched_gemm(1.0, a_list, b_list, 0.0, c_list);
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(near(got.data()[i], ref.data()[i]));
        }
    }

    SECTION("eager") {
        RuntimeTensor<T> got("got", std::vector<size_t>{m, n * count});
        got.zero();
        cg::batched_gemm_blocked(1.0, a_list, b_list, 0.0, &got, offsets, m, n);
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(near(got.data()[i], ref.data()[i]));
        }
    }

    SECTION("captured, and it registers ONE output rather than one per member") {
        RuntimeTensor<T> got("got", std::vector<size_t>{m, n * count});
        got.zero();
        cg::Graph g("blocked");
        {
            cg::CaptureGuard guard(g);
            cg::batched_gemm_blocked(1.0, a_list, b_list, 0.0, &got, offsets, m, n);
        }
        REQUIRE(g.num_nodes() == 1);
        g.execute();
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(near(got.data()[i], ref.data()[i]));
        }
    }

    SECTION("beta accumulates, and the offsets need not be in order") {
        RuntimeTensor<T> got("got", std::vector<size_t>{m, n * count});
        T const          fill = pf<T>(1.0, -0.5);
        for (size_t i = 0; i < got.size(); ++i) {
            got.data()[i] = fill;
        }
        std::vector<size_t> const perm{3, 0, 4, 1, 2};
        std::vector<size_t>       shuffled;
        for (size_t p : perm) {
            shuffled.push_back(p * n * m);
        }
        cg::batched_gemm_blocked(1.0, a_list, b_list, 1.0, &got, shuffled, m, n);

        // Member i landed on block perm[i], on top of the fill already there.
        for (size_t i = 0; i < count; ++i) {
            for (size_t c = 0; c < n; ++c) {
                for (size_t r = 0; r < m; ++r) {
                    T const expected = fill + ref.data()[i * n * m + c * m + r];
                    REQUIRE(near(got.data()[perm[i] * n * m + c * m + r], expected));
                }
            }
        }
    }

    SECTION("out-of-range and mismatched shapes are rejected at the call") {
        RuntimeTensor<T> got("got", std::vector<size_t>{m, n * count});
        // A block reaching past the end: the executor only sees a raw pointer,
        // so this has to be caught here or it is silent corruption.
        REQUIRE_THROWS(cg::batched_gemm_blocked(1.0, a_list, b_list, 0.0, &got, offsets, m, n * count));
        // One offset per member.
        std::vector<size_t> short_offsets(offsets.begin(), offsets.end() - 1);
        REQUIRE_THROWS(cg::batched_gemm_blocked(1.0, a_list, b_list, 0.0, &got, short_offsets, m, n));
        // A block taller than the parent's leading dimension would run into the
        // next column rather than staying inside its own.
        REQUIRE_THROWS(cg::batched_gemm_blocked(1.0, a_list, b_list, 0.0, &got, offsets, m + 1, n));
    }
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching: an operand BLAS cannot address as a matrix is left alone", "[ComputeGraph][Optimizer][GEMMBatching]",
                        testing::AllScalarTypes) {
    // A view that drops the LEADING axis of a three-index tensor leaves a rank-two operand whose
    // minor axis steps by the parent's next extent. A GEMM is handed a base pointer and a leading
    // dimension and has no way to say that, so such an operand is not a matrix and a batch that
    // addressed it as one would read the wrong elements. The generic algorithm reads strides and
    // is right either way, which is why the unbatched form of this program has always been
    // correct and only the collapsed one was wrong.
    //
    // Found by the region fuzz's tiling arm once it ran the full default pipeline: the schedule
    // that slices a leading axis writes exactly these operands, several of one shape, at one
    // dependency level, which is the batch's own grouping key.
    using T                 = TestType;
    constexpr size_t Slices = 2, M = 5, K = 4, N = 3;
    auto             A  = create_random_tensor<T>("A", Slices, M, K);
    auto             B1 = create_random_tensor<T>("B1", K, N);
    auto             B2 = create_random_tensor<T>("B2", K, N);

    auto slice0 = A(0, All, All);
    auto slice1 = A(1, All, All);
    REQUIRE(slice0.stride(0) != 1);
    REQUIRE(slice1.stride(0) == slice0.stride(0));

    auto C1_ref = create_zero_tensor<T>("C1_ref", M, N);
    auto C2_ref = create_zero_tensor<T>("C2_ref", M, N);
    reference_einsum("ij <- ik ; kj", &C1_ref, slice0, B1);
    reference_einsum("ij <- ik ; kj", &C2_ref, slice1, B2);

    auto      C1 = create_zero_tensor<T>("C1", M, N);
    auto      C2 = create_zero_tensor<T>("C2", M, N);
    cg::Graph graph("strided_operands");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C1, slice0, B1);
        cg::einsum("ik;kj->ij", &C2, slice1, B2);
    }
    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::GEMMBatching>();
    CHECK_FALSE(modified);
    CHECK(pass.num_batches() == 0);
    CHECK(graph.num_nodes() == 2);

    graph.execute();
    require_close(C1, C1_ref);
    require_close(C2, C2_ref);
}

TEMPLATE_LIST_TEST_CASE("GEMMBatching: members hoisted out of a loop body are batched against the parent's ids",
                        "[ComputeGraph][Optimizer][GEMMBatching]", testing::AllScalarTypes) {
    // The batch resolves its operands through the node's own dataflow lists rather than through
    // the ids the hint recorded at capture, and this is the program where the two part company.
    // Both contractions are loop-invariant, so `LoopInvariantHoisting` lifts them into the parent
    // and remaps the lists they carry, because a body id means nothing in the parent's table. The
    // hint's copy of those ids stays as capture left it, and a batch built from them writes
    // somewhere else entirely: before the fix both destinations came back untouched.
    using T            = TestType;
    constexpr size_t M = 6, K = 2, N = 2;
    auto             A  = create_random_tensor<T>("A", M, K);
    auto             B1 = create_random_tensor<T>("B1", K, N);
    auto             B2 = create_random_tensor<T>("B2", K, N);

    auto C1_ref = create_zero_tensor<T>("C1_ref", M, N);
    auto C2_ref = create_zero_tensor<T>("C2_ref", M, N);
    reference_einsum("ij <- ik ; kj", &C1_ref, A, B1);
    reference_einsum("ij <- ik ; kj", &C2_ref, A, B2);

    auto      C1 = create_zero_tensor<T>("C1", M, N);
    auto      C2 = create_zero_tensor<T>("C2", M, N);
    cg::Graph graph("hoisted_then_batched");
    // Unrelated work in the parent FIRST, so the parent's id counter is ahead of the body's and a
    // body id does not happen to name the same buffer in both tables. Without it the two numbering
    // schemes coincide and the stale ids read the right buffers by accident.
    auto D  = create_random_tensor<T>("D", N, N);
    auto DD = create_zero_tensor<T>("DD", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &DD, D, D);
    }
    auto &body = graph.add_loop("iteration", 1, [](size_t it) { return it < 0; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C1, A, B1);
        cg::einsum("ik;kj->ij", &C2, A, B2);
    }

    cg::PassManager pm;
    pm.add<cg::passes::LoopInvariantHoisting>();
    pm.add<cg::passes::GEMMBatching>();
    REQUIRE(pm.run(graph));

    // The precondition: the hoist really did move both statements up, and the batch really did
    // form in the parent, so the case is about ids that crossed a graph boundary.
    REQUIRE(body.num_nodes() == 0);
    REQUIRE(std::ranges::count_if(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::BatchedGemm; }) == 1);

    graph.execute();
    require_close(C1, C1_ref);
    require_close(C2, C2_ref);
}
