//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Graph-path parity tests for eager-dispatcher special cases that the
// TensorAlgebra unit suite covers but the ComputeGraph suite historically did
// not: mixed-operand-dtype einsum, repeated-index (Hadamard/diagonal)
// contractions, the Khatri-Rao pattern, and the #283 non-contiguous
// outer-product targets. Each case computes the eager result as the oracle
// and executes the same contraction through graph capture.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

template <typename T, size_t Rank>
void require_close(Tensor<T, Rank> const &got, Tensor<T, Rank> const &want, double tol = 1e-10) {
    auto const n = got.size();
    for (size_t flat = 0; flat < n; ++flat) {
        REQUIRE(std::abs(got.data()[flat] - want.data()[flat]) <= tol * (1.0 + std::abs(want.data()[flat])));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Gap 1: mixed-operand-dtype einsum (TensorAlgebra/MixedPrecision.cpp parity)
// ---------------------------------------------------------------------------

TEST_CASE("cg parity - mixed-dtype einsum is not yet expressible through capture", "[ComputeGraph][EagerParity][mixed-precision]") {
    // The eager dispatcher supports einsum with different scalar types per
    // operand (TensorAlgebra/MixedPrecision.cpp: d<-f*d, f<-d*d, cd<-cd*d).
    // cg::einsum (Operations.hpp) constrains A/B/C to a single ValueType, so
    // the graph path cannot express those contractions at all. These
    // placeholder records that CURRENT state; when the constraint is lifted,
    // replace it with real graph-vs-eager parity checks over the
    // MixedPrecision.cpp dtype combinations (d<-f*d, f<-d*d, cd<-cd*d).
    SUCCEED("mixed-dtype einsum is rejected at the capture boundary; eager parity is untestable until the feature exists");
}

// ---------------------------------------------------------------------------
// Gap 2: repeated-index / diagonal einsum (TensorAlgebra/Hadamard.cpp parity)
//
// These found bug-1023 on first run (2026-07-17): string_einsum's fast
// paths classified repeated-letter specs as outer products (empty link
// set) and silently computed wrong values, and even the generic loop
// stored only the FIRST position of each letter per operand. Fixed by
// routing repeated-letter specs to the (now repeat-aware) generic loop.
// ---------------------------------------------------------------------------

TEST_CASE("cg parity - Hadamard diagonal outer ij<-ii;jj", "[ComputeGraph][EagerParity][hadamard]") {
    constexpr size_t N = 5;
    auto             A = create_random_tensor<double>("A", N, N);
    auto             B = create_random_tensor<double>("B", N, N);

    auto C_eager = create_zero_tensor<double>("Ce", N, N);
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_eager, Indices{i, i}, A, Indices{j, j}, B));

    auto      C_graph = create_zero_tensor<double>("Cg", N, N);
    cg::Graph graph("hadamard_ii_jj");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_NOTHROW(cg::einsum("ij <- ii ; jj", &C_graph, A, B));
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

TEST_CASE("cg parity - Hadamard rank-3 operands ij<-iij;jji", "[ComputeGraph][EagerParity][hadamard]") {
    constexpr size_t N = 4;
    auto             A = create_random_tensor<double>("A", N, N, N);
    auto             B = create_random_tensor<double>("B", N, N, N);

    auto C_eager = create_zero_tensor<double>("Ce", N, N);
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_eager, Indices{i, i, j}, A, Indices{j, j, i}, B));

    auto      C_graph = create_zero_tensor<double>("Cg", N, N);
    cg::Graph graph("hadamard_iij_jji");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_NOTHROW(cg::einsum("ij <- iij ; jji", &C_graph, A, B));
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

TEST_CASE("cg parity - Hadamard repeated output index iji<-iji;jij", "[ComputeGraph][EagerParity][hadamard]") {
    constexpr size_t N = 4;
    auto             A = create_random_tensor<double>("A", N, N, N);
    auto             B = create_random_tensor<double>("B", N, N, N);

    auto C_eager = create_zero_tensor<double>("Ce", N, N, N);
    REQUIRE_NOTHROW(einsum(Indices{i, j, i}, &C_eager, Indices{i, j, i}, A, Indices{j, i, j}, B));

    auto      C_graph = create_zero_tensor<double>("Cg", N, N, N);
    cg::Graph graph("hadamard_iji");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_NOTHROW(cg::einsum("iji <- iji ; jij", &C_graph, A, B));
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

TEST_CASE("cg parity - Hadamard diagonal accumulation ii<-ijk;jik", "[ComputeGraph][EagerParity][hadamard]") {
    constexpr size_t N = 4;
    auto             A = create_random_tensor<double>("A", N, N, N);
    auto             B = create_random_tensor<double>("B", N, N, N);

    auto C_eager = create_zero_tensor<double>("Ce", N, N);
    REQUIRE_NOTHROW(einsum(Indices{i, i}, &C_eager, Indices{i, j, k}, A, Indices{j, i, k}, B));

    auto      C_graph = create_zero_tensor<double>("Cg", N, N);
    cg::Graph graph("hadamard_ii_sum");
    {
        cg::CaptureGuard const guard(graph);
        REQUIRE_NOTHROW(cg::einsum("ii <- ijk ; jik", &C_graph, A, B));
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

// ---------------------------------------------------------------------------
// Gap 3: Khatri-Rao pattern (TensorAlgebra/KhatriRao.cpp parity)
// ---------------------------------------------------------------------------

TEMPLATE_TEST_CASE("cg parity - Khatri-Rao einsum imr<-ir;mr", "[ComputeGraph][EagerParity][khatri-rao]", float, double,
                   std::complex<double>) {
    constexpr size_t I_dim = 4, M_dim = 3, R_dim = 5;
    auto             T_op = create_random_tensor<TestType>("T", I_dim, R_dim);
    auto             U_op = create_random_tensor<TestType>("U", M_dim, R_dim);

    auto C_eager = create_zero_tensor<TestType>("Ce", I_dim, M_dim, R_dim);
    einsum(Indices{I, M, r}, &C_eager, Indices{I, r}, T_op, Indices{M, r}, U_op);

    auto      C_graph = create_zero_tensor<TestType>("Cg", I_dim, M_dim, R_dim);
    cg::Graph graph("khatri_rao");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("imr <- ir ; mr", &C_graph, T_op, U_op);
    }
    graph.execute();

    double const tol = std::is_same_v<TestType, float> ? 1e-5 : 1e-10;
    auto const   n   = C_graph.size();
    for (size_t flat = 0; flat < n; ++flat) {
        REQUIRE(std::abs(C_graph.data()[flat] - C_eager.data()[flat]) <= tol * (1.0 + std::abs(C_eager.data()[flat])));
    }
}

// ---------------------------------------------------------------------------
// Gap 6: #283 outer products with non-contiguous operand indices in the
// output (TensorAlgebra/OuterProduct.cpp sweep parity). The eager path is
// KNOWN WRONG for these orderings (see PR #257 discussion; eager cases are
// tagged [!shouldfail]). FINDING (2026-07-17): the graph path does NOT
// share that bug - its string-einsum lowering routes these through a
// kernel that handles non-contiguous operand orderings correctly, so the
// cases below assert correct results with no shouldfail tag.
// ---------------------------------------------------------------------------

namespace {

// Hand-rolled outer-product oracle: C(perm of a,b indices) = A * B.
template <typename Fill>
Tensor<double, 3> outer3_reference(Tensor<double, 2> const &A, Tensor<double, 1> const &B, Fill fill) {
    Tensor<double, 3> ref{"ref", A.dim(0), B.dim(0), A.dim(1)};
    for (size_t a = 0; a < A.dim(0); ++a)
        for (size_t b = 0; b < B.dim(0); ++b)
            for (size_t c = 0; c < A.dim(1); ++c)
                fill(ref, a, b, c, A(a, c) * B(b));
    return ref;
}

} // namespace

TEST_CASE("cg parity - outer product contiguous control ijk<-ij;k", "[ComputeGraph][EagerParity][outer-product]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 5);

    auto C_eager = create_zero_tensor<double>("Ce", 3, 4, 5);
    einsum(Indices{i, j, k}, &C_eager, Indices{i, j}, A, Indices{k}, B);

    auto      C_graph = create_zero_tensor<double>("Cg", 3, 4, 5);
    cg::Graph graph("outer_contig");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ijk <- ij ; k", &C_graph, A, B);
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

TEST_CASE("cg parity - #283 non-contiguous outer abc<-ac;b", "[ComputeGraph][EagerParity][outer-product]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 5);

    auto ref = outer3_reference(A, B, [](auto &t, size_t a, size_t b, size_t c, double v) { t(a, b, c) = v; });

    auto      C_graph = create_zero_tensor<double>("Cg", 3, 5, 4);
    cg::Graph graph("outer_noncontig_acb");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("abc <- ac ; b", &C_graph, A, B);
    }
    graph.execute();

    require_close(C_graph, ref);
}

TEST_CASE("cg parity - #283 non-contiguous outer abcd<-ad;bc", "[ComputeGraph][EagerParity][outer-product]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 5, 2);

    Tensor<double, 4> ref{"ref", 3, 5, 2, 4};
    for (size_t a = 0; a < 3; ++a)
        for (size_t b = 0; b < 5; ++b)
            for (size_t c = 0; c < 2; ++c)
                for (size_t d = 0; d < 4; ++d)
                    ref(a, b, c, d) = A(a, d) * B(b, c);

    auto      C_graph = create_zero_tensor<double>("Cg", 3, 5, 2, 4);
    cg::Graph graph("outer_noncontig_adbc");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("abcd <- ad ; bc", &C_graph, A, B);
    }
    graph.execute();

    require_close(C_graph, ref);
}

// ---------------------------------------------------------------------------
// Output-aliasing policy: C overlapping an input is rejected unless the
// update is provably elementwise (aliased operand's index list identical to
// C's). A and B sharing a buffer is always allowed - inputs are read-only.
// ---------------------------------------------------------------------------

TEST_CASE("cg aliasing - contraction with C as input throws", "[ComputeGraph][EagerParity][aliasing]") {
    auto C = create_random_tensor<double>("C", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    REQUIRE_THROWS_AS(cg::einsum("ij <- ik ; kj", &C, C, B), std::invalid_argument);

    cg::Graph graph("alias_reject");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, C, B);
    }
    REQUIRE_THROWS_AS(graph.execute(), std::invalid_argument);
}

TEST_CASE("cg aliasing - elementwise in-place update is allowed", "[ComputeGraph][EagerParity][aliasing]") {
    auto C = create_random_tensor<double>("C", 3, 3);
    auto D = create_random_tensor<double>("D", 3, 3);

    auto expected = create_zero_tensor<double>("E", 3, 3);
    for (size_t a = 0; a < 3; ++a)
        for (size_t b = 0; b < 3; ++b)
            expected(a, b) = C(a, b) * D(a, b);

    cg::Graph graph("alias_elementwise");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ij ; ij", &C, C, D);
    }
    graph.execute();

    require_close(C, expected);
}

TEST_CASE("cg aliasing - A and B sharing a tensor is allowed", "[ComputeGraph][EagerParity][aliasing]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    auto expected = create_zero_tensor<double>("E", 3, 3);
    einsum(Indices{i, j}, &expected, Indices{i, k}, A, Indices{k, j}, A);

    auto      C = create_zero_tensor<double>("C", 3, 3);
    cg::Graph graph("alias_ab");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, A);
    }
    graph.execute();

    require_close(C, expected);
}

TEST_CASE("eager aliasing - typed-Indices dispatcher matches the policy", "[ComputeGraph][EagerParity][aliasing]") {
    auto C = create_random_tensor<double>("C", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    // Contraction with C as an input: rejected.
    REQUIRE_THROWS_AS(einsum(Indices{i, j}, &C, Indices{i, k}, C, Indices{k, j}, B), std::invalid_argument);

    // Elementwise in-place (identical index lists): allowed and correct.
    auto D        = create_random_tensor<double>("D", 4, 4);
    auto expected = create_zero_tensor<double>("E", 4, 4);
    for (size_t a = 0; a < 4; ++a)
        for (size_t b = 0; b < 4; ++b)
            expected(a, b) = D(a, b) * B(a, b);
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &D, Indices{i, j}, D, Indices{i, j}, B));
    require_close(D, expected);

    // A and B sharing a tensor: always allowed.
    auto A  = create_random_tensor<double>("A", 4, 4);
    auto C2 = create_zero_tensor<double>("C2", 4, 4);
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C2, Indices{i, k}, A, Indices{k, j}, A));
}

// ---------------------------------------------------------------------------
// Dispatch introspection: assert the intended kernel route fired, mirroring
// eager DispatchCoverage.cpp's AlgorithmChoice assertions. A regression that
// silently falls back to the generic loop passes every value test while
// losing orders of magnitude of performance - this is the tripwire.
// ---------------------------------------------------------------------------

TEST_CASE("cg dispatch route - fast paths fire where intended", "[ComputeGraph][EagerParity][dispatch-route]") {
    namespace cgd = einsums::compute_graph::dispatch;

    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C, A, B);
    CHECK(std::string_view{cgd::last_dispatch_route()} == "gemm_direct");

    auto x = create_random_tensor<double>("x", 6);
    auto y = create_random_tensor<double>("y", 6);
    auto G = create_zero_tensor<double>("G", 6, 6);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- i ; j", &G, x, y);
    CHECK(std::string_view{cgd::last_dispatch_route()} == "ger");

    auto D  = create_random_tensor<double>("D", 4, 5);
    auto C2 = create_zero_tensor<double>("C2", 4, 5);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ij ; ij", &C2, C, D);
    CHECK(std::string_view{cgd::last_dispatch_route()} == "direct_product");

    // Repeated letters must take the repeat-aware generic loop.
    auto H = create_zero_tensor<double>("H", 4, 4);
    auto S = create_random_tensor<double>("S", 4, 4);
    auto R = create_random_tensor<double>("R", 4, 4);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ii ; jj", &H, S, R);
    CHECK(std::string_view{cgd::last_dispatch_route()} == "generic_loop_repeated_indices");

    // Zero-extent input: prefactor-only path.
    auto Ez = create_random_tensor<double>("Ez", size_t{2}, size_t{0});
    auto Bz = create_random_tensor<double>("Bz", size_t{0}, size_t{3});
    auto Cz = create_zero_tensor<double>("Cz", 2, 3);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &Cz, Ez, Bz);
    CHECK(std::string_view{cgd::last_dispatch_route()} == "empty_input_scale_only");
}

// ---------------------------------------------------------------------------
// SortGemmExpanded parity: the batch-scrambled and combined-conjugation
// sort-gemm shapes, pinned exactly (the fuzzers cover the space
// statistically but not these orderings).
// ---------------------------------------------------------------------------

TEST_CASE("cg parity - sort-gemm batch scrambled pilj<-pjki;plk", "[ComputeGraph][EagerParity][sort-gemm]") {
    size_t const dp = 3, di = 4, dj = 5, dk = 6, dl = 3;
    auto         A = create_random_tensor<double>("A", dp, dj, dk, di);
    auto         B = create_random_tensor<double>("B", dp, dl, dk);

    auto C_eager = create_zero_tensor<double>("Ce", dp, di, dl, dj);
    einsum(Indices{p, i, l, j}, &C_eager, Indices{p, j, k, i}, A, Indices{p, l, k}, B);

    auto      C_graph = create_zero_tensor<double>("Cg", dp, di, dl, dj);
    cg::Graph graph("sort_gemm_scrambled");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pilj <- pjki ; plk", &C_graph, A, B);
    }
    graph.execute();

    require_close(C_graph, C_eager);
}

TEST_CASE("cg parity - sort-gemm combined conjugation ilj<-conj(jki);conj(lk)", "[ComputeGraph][EagerParity][sort-gemm]") {
    using T         = std::complex<double>;
    size_t const di = 3, dj = 4, dk = 5, dl = 3;
    auto         A = create_random_tensor<T>("A", dj, dk, di);
    auto         B = create_random_tensor<T>("B", dl, dk);

    auto C_eager = create_zero_tensor<T>("Ce", di, dl, dj);
    einsum<true, true>(T{0.0}, Indices{i, l, j}, &C_eager, T{1.0}, Indices{j, k, i}, A, Indices{l, k}, B);

    auto      C_graph = create_zero_tensor<T>("Cg", di, dl, dj);
    cg::Graph graph("sort_gemm_conj");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ilj <- conj(jki) ; conj(lk)", &C_graph, A, B);
    }
    graph.execute();

    auto const n = C_graph.size();
    for (size_t flat = 0; flat < n; ++flat) {
        REQUIRE(std::abs(C_graph.data()[flat] - C_eager.data()[flat]) <= 1e-10 * (1.0 + std::abs(C_eager.data()[flat])));
    }
}

// ---------------------------------------------------------------------------
// Rank-0 scalar tensor OPERANDS (scalar output is covered by the dot tests).
// ---------------------------------------------------------------------------

TEST_CASE("cg parity - rank-0 einsum operands are not expressible in either API", "[ComputeGraph][EagerParity][rank0]") {
    // Tensor<T, 0> as an einsum INPUT does not compile on the eager path
    // (rank-0 lacks is_totally_vectorable; empty index tuples hit
    // std::tuple<> out-of-bounds access in the dispatcher) and therefore has
    // no oracle to test the graph path against. Scalar multiplication is
    // expressed via scale()/prefactors instead. This placeholder records the
    // shared limitation; if rank-0 operands ever become supported eagerly,
    // replace it with a graph-vs-eager parity check.
    SUCCEED("rank-0 einsum operands are rejected at compile time by both the eager and graph APIs");
}

TEST_CASE("cg parity - smart-pointer operands are eager-only", "[ComputeGraph][EagerParity][smart-pointer]") {
    // The eager dispatcher auto-derefs shared_ptr/unique_ptr operands in
    // every C/A/B combination (TensorAlgebra SharedPointer.cpp /
    // UniquePointer.cpp); cg::einsum constrains on tensor concepts and
    // rejects smart pointers at compile time. This is deliberate: the graph
    // tracks tensors by identity and lifetime (TensorLifetime.cpp), and
    // owning-pointer operands would bypass that tracking. Callers capture
    // the dereferenced tensor instead. Placeholder records the intended
    // divergence; the eager smart-pointer path keeps its own test coverage.
    auto sp = std::make_shared<Tensor<double, 2>>(create_random_tensor<double>("sp", 3, 3));
    auto B  = create_random_tensor<double>("B", 3, 3);
    auto C  = create_zero_tensor<double>("C", 3, 3);

    // Eager: smart-pointer operand works.
    auto C_eager = create_zero_tensor<double>("Ce", 3, 3);
    REQUIRE_NOTHROW(einsum(Indices{i, j}, &C_eager, Indices{i, k}, sp, Indices{k, j}, B));

    // Graph: capture the DEREFERENCED tensor - the supported spelling.
    cg::Graph graph("smart_ptr_deref");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, *sp, B);
    }
    graph.execute();

    require_close(C, C_eager);
}
