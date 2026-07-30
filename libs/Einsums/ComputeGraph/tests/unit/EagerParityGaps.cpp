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
using namespace einsums::index;
namespace cg = einsums::compute_graph;
namespace ta = einsums::tensor_algebra;

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
// These caught a real defect on their first run: string_einsum's fast
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
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &C_eager, Indices{i, i}, A, Indices{j, j}, B));

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
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &C_eager, Indices{i, i, j}, A, Indices{j, j, i}, B));

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
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j, i}, &C_eager, Indices{i, j, i}, A, Indices{j, i, j}, B));

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
    REQUIRE_NOTHROW(ta::einsum(Indices{i, i}, &C_eager, Indices{i, j, k}, A, Indices{j, i, k}, B));

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
// Lone summed index ("weighted trace"): a letter in exactly one operand,
// absent from C AND from the shared link. The graph/string path handles it -
// string_einsum's has_lone_summed_index guard routes it to the repeat-aware
// generic loop, which sums it. The eager typed-Indices dispatcher
// (TensorAlgebra Backends/Dispatch.hpp) still MIS-HANDLES it, in two modes:
// the empty-link case computes a wrong value (the lone axis is not summed),
// and the shared-link case throws std::out_of_range on a stride access. The
// graph cases assert correctness (guarding the string_einsum fix); the eager
// cases are tagged [!shouldfail] until the eager dispatcher gains the same
// guard - when it does, the run turns red here as a reminder to drop the tag.
// ---------------------------------------------------------------------------

namespace {

// C(i,j) = (sum_k S(i,j,k)) * W(i,j)  -- empty link, lone k summed in A (P1).
Tensor<double, 2> lone_empty_link_reference(Tensor<double, 3> const &S, Tensor<double, 2> const &W) {
    Tensor<double, 2> ref{"ref", S.dim(0), S.dim(1)};
    for (size_t a = 0; a < S.dim(0); ++a)
        for (size_t b = 0; b < S.dim(1); ++b) {
            double s = 0.0;
            for (size_t c = 0; c < S.dim(2); ++c)
                s += S(a, b, c);
            ref(a, b) = s * W(a, b);
        }
    return ref;
}

// C(j,k) = sum_l sum_p A(j,l) * B(p,l,k)  -- shared link l, lone p summed in B.
Tensor<double, 2> link_plus_lone_reference(Tensor<double, 2> const &A, Tensor<double, 3> const &B) {
    Tensor<double, 2> ref{"ref", A.dim(0), B.dim(2)};
    for (size_t jj = 0; jj < A.dim(0); ++jj)
        for (size_t kk = 0; kk < B.dim(2); ++kk) {
            double s = 0.0;
            for (size_t ll = 0; ll < A.dim(1); ++ll)
                for (size_t pp = 0; pp < B.dim(0); ++pp)
                    s += A(jj, ll) * B(pp, ll, kk);
            ref(jj, kk) = s;
        }
    return ref;
}

} // namespace

TEST_CASE("cg parity - lone summed index empty link ij<-ijk;ij", "[ComputeGraph][EagerParity][lone-summed]") {
    auto S   = create_random_tensor<double>("S", 3, 4, 5);
    auto W   = create_random_tensor<double>("W", 3, 4);
    auto ref = lone_empty_link_reference(S, W);

    auto      C_graph = create_zero_tensor<double>("Cg", 3, 4);
    cg::Graph graph("lone_empty_link");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ijk ; ij", &C_graph, S, W);
    }
    graph.execute();

    require_close(C_graph, ref);
}

TEST_CASE("cg parity - lone summed index with shared link jk<-jl;plk", "[ComputeGraph][EagerParity][lone-summed]") {
    auto A   = create_random_tensor<double>("A", 3, 4);
    auto B   = create_random_tensor<double>("B", 2, 4, 5);
    auto ref = link_plus_lone_reference(A, B);

    auto      C_graph = create_zero_tensor<double>("Cg", 3, 5);
    cg::Graph graph("link_plus_lone");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("jk <- jl ; plk", &C_graph, A, B);
    }
    graph.execute();

    require_close(C_graph, ref);
}

TEST_CASE("eager BUG - lone summed index dropped empty link ij<-ijk;ij", "[ComputeGraph][EagerParity][lone-summed][!shouldfail]") {
    // Eager typed-Indices path does not sum the lone k: wrong value. Drop the
    // [!shouldfail] once Backends/Dispatch.hpp gains a lone-summed guard.
    auto S   = create_random_tensor<double>("S", 3, 4, 5);
    auto W   = create_random_tensor<double>("W", 3, 4);
    auto ref = lone_empty_link_reference(S, W);

    auto C_eager = create_zero_tensor<double>("Ce", 3, 4);
    ta::einsum(Indices{i, j}, &C_eager, Indices{i, j, k}, S, Indices{i, j}, W);

    require_close(C_eager, ref);
}

TEST_CASE("eager BUG - lone summed index dropped with shared link jk<-jl;plk", "[ComputeGraph][EagerParity][lone-summed][!shouldfail]") {
    // Eager typed-Indices path throws std::out_of_range on the lone p here
    // (stride access past the operand rank). Drop the [!shouldfail] once the
    // eager dispatcher handles lone summed indices.
    auto A   = create_random_tensor<double>("A", 3, 4);
    auto B   = create_random_tensor<double>("B", 2, 4, 5);
    auto ref = link_plus_lone_reference(A, B);

    auto C_eager = create_zero_tensor<double>("Ce", 3, 5);
    ta::einsum(Indices{j, k}, &C_eager, Indices{j, l}, A, Indices{p, l, k}, B);

    require_close(C_eager, ref);
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
    ta::einsum(Indices{I, M, r}, &C_eager, Indices{I, r}, T_op, Indices{M, r}, U_op);

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
    ta::einsum(Indices{i, j, k}, &C_eager, Indices{i, j}, A, Indices{k}, B);

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
    ta::einsum(Indices{i, j}, &expected, Indices{i, k}, A, Indices{k, j}, A);

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
    REQUIRE_THROWS_AS(ta::einsum(Indices{i, j}, &C, Indices{i, k}, C, Indices{k, j}, B), std::invalid_argument);

    // Elementwise in-place (identical index lists): allowed and correct.
    auto D        = create_random_tensor<double>("D", 4, 4);
    auto expected = create_zero_tensor<double>("E", 4, 4);
    for (size_t a = 0; a < 4; ++a)
        for (size_t b = 0; b < 4; ++b)
            expected(a, b) = D(a, b) * B(a, b);
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &D, Indices{i, j}, D, Indices{i, j}, B));
    require_close(D, expected);

    // A and B sharing a tensor: always allowed.
    auto A  = create_random_tensor<double>("A", 4, 4);
    auto C2 = create_zero_tensor<double>("C2", 4, 4);
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &C2, Indices{i, k}, A, Indices{k, j}, A));
}

// ---------------------------------------------------------------------------
// Dispatch introspection: assert the intended kernel route fired, mirroring
// eager DispatchCoverage.cpp's AlgorithmChoice assertions. A regression that
// silently falls back to the generic loop passes every value test while
// losing orders of magnitude of performance - this is the tripwire.
// ---------------------------------------------------------------------------

TEST_CASE("cg dispatch route - every route in the cascade fires where intended", "[ComputeGraph][EagerParity][dispatch-route]") {
    namespace cgd = einsums::compute_graph::dispatch;

    // last_dispatch_route() is thread-local and set by the call just made.
    auto route = []() { return std::string{cgd::last_dispatch_route()}; };

    // ── Zero-extent quick paths ─────────────────────────────────────────────
    SECTION("empty_output_noop") {
        auto A = create_random_tensor<double>(std::string("A"), size_t{2}, size_t{3});
        auto B = create_random_tensor<double>(std::string("B"), size_t{3}, size_t{0});
        auto C = create_zero_tensor<double>(std::string("C"), size_t{2}, size_t{0});
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A, B);
        CHECK(route() == "empty_output_noop");
    }

    SECTION("empty_input_scale_only") {
        // The explicit std::string keeps the call unambiguous for GCC, which
        // otherwise also considers the (bool row_major, ...) overload via the
        // const char* -> bool conversion.
        auto A = create_random_tensor<double>(std::string("A"), size_t{2}, size_t{0});
        auto B = create_random_tensor<double>(std::string("B"), size_t{0}, size_t{3});
        auto C = create_zero_tensor<double>("C", 2, 3);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A, B);
        CHECK(route() == "empty_input_scale_only");
    }

    // ── Generic-loop-only shapes, claimed before any fast path ──────────────
    SECTION("generic_loop_repeated_indices") {
        auto S = create_random_tensor<double>("S", 4, 4);
        auto R = create_random_tensor<double>("R", 4, 4);
        auto H = create_zero_tensor<double>("H", 4, 4);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ii ; jj", &H, S, R);
        CHECK(route() == "generic_loop_repeated_indices");
    }

    SECTION("generic_loop_lone_summed") {
        // `l` lives in A only and is absent from C and from the links, so it is
        // a single-operand reduction no BLAS or PackedGemm call can express.
        auto A = create_random_tensor<double>("A", 3, 2, 4);
        auto B = create_random_tensor<double>("B", 2, 5);
        auto C = create_zero_tensor<double>("C", 3, 5);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ikl ; kj", &C, A, B);
        CHECK(route() == "generic_loop_lone_summed");
    }

    // ── Typed (compile-time rank) BLAS ladder ───────────────────────────────
    SECTION("dot") {
        auto x = create_random_tensor<double>("x", 6);
        auto y = create_random_tensor<double>("y", 6);
        auto s = create_zero_tensor<double>("s", 1);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- i ; i", &s, x, y);
        CHECK(route() == "dot");

        // The scalar-output spec form is not exercised anywhere else, so pin
        // the value too: a route assertion alone would pass on a path that
        // fires correctly and computes nothing.
        double want = 0.0;
        for (size_t n = 0; n < 6; n++) {
            want += x.data()[n] * y.data()[n];
        }
        CHECK(std::abs(s.data()[0] - want) <= 1e-12 * (1.0 + std::abs(want)));
    }

    SECTION("gemv_mat_vec") {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto x = create_random_tensor<double>("x", 3);
        auto y = create_zero_tensor<double>("y", 4);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("i <- ij ; j", &y, A, x);
        CHECK(route() == "gemv_mat_vec");
    }

    SECTION("gemv_vec_mat") {
        auto x = create_random_tensor<double>("x", 4);
        auto A = create_random_tensor<double>("A", 4, 3);
        auto y = create_zero_tensor<double>("y", 3);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("j <- i ; ij", &y, x, A);
        CHECK(route() == "gemv_vec_mat");
    }

    SECTION("ger") {
        auto x = create_random_tensor<double>("x", 6);
        auto y = create_random_tensor<double>("y", 6);
        auto G = create_zero_tensor<double>("G", 6, 6);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- i ; j", &G, x, y);
        CHECK(route() == "ger");
    }

    SECTION("gemm_direct") {
        auto A = create_random_tensor<double>("A", 4, 3);
        auto B = create_random_tensor<double>("B", 3, 5);
        auto C = create_zero_tensor<double>("C", 4, 5);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A, B);
        CHECK(route() == "gemm_direct");
    }

    SECTION("direct_product") {
        auto A = create_random_tensor<double>("A", 4, 5);
        auto B = create_random_tensor<double>("B", 4, 5);
        auto C = create_zero_tensor<double>("C", 4, 5);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ij ; ij", &C, A, B);
        CHECK(route() == "direct_product");
    }

    // ── Runtime-rank mirror of the same ladder ──────────────────────────────
    // The runtime ladder only fires when ALL THREE operands are runtime-rank;
    // these pin that, and 2.2's mixed typed/runtime gap is a separate test.
    SECTION("dot_runtime") {
        auto                  x_t = create_random_tensor<double>("x", 6);
        auto                  y_t = create_random_tensor<double>("y", 6);
        auto                  s_t = create_zero_tensor<double>("s", 1);
        RuntimeTensor<double> x(x_t), y(y_t), s(s_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- i ; i", &s, x, y);
        CHECK(route() == "dot_runtime");
    }

    SECTION("gemv_mat_vec_runtime") {
        auto                  A_t = create_random_tensor<double>("A", 4, 3);
        auto                  x_t = create_random_tensor<double>("x", 3);
        auto                  y_t = create_zero_tensor<double>("y", 4);
        RuntimeTensor<double> A(A_t), x(x_t), y(y_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("i <- ij ; j", &y, A, x);
        CHECK(route() == "gemv_mat_vec_runtime");
    }

    SECTION("gemv_vec_mat_runtime") {
        auto                  x_t = create_random_tensor<double>("x", 4);
        auto                  A_t = create_random_tensor<double>("A", 4, 3);
        auto                  y_t = create_zero_tensor<double>("y", 3);
        RuntimeTensor<double> x(x_t), A(A_t), y(y_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("j <- i ; ij", &y, x, A);
        CHECK(route() == "gemv_vec_mat_runtime");
    }

    SECTION("ger_runtime") {
        auto                  x_t = create_random_tensor<double>("x", 6);
        auto                  y_t = create_random_tensor<double>("y", 6);
        auto                  G_t = create_zero_tensor<double>("G", 6, 6);
        RuntimeTensor<double> x(x_t), y(y_t), G(G_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- i ; j", &G, x, y);
        CHECK(route() == "ger_runtime");
    }

    SECTION("gemm_direct_runtime") {
        auto                  A_t = create_random_tensor<double>("A", 4, 3);
        auto                  B_t = create_random_tensor<double>("B", 3, 5);
        auto                  C_t = create_zero_tensor<double>("C", 4, 5);
        RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A, B);
        CHECK(route() == "gemm_direct_runtime");
    }

    SECTION("direct_product_runtime") {
        auto                  A_t = create_random_tensor<double>("A", 4, 5);
        auto                  B_t = create_random_tensor<double>("B", 4, 5);
        auto                  C_t = create_zero_tensor<double>("C", 4, 5);
        RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ij ; ij", &C, A, B);
        CHECK(route() == "direct_product_runtime");
    }

    // ── PackedGemm and the floor beneath it ─────────────────────────────────
    SECTION("packed_gemm") {
        // Two M indices and two N indices, so the single-M/N/K deferral to a
        // direct BLAS GEMM does not apply and PackedGemm forms the contraction.
        auto A = create_random_tensor<double>("A", 3, 4, 2);
        auto B = create_random_tensor<double>("B", 2, 5, 6);
        auto C = create_zero_tensor<double>("C", 3, 4, 5, 6);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ijkl <- ijm ; mkl", &C, A, B);
        CHECK(route() == "packed_gemm");
    }

    // ── Shapes with a BLAS mapping that used to land on the generic loop ────
    // Every one of these was measured falling through to the serial odometer
    // loop before the routes below existed; they are the phase-2.2 additions.
    SECTION("scalar-output full contraction, rank 2") {
        auto A = create_random_tensor<double>("A", 4, 5);
        auto B = create_random_tensor<double>("B", 4, 5);
        auto s = create_zero_tensor<double>("s", 1);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- ij ; ij", &s, A, B);
        CHECK(route() == "dot");

        double want = 0.0;
        for (size_t n = 0; n < A.size(); n++) {
            want += A.data()[n] * B.data()[n];
        }
        CHECK(std::abs(s.data()[0] - want) <= 1e-12 * (1.0 + std::abs(want)));
    }

    SECTION("scalar-output full contraction, runtime rank") {
        auto                  A_t = create_random_tensor<double>("A", 4, 5);
        auto                  B_t = create_random_tensor<double>("B", 4, 5);
        auto                  s_t = create_zero_tensor<double>("s", 1);
        RuntimeTensor<double> A(A_t), B(B_t), s(s_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- ij ; ij", &s, A, B);
        CHECK(route() == "dot_runtime");
    }

    SECTION("elementwise at rank 1") {
        auto A = create_random_tensor<double>("A", 7);
        auto B = create_random_tensor<double>("B", 7);
        auto C = create_zero_tensor<double>("C", 7);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("i <- i ; i", &C, A, B);
        CHECK(route() == "direct_product");
        for (size_t n = 0; n < C.size(); n++) {
            CHECK(std::abs(C.data()[n] - A.data()[n] * B.data()[n]) <= 1e-12);
        }
    }

    SECTION("elementwise at rank 3") {
        auto A = create_random_tensor<double>("A", 3, 4, 5);
        auto B = create_random_tensor<double>("B", 3, 4, 5);
        auto C = create_zero_tensor<double>("C", 3, 4, 5);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ijk <- ijk ; ijk", &C, A, B);
        CHECK(route() == "direct_product");
        for (size_t n = 0; n < C.size(); n++) {
            CHECK(std::abs(C.data()[n] - A.data()[n] * B.data()[n]) <= 1e-12);
        }
    }

    SECTION("elementwise at rank 3, runtime rank") {
        auto                  A_t = create_random_tensor<double>("A", 3, 4, 5);
        auto                  B_t = create_random_tensor<double>("B", 3, 4, 5);
        auto                  C_t = create_zero_tensor<double>("C", 3, 4, 5);
        RuntimeTensor<double> A(A_t), B(B_t), C(C_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ijk <- ijk ; ijk", &C, A, B);
        CHECK(route() == "direct_product_runtime");
    }

    SECTION("conjugated full contraction") {
        auto x = create_random_tensor<std::complex<double>>("x", 6);
        auto y = create_random_tensor<std::complex<double>>("y", 6);

        auto s = create_zero_tensor<std::complex<double>>("s", 1);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- conj(i) ; i", &s, x, y);
        CHECK(route() == "true_dot");

        std::complex<double> want{};
        for (size_t n = 0; n < x.size(); n++) {
            want += std::conj(x.data()[n]) * y.data()[n];
        }
        CHECK(std::abs(s.data()[0] - want) <= 1e-12 * (1.0 + std::abs(want)));

        // conj on B instead: sum A * conj(B).
        auto s2 = create_zero_tensor<std::complex<double>>("s2", 1);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- i ; conj(i)", &s2, x, y);
        CHECK(route() == "true_dot");

        std::complex<double> want2{};
        for (size_t n = 0; n < x.size(); n++) {
            want2 += x.data()[n] * std::conj(y.data()[n]);
        }
        CHECK(std::abs(s2.data()[0] - want2) <= 1e-12 * (1.0 + std::abs(want2)));

        // Both conjugated: conj of the plain dot.
        auto s3 = create_zero_tensor<std::complex<double>>("s3", 1);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("<- conj(i) ; conj(i)", &s3, x, y);
        CHECK(route() == "true_dot");

        std::complex<double> want3{};
        for (size_t n = 0; n < x.size(); n++) {
            want3 += std::conj(x.data()[n]) * std::conj(y.data()[n]);
        }
        CHECK(std::abs(s3.data()[0] - want3) <= 1e-12 * (1.0 + std::abs(want3)));
    }

    SECTION("mixed typed/runtime operand triple") {
        // One typed and one runtime-rank operand satisfied neither ladder, so
        // a plain matmul reached PackedGemm - which defers single-M/N/K back
        // to direct BLAS - and ended up on the serial generic loop.
        auto                  A   = create_random_tensor<double>("A", 4, 3);
        auto                  B_t = create_random_tensor<double>("B", 3, 5);
        auto                  C   = create_zero_tensor<double>("C", 4, 5);
        RuntimeTensor<double> B(B_t);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A, B);
        CHECK(route() == "gemm_direct_runtime");

        for (size_t r = 0; r < 4; r++) {
            for (size_t c = 0; c < 5; c++) {
                double want = 0.0;
                for (size_t k = 0; k < 3; k++) {
                    want += A(r, k) * B_t(k, c);
                }
                CHECK(std::abs(C(r, c) - want) <= 1e-12 * (1.0 + std::abs(want)));
            }
        }
    }

    SECTION("conjugated gemv already reaches PackedGemm") {
        // Not a gap: the conjugating gate skips the BLAS ladder, but PackedGemm
        // conjugates natively during packing, so gemm- and gemv-shaped
        // conjugated contractions never saw the generic loop. Pinned so a
        // future change to the conj gate cannot quietly send them there.
        auto A = create_random_tensor<std::complex<double>>("A", 4, 3);
        auto v = create_random_tensor<std::complex<double>>("v", 3);
        auto w = create_zero_tensor<std::complex<double>>("w", 4);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("i <- conj(ij) ; j", &w, A, v);
        CHECK(route() == "packed_gemm");
    }

    SECTION("generic_loop") {
        // A small rank-3 outer product: no link indices, and below the ~4k
        // output elements where PackedGemm's fixed setup starts to pay off, so
        // it declines and the runtime nested loop is what is left.
        auto A = create_random_tensor<double>("A", 2, 3);
        auto B = create_random_tensor<double>("B", 4);
        auto C = create_zero_tensor<double>("C", 2, 3, 4);
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ijk <- ij ; k", &C, A, B);
        CHECK(route() == "generic_loop");
    }
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
    ta::einsum(Indices{p, i, l, j}, &C_eager, Indices{p, j, k, i}, A, Indices{p, l, k}, B);

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
    ta::einsum<true, true>(T{0.0}, Indices{i, l, j}, &C_eager, T{1.0}, Indices{j, k, i}, A, Indices{l, k}, B);

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
    REQUIRE_NOTHROW(ta::einsum(Indices{i, j}, &C_eager, Indices{i, k}, sp, Indices{k, j}, B));

    // Graph: capture the DEREFERENCED tensor - the supported spelling.
    cg::Graph graph("smart_ptr_deref");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, *sp, B);
    }
    graph.execute();

    require_close(C, C_eager);
}
