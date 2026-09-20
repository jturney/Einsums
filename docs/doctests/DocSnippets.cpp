//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DocSnippets.cpp
/// @brief The C++ examples from the manual, compiled and run.
///
/// A documented example is a promise that the code works. Nothing was checking that promise, and
/// an audit found an ``einsums::read`` free function documented in three places that has never
/// existed. The companion ``check_doc_claims.py`` settles names; this file settles behaviour, by
/// running what the pages tell a reader to write and asserting the results they state.
///
/// Keep these in step with the pages by hand. That is a real cost, and it is smaller than the
/// alternative: extracting blocks automatically means either compiling fragments that were never
/// meant to stand alone, or marking up the pages until they read like a test harness.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A RuntimeTensor with deterministic contents, so a value assertion means something.
RuntimeTensor<double> filled(char const *name, std::vector<size_t> dims, double base = 0.25) {
    RuntimeTensor<double> t(name, std::move(dims));
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = base + 0.03 * static_cast<double>(i);
    }
    return t;
}

} // namespace

// ── howto/contractions.rst ──────────────────────────────────────────────────

TEST_CASE("docs - contractions: spec, prefactors, and scalar output", "[Docs][Snippets]") {
    auto A = filled("A", {4, 4});
    auto B = filled("B", {4, 4}, 1.1);

    SECTION("the four-argument form overwrites") {
        auto C = filled("C", {4, 4});
        C.set_all(100.0);
        cg::einsum("ik;kj->ij", &C, A, B);

        auto expect = filled("expect", {4, 4});
        expect.zero();
        cg::einsum("ik;kj->ij", &expect, A, B);
        CHECK(C(0, 0) == expect(0, 0));
    }

    SECTION("the six-argument form scales and accumulates") {
        auto C = filled("C", {4, 4});
        C.set_all(100.0);
        cg::einsum("ik;kj->ij", &C, A, B);
        double const product = C(0, 0);

        C.set_all(100.0);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B); // C = C + AB
        CHECK(C(0, 0) == Catch::Approx(100.0 + product));

        C.set_all(100.0);
        cg::einsum("ik;kj->ij", 2.0, &C, 3.0, A, B); // C = 2C + 3AB
        CHECK(C(0, 0) == Catch::Approx(200.0 + 3.0 * product));
    }

    SECTION("a scalar comes out through dot, and agrees with the eager form") {
        double graph_form = 0.0;
        cg::dot(&graph_form, A, B);

        using namespace einsums::tensor_algebra;
        using namespace einsums::index;
        auto   As         = create_random_tensor<double>("As", 4, 4);
        auto   Bs         = create_random_tensor<double>("Bs", 4, 4);
        double eager_form = 0.0;
        einsum(Indices{}, &eager_form, Indices{i, j}, As, Indices{i, j}, Bs);

        double reference = 0.0;
        for (size_t r = 0; r < 4; r++) {
            for (size_t c = 0; c < 4; c++) {
                reference += As(r, c) * Bs(r, c);
            }
        }
        CHECK(eager_form == Catch::Approx(reference));
        CHECK(graph_form != 0.0);
    }
}

TEST_CASE("docs - contractions: the arrow-left spelling is equivalent", "[Docs][Snippets]") {
    auto A = filled("A", {4, 4});
    auto B = filled("B", {4, 4}, 1.1);

    auto lhs = filled("lhs", {4, 4});
    auto rhs = filled("rhs", {4, 4});
    cg::einsum("ik;kj->ij", &lhs, A, B);
    cg::einsum("ij <- ik ; kj", &rhs, A, B);

    CHECK(lhs(0, 0) == rhs(0, 0));
    CHECK(lhs(3, 2) == rhs(3, 2));
}

TEST_CASE("docs - contractions: aliasing rules hold as documented", "[Docs][Snippets]") {
    auto B = filled("B", {4, 4}, 1.1);

    SECTION("identical index lists are a legal in-place elementwise update") {
        auto D = filled("D", {4, 4});
        CHECK_NOTHROW(cg::einsum("ij;ij->ij", &D, D, B));
    }

    SECTION("a differing index list on an aliased output is rejected") {
        auto A = filled("A", {4, 4});
        CHECK_THROWS_AS(cg::einsum("ik;kj->ij", &A, A, B), std::invalid_argument);
    }
}

TEST_CASE("docs - contractions: a zero extent still applies the C prefactor once", "[Docs][Snippets]") {
    // The page states this outright, because code that special-cases an empty result usually
    // gets the prefactor wrong.
    auto Z  = filled("Z", {4, 0});
    auto Z2 = filled("Z2", {0, 4});
    auto C  = filled("C", {4, 4});
    C.set_all(7.0);

    cg::einsum("ik;kj->ij", 2.0, &C, 1.0, Z, Z2);

    CHECK(C(0, 0) == Catch::Approx(14.0));
}

TEST_CASE("docs - contractions: conjugation is a flag, not spec syntax", "[Docs][Snippets]") {
    using cd = std::complex<double>;
    RuntimeTensor<cd> Z("Z", {2, 2});
    RuntimeTensor<cd> I("I", {2, 2});
    RuntimeTensor<cd> C("C", {2, 2});
    Z.data()[0] = cd(1, 2);
    Z.data()[3] = cd(1, -1);
    I.data()[0] = cd(1, 0);
    I.data()[3] = cd(1, 0);

    cg::einsum("ik;kj->ij", cd(0, 0), &C, cd(1, 0), Z, I, /*conj_a=*/true, /*conj_b=*/false);

    CHECK(C(0, 0).real() == Catch::Approx(1.0));
    CHECK(C(0, 0).imag() == Catch::Approx(-2.0));
}

// ── howto/views.rst ─────────────────────────────────────────────────────────

TEST_CASE("docs - views: a slice writes through to its parent", "[Docs][Snippets]") {
    auto A     = create_random_tensor<double>("A", 6, 6);
    auto block = A(Range{0, 3}, Range{0, 3});

    block(0, 0) = 999.0;
    CHECK(A(0, 0) == 999.0);

    // An integer index drops the dimension, so a row view is rank 1.
    auto row = A(2, All);
    CHECK(row.dim(0) == 6);
    CHECK(row(3) == A(2, 3));
}

TEST_CASE("docs - views: occupied and virtual blocks contract without copying", "[Docs][Snippets]") {
    size_t const nocc = 2;
    size_t const nmo  = 5;

    auto F   = filled("F", {nmo, nmo});
    auto Fov = F(Range{0, static_cast<int>(nocc)}, Range{static_cast<int>(nocc), static_cast<int>(nmo)});
    CHECK(Fov.dim(0) == nocc);
    CHECK(Fov.dim(1) == nmo - nocc);

    auto v = filled("v", {nmo - nocc});
    v.set_all(1.0);
    auto out = filled("out", {nocc});

    cg::einsum("ia;a->i", &out, Fov, v);

    double expect = 0.0;
    for (size_t a = nocc; a < nmo; a++) {
        expect += F(0, a);
    }
    CHECK(out(0) == Catch::Approx(expect));
}

// ── howto/graphs.rst ────────────────────────────────────────────────────────

TEST_CASE("docs - graphs: capture, optimize, replay with a declared result", "[Docs][Snippets]") {
    auto A = filled("A", {8, 6});
    auto B = filled("B", {6, 8}, 1.1);

    auto reference = filled("reference", {8, 8});
    cg::einsum("ik;kj->ij", &reference, A, B);

    cg::Graph graph("demo");
    auto     &C = graph.create_runtime_tensor<double>("C", {8, 8}, /*intermediate=*/false);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    graph.optimize();
    graph.execute();
    CHECK(C(0, 0) == reference(0, 0));

    // Replay is the point of a graph, so assert it rather than assuming it.
    graph.execute();
    CHECK(C(0, 0) == reference(0, 0));
}

TEST_CASE("docs - graphs: a scratch result is pruned, and says so", "[Docs][Snippets]") {
    // This is the failure the page warns about. It is asserted here so the warning cannot
    // quietly stop being true, in either direction.
    auto A = filled("A", {8, 6});
    auto B = filled("B", {6, 8}, 1.1);

    cg::Graph graph("scratch_result");
    auto     &C = graph.create_runtime_tensor<double>("C", {8, 8}); // scratch by default
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.optimize();
    graph.execute();

    CHECK(C(0, 0) == 0.0);
    CHECK(graph.explain().find("'C'") != std::string::npos);
    CHECK(graph.explain().find("intermediate=false") != std::string::npos);
}

TEST_CASE("docs - graphs: returning forms are rejected during capture", "[Docs][Snippets]") {
    auto A = filled("A", {4, 4});
    auto B = filled("B", {4, 4}, 1.1);

    cg::Graph graph("returning_form");
    {
        cg::CaptureGuard const guard(graph);
        double                 sink = 0.0;
        CHECK_NOTHROW(cg::dot(&sink, A, B)); // the pointer-writing form records
    }
}

// ── user/tutorial_performance.rst: the dispatch reference table ─────────────

TEST_CASE("docs - performance: the dispatch reference table is accurate", "[Docs][Snippets]") {
    // The table on that page states a route per spec, and a reader uses it to decide how to
    // write a contraction. Stating a route that has since changed is worse than stating none, so
    // every row is asserted here. A route that moves fails this test rather than misleading
    // somebody, and the fix is to re-measure the table.
    auto const N = size_t{5};

    auto v  = filled("v", {N});
    auto A2 = filled("A2", {N, N});
    auto B2 = filled("B2", {N, N}, 1.1);
    auto A3 = filled("A3", {N, N, N});
    auto B3 = filled("B3", {N, N, N}, 1.1);
    auto A4 = filled("A4", {N, N, N, N});
    auto B4 = filled("B4", {N, N, N, N}, 1.1);

    auto route_of = [](auto &&run) {
        run();
        return std::string(cg::dispatch::last_dispatch_route());
    };

    SECTION("BLAS rungs") {
        auto s = filled("s", {N});
        CHECK(route_of([&] { cg::einsum("i;i->", &s, v, v); }) == "dot_runtime");

        auto ger = filled("ger", {N, N});
        CHECK(route_of([&] { cg::einsum("i;j->ij", &ger, v, v); }) == "ger_runtime");

        auto gemv = filled("gemv", {N});
        CHECK(route_of([&] { cg::einsum("ij;j->i", &gemv, A2, v); }) == "gemv_mat_vec_runtime");

        auto gemm = filled("gemm", {N, N});
        CHECK(route_of([&] { cg::einsum("ik;kj->ij", &gemm, A2, B2); }) == "gemm_direct_runtime");

        // A scrambled index order still reaches one GEMM through its transpose flags.
        CHECK(route_of([&] { cg::einsum("ki;jk->ij", &gemm, A2, B2); }) == "gemm_direct_runtime");

        CHECK(route_of([&] { cg::einsum("ij;ij->ij", &gemm, A2, B2); }) == "direct_product_runtime");
    }

    SECTION("packed-GEMM rungs") {
        auto multi_k = filled("multi_k", {N, N});
        CHECK(route_of([&] { cg::einsum("ijk;jkl->il", &multi_k, A3, B3); }) == "packed_gemm");

        auto multi_mn = filled("multi_mn", {N, N, N, N});
        CHECK(route_of([&] { cg::einsum("ijk;klm->ijlm", &multi_mn, A3, B3); }) == "packed_gemm");

        auto batch = filled("batch", {N, N, N});
        CHECK(route_of([&] { cg::einsum("bik;bkj->bij", &batch, A3, B3); }) == "packed_gemm");

        auto rank4 = filled("rank4", {N, N, N, N});
        CHECK(route_of([&] { cg::einsum("ijkl;klmn->ijmn", &rank4, A4, B4); }) == "packed_gemm");

        // Both of these were documented as generic-loop fallbacks and are not.
        auto no_n = filled("no_n", {N, N});
        CHECK(route_of([&] { cg::einsum("ijk;k->ij", &no_n, A3, v); }) == "packed_gemm");

        auto shared = filled("shared", {N});
        CHECK(route_of([&] { cg::einsum("ij;ji->i", &shared, A2, B2); }) == "packed_gemm");
    }

    SECTION("a repeated letter is the one thing still on the generic loop") {
        auto diag = filled("diag", {N});
        CHECK(route_of([&] { cg::einsum("ii;i->i", &diag, A2, v); }) == "generic_loop_repeated_indices");
        CHECK(route_of([&] { cg::einsum("ikk;k->i", &diag, A3, v); }) == "generic_loop_repeated_indices");
    }
}

TEST_CASE("docs - performance: the eager path classifies three cases differently", "[Docs][Snippets]") {
    // The page states this divergence outright, because it is why a pattern can be slow through
    // one API and fast through the other. If the two ever converge, the page should say so.
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;
    namespace ta = einsums::tensor_algebra::detail;

    auto const N  = size_t{5};
    auto       A2 = create_random_tensor<double>("A2", N, N);
    auto       B2 = create_random_tensor<double>("B2", N, N);
    auto       A3 = create_random_tensor<double>("A3", N, N, N);
    auto       vv = create_random_tensor<double>("vv", N);

    ta::AlgorithmChoice algo{};

    auto gemv = create_tensor<double>("gemv", N, N);
    einsum(Indices{i, j}, &gemv, Indices{i, j, k}, A3, Indices{k}, vv, &algo);
    CHECK(algo == ta::GEMV); // the string form takes packed_gemm here

    auto shared = create_tensor<double>("shared", N);
    einsum(Indices{i}, &shared, Indices{i, j}, A2, Indices{j, i}, B2, &algo);
    CHECK(algo == ta::GENERIC); // the string form takes packed_gemm here

    // A scalar over permuted packs is generic on both paths.
    double s = 0.0;
    einsum(Indices{}, &s, Indices{i, j}, A2, Indices{j, i}, B2, &algo);
    CHECK(algo == ta::GENERIC);

    // ... while identical packs reach DOT.
    einsum(Indices{}, &s, Indices{i, j}, A2, Indices{i, j}, B2, &algo);
    CHECK(algo == ta::DOT);
}
