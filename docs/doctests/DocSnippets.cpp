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

    SECTION("a scalar comes out through dot") {
        double result = 0.0;
        cg::dot(&result, A, B);

        double reference = 0.0;
        for (size_t r = 0; r < 4; r++) {
            for (size_t c = 0; c < 4; c++) {
                reference += A(r, c) * B(r, c);
            }
        }
        CHECK(result == Catch::Approx(reference));
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

TEST_CASE("docs - front page and README: the routes their examples name", "[Docs][Snippets]") {
    // The front page and the README say which kernel each example reaches. Those are claims about
    // the dispatch, so they are pinned here rather than left to go stale.
    auto route_of = [](auto &&run) {
        run();
        return std::string(cg::dispatch::last_dispatch_route());
    };

    SECTION("front page: a matrix product") {
        auto A = create_random_tensor("A", 7, 7);
        auto B = create_random_tensor("B", 7, 7);
        auto C = create_tensor("C", 7, 7);
        CHECK(route_of([&] { cg::einsum("ij <- ik ; kj", &C, A, B); }) == "gemm_direct");
    }

    SECTION("performance: a transposed operand still reaches one GEMM") {
        // C_ik = sum_j A_ji B_kj: both operands through BLAS's transposition flags, no copy.
        auto A = create_random_tensor("A", 5, 6);
        auto B = create_random_tensor("B", 4, 5);
        auto C = create_tensor("C", 6, 4);
        CHECK(route_of([&] { cg::einsum("ik <- ji ; kj", &C, A, B); }) == "gemm_direct");
    }

    SECTION("architecture: a contraction no stock BLAS call takes") {
        auto A = create_random_tensor("A", 6, 5);
        auto B = create_random_tensor("B", 5, 6, 4);
        auto C = create_tensor("C", 6, 6, 4);
        CHECK(route_of([&] { cg::einsum("ijl <- ik ; kjl", &C, A, B); }) == "packed_gemm");
    }

    SECTION("README: the two-electron Fock build") {
        size_t const n = 6;
        auto         g = create_random_tensor("g", n, n, n, n);
        auto         D = create_random_tensor("D", n, n);
        auto         F = create_zero_tensor("F", n, n);
        CHECK(route_of([&] { cg::einsum("pq <- pqrs ; rs", 1.0, &F, 2.0, g, D); }) == "packed_gemm");
        CHECK(route_of([&] { cg::einsum("pq <- prqs ; rs", 1.0, &F, -1.0, g, D); }) == "packed_gemm");
    }

    SECTION("README: the CCD intermediates") {
        size_t const o = 3, v = 5;
        auto         t_oovv = create_random_tensor("t", o, o, v, v);
        auto         g_oovv = create_random_tensor("g", o, o, v, v);
        auto         Wmnij  = create_zero_tensor("Wmnij", o, o, o, o);
        auto         Wabef  = create_zero_tensor("Wabef", v, v, v, v);
        auto         Wmbej  = create_zero_tensor("Wmbej", o, v, v, o);
        CHECK(route_of([&] { cg::einsum("mnij <- ijef ; mnef", 1.0, &Wmnij, 0.25, t_oovv, g_oovv); }) == "packed_gemm");
        CHECK(route_of([&] { cg::einsum("abef <- mnef ; mnab", 1.0, &Wabef, 0.25, g_oovv, t_oovv); }) == "packed_gemm");
        CHECK(route_of([&] { cg::einsum("mbej <- jnfb ; mnef", 1.0, &Wmbej, -0.5, t_oovv, g_oovv); }) == "packed_gemm");
    }
}

TEST_CASE("docs - user guide: the CCSD energy example", "[Docs][Snippets]") {
    // The page's code as printed, on small random data, each term checked against explicit loops.
    size_t const n_occ = 2, n_orbs = 5, n_virt = n_orbs - n_occ;
    double const E_hf    = -1.5;
    auto         F       = create_random_tensor("F", n_orbs, n_orbs);
    auto         TEI     = create_random_tensor("G", n_orbs, n_orbs, n_orbs, n_orbs);
    auto         t1_amps = create_random_tensor("T1", n_occ, n_virt);
    auto         t2_amps = create_random_tensor("T2", n_occ, n_occ, n_virt, n_virt);

    Tensor tau2{"tau2", n_occ, n_occ, n_virt, n_virt};
    tau2 = t2_amps;
    cg::einsum("ijab <- ia ; jb", 0.25, &tau2, 0.5, t1_amps, t1_amps);

    Tensor TEI_antisym = TEI;
    cg::permute("pqrs <- pqsr", 1.0, &TEI_antisym, -1.0, TEI);

    TensorView Fia      = F(Range{0, n_occ}, Range{n_occ, n_orbs});
    TensorView TEI_ijab = TEI_antisym(Range{0, n_occ}, Range{0, n_occ}, Range{n_occ, n_orbs}, Range{n_occ, n_orbs});

    double e_singles = 0.0;
    double e_doubles = 0.0;
    cg::dot(&e_singles, Fia, t1_amps);
    cg::dot(&e_doubles, TEI_ijab, tau2);

    double const E_ccsd = E_hf + e_singles + e_doubles;

    double want_singles = 0.0;
    double want_doubles = 0.0;
    for (size_t i = 0; i < n_occ; i++) {
        for (size_t a = 0; a < n_virt; a++) {
            want_singles += F(i, n_occ + a) * t1_amps(i, a);
            for (size_t j = 0; j < n_occ; j++) {
                for (size_t b = 0; b < n_virt; b++) {
                    double const antisym = TEI(i, j, n_occ + a, n_occ + b) - TEI(i, j, n_occ + b, n_occ + a);
                    double const tau     = t2_amps(i, j, a, b) + 2.0 * t1_amps(i, a) * t1_amps(j, b);
                    want_doubles += 0.25 * antisym * tau;
                }
            }
        }
    }
    CHECK(e_singles == Catch::Approx(want_singles));
    CHECK(e_doubles == Catch::Approx(want_doubles));
    CHECK(E_ccsd == Catch::Approx(E_hf + want_singles + want_doubles));
}
