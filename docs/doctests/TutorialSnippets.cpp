//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file TutorialSnippets.cpp
/// @brief The C++ examples from the user-guide tutorials, compiled and run.
///
/// Converting these pages to RuntimeTensor found nine examples that did not compile as written,
/// including three tensor declarations on the first page a newcomer reads, a reshape constructor
/// called without the name and the move it takes, a one-argument norm that has no such overload,
/// and an eigendecomposition passed real matrices where it requires complex ones. None of that
/// was exotic; nothing had ever compiled them.
///
/// Keep these in step with the pages by hand. Extracting blocks automatically would mean either
/// compiling fragments that were never meant to stand alone, or marking up the prose until it
/// reads like a test harness.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/ComputeGraph/Workspace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
namespace la = einsums::linear_algebra;

// ── user/absolute_beginners.rst ─────────────────────────────────────────────

TEST_CASE("tutorial - beginners: creating tensors of each supported type", "[Docs][Tutorials]") {
    RuntimeTensor<double>               A{"A", {2, 2}};
    RuntimeTensor<float>                B{"B", {2, 2}};
    RuntimeTensor<std::complex<float>>  C{"C", {2, 2}};
    RuntimeTensor<std::complex<double>> D{"D", {2, 2}};

    CHECK(A.rank() == 2);
    CHECK(A.dim(0) == 2);
    // The statically ranked form the page mentions, whose parameters are <T, Rank>.
    Tensor<double, 2> S{"S", 2, 2};
    CHECK(S.dim(1) == 2);
}

TEST_CASE("tutorial - beginners: filling and scalar arithmetic", "[Docs][Tutorials]") {
    RuntimeTensor<double> A{"A", {10, 10}};
    auto                  B = create_random_tensor<double>("B", {10, 10});

    A = B;
    CHECK(A(0, 0) == B(0, 0));

    A.zero();
    CHECK(A(0, 0) == 0.0);

    A.set_all(0.3);
    CHECK(A(0, 0) == Catch::Approx(0.3));

    A = 0.5;
    CHECK(A(0, 0) == Catch::Approx(0.5));

    A += 2;
    A -= 2;
    A *= 2;
    A /= 2;
    CHECK(A(0, 0) == Catch::Approx(0.5));

    // Element-wise between two tensors is a contraction, not an operator.
    cg::einsum("ij;ij->ij", &A, A, B);
    CHECK(A(0, 0) == Catch::Approx(0.5 * B(0, 0)));
}

TEST_CASE("tutorial - beginners: indexing and slicing", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {3, 3});

    CHECK(A(-1, -1) == A(2, 2)); // negative indices wrap
    A(2, 2) = 10.0;
    CHECK(A(2, 2) == 10.0);

    // A Range is half-open, so Range{0, 2} is two entries.
    auto View1 = A(Range{0, 2}, All);
    auto View2 = A(2, All);
    auto View3 = A(All, 2);
    auto View4 = A(Range{1, 3}, Range{0, 2});

    CHECK(View1.dim(0) == 2);
    CHECK(View2.dim(0) == 3);
    CHECK(View3.dim(0) == 3);
    CHECK(View4.dim(0) == 2);
    CHECK(View4.dim(1) == 2);
}

TEST_CASE("tutorial - beginners: reshaping takes an rvalue and a name", "[Docs][Tutorials]") {
    // The page used to call this without either, which does not compile.
    Tensor<double, 3> A{"A", 3, 4, 5};
    Tensor<double, 3> B{std::move(A), "B", 2, 3, 10};
    CHECK(B.dim(0) == 2);
    CHECK(B.dim(2) == 10);

    // A negative extent is a wildcard.
    Tensor<double, 3> C{"C", 3, 4, 5};
    Tensor<double, 2> D{std::move(C), "D", 10, -1};
    CHECK(D.dim(0) == 10);
    CHECK(D.dim(1) == 6);

    Tensor<double, 1> E{"E", 30};
    Tensor<double, 2> F{std::move(E), "F", -1, 10};
    CHECK(F.dim(0) == 3);
    CHECK(F.dim(1) == 10);
}

TEST_CASE("tutorial - beginners: permute scales both sides", "[Docs][Tutorials]") {
    auto                  A = create_random_tensor<double>("A", {3, 4, 5});
    auto                  B = create_random_tensor<double>("B", {5, 4, 3});
    RuntimeTensor<double> C{"C", {5, 4, 3}};
    C = B;

    cg::permute("ijk <- kji", 1.0, &C, 0.5, A);

    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 4; j++) {
            for (size_t k = 0; k < 3; k++) {
                REQUIRE(C(i, j, k) == Catch::Approx(B(i, j, k) + 0.5 * A(k, j, i)));
            }
        }
    }
}

TEST_CASE("tutorial - beginners: eigendecomposition wants complex outputs", "[Docs][Tutorials]") {
    // geev is constrained to a compile-time rank of two, and its eigenvalues and eigenvectors
    // are complex even for a real input. The page passed real matrices, which does not compile.
    using cd        = std::complex<double>;
    auto          M = create_random_tensor<double>("M", 10, 10);
    Tensor<cd, 1> evals{"evals", 10};
    Tensor<cd, 2> lvecs{"lvecs", 10, 10};
    Tensor<cd, 2> rvecs{"rvecs", 10, 10};

    CHECK_NOTHROW(la::geev(&M, &evals, &lvecs, &rvecs));

    // A null pointer skips a set. geev overwrites its input, so this gets a fresh copy.
    auto M2 = create_random_tensor<double>("M2", 10, 10);
    CHECK_NOTHROW(la::geev(&M2, &evals, nullptr, &rvecs));
}

TEST_CASE("tutorial - beginners: the closing program, eager and captured", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {64, 64});
    auto B = create_random_tensor<double>("B", {64, 64});

    auto C = create_zero_tensor<double>("C", {64, 64});
    cg::einsum("ik;kj->ij", &C, A, B);
    cg::scale(0.5, &C);

    cg::Graph graph("scaled product");
    auto     &Cg = graph.create_runtime_tensor<double>("C", {64, 64}, /*intermediate=*/false);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &Cg, A, B);
        cg::scale(0.5, &Cg);
    }
    graph.optimize();
    graph.execute();
    CHECK(Cg(0, 0) == C(0, 0));

    // Replay is the point, so assert it rather than assuming it.
    graph.execute();
    CHECK(Cg(0, 0) == C(0, 0));
}

// ── user/tutorial_tensors.rst ───────────────────────────────────────────────

TEST_CASE("tutorial - tensors: the convenience creators return what the page says", "[Docs][Tutorials]") {
    // Two questions worth pinning, because `auto` hides both answers.
    //
    // The dimension list needs no std::vector spelling: a braced list converts, at every rank,
    // including the single-element case that could plausibly have gone to a variadic overload.
    auto Z  = create_zero_tensor<double>("Z", {4, 4});
    auto R  = create_random_tensor<double>("R", {4, 4});
    auto v  = create_random_tensor<double>("v", {100});
    auto r4 = create_zero_tensor<double>("r4", {4, 4, 4, 4});

    STATIC_CHECK(std::is_same_v<decltype(Z), RuntimeTensor<double>>);
    STATIC_CHECK(std::is_same_v<decltype(R), RuntimeTensor<double>>);
    STATIC_CHECK(std::is_same_v<decltype(v), RuntimeTensor<double>>);
    STATIC_CHECK(std::is_same_v<decltype(r4), RuntimeTensor<double>>);
    CHECK(v.rank() == 1);
    CHECK(r4.rank() == 4);

    // Identity now takes a list too, and returns the same type as the other two.
    auto I = create_identity_tensor<double>("I", {4, 4});
    STATIC_CHECK(std::is_same_v<decltype(I), RuntimeTensor<double>>);
    CHECK(I(0, 0) == 1.0);
    CHECK(I(0, 1) == 0.0);

    // Separate extents still give a statically ranked tensor, for all three of them. That is the
    // distinction the page draws, so it is asserted rather than described.
    auto Is = create_identity_tensor<double>("Is", 4, 4);
    auto Zs = create_zero_tensor<double>("Zs", 4, 4);
    STATIC_CHECK(std::is_same_v<decltype(Is), Tensor<double, 2>>);
    STATIC_CHECK(std::is_same_v<decltype(Zs), Tensor<double, 2>>);

    // An identity need not be square: the diagonal is as long as the shortest axis.
    auto rect = create_identity_tensor<double>("rect", {6, 3});
    CHECK(rect(0, 0) == 1.0);
    CHECK(rect(2, 2) == 1.0);
    CHECK(rect(3, 0) == 0.0);
}

TEST_CASE("tutorial - tensors: shape queries and copying", "[Docs][Tutorials]") {
    RuntimeTensor<double> A{"A", {3, 4, 5}};

    CHECK(A.rank() == 3);
    CHECK(A.dim(0) == 3);
    CHECK(A.dim(1) == 4);
    CHECK(A.dim(2) == 5);
    CHECK(A.size() == 3 * 4 * 5);
    CHECK(A.name() == "A");

    auto B  = create_random_tensor<double>("B", {4, 4});
    auto D  = B; // deep copy
    D(0, 0) = 999.0;
    CHECK(B(0, 0) != 999.0);
}

TEST_CASE("tutorial - tensors: a graph-owned result must say it is one", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {4, 4});
    auto B = create_random_tensor<double>("B", {4, 4});

    auto reference = create_zero_tensor<double>("reference", {4, 4});
    cg::einsum("ik;kj->ij", &reference, A, B);

    cg::Graph graph("example");
    auto     &C = graph.create_runtime_tensor<double>("C", {4, 4}, /*intermediate=*/false);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph.optimize();
    graph.execute();

    CHECK(C(0, 0) == reference(0, 0));
}

// ── user/tutorial_views.rst ─────────────────────────────────────────────────

TEST_CASE("tutorial - views: slicing writes through and keeps parent strides", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {10, 10});

    auto block  = A(Range{0, 3}, Range{0, 3});
    block(0, 0) = 999.0;
    CHECK(A(0, 0) == 999.0);

    auto row = A(5, All);
    auto col = A(All, 3);
    CHECK(row.dim(0) == 10);
    CHECK(col.dim(0) == 10);

    // Column-major: the FIRST index is contiguous. The page had this inverted.
    CHECK(A.stride(0) == 1);
    CHECK(A.stride(1) == 10);

    auto slice = A(Range{2, 5}, Range{3, 7});
    CHECK(slice.dim(0) == 3);
    CHECK(slice.dim(1) == 4);
    CHECK(slice.stride(0) == 1);
    CHECK(slice.stride(1) == 10);
}

TEST_CASE("tutorial - views: orbital blocks contract without copying", "[Docs][Tutorials]") {
    size_t const n_occ  = 5;
    size_t const n_virt = 15;
    size_t const n_orbs = n_occ + n_virt;

    auto F   = create_random_tensor<double>("Fock", {n_orbs, n_orbs});
    auto Foo = F(Range{0, n_occ}, Range{0, n_occ});
    auto Fov = F(Range{0, n_occ}, Range{n_occ, n_orbs});
    auto Fvv = F(Range{n_occ, n_orbs}, Range{n_occ, n_orbs});

    CHECK(Foo.dim(0) == n_occ);
    CHECK(Fov.dim(1) == n_virt);
    CHECK(Fvv.dim(0) == n_virt);
    CHECK(Fov(0, 0) == F(0, n_occ));

    auto T = create_random_tensor<double>("T", {n_occ, n_virt});

    cg::Graph graph("blocks");
    auto     &out = graph.create_runtime_tensor<double>("out", {n_occ, n_occ}, false);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ja->ij", &out, Fov, T);
    }
    graph.optimize();
    CHECK_NOTHROW(graph.execute());
}

// ── user/tutorial_einsum.rst ────────────────────────────────────────────────

TEST_CASE("tutorial - einsum: the forms the page teaches", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {7, 7});
    auto B = create_random_tensor<double>("B", {7, 7});
    auto C = create_zero_tensor<double>("C", {7, 7});

    cg::einsum("ik;kj->ij", &C, A, B);
    double const product = C(0, 0);

    // The arrow-left spelling is the same contraction.
    auto C2 = create_zero_tensor<double>("C2", {7, 7});
    cg::einsum("ij <- ik ; kj", &C2, A, B);
    CHECK(C2(0, 0) == product);

    // Prefactors.
    cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
    CHECK(C(0, 0) == Catch::Approx(2.0 * product));

    // A scalar comes out through dot, not through a spec.
    auto   u      = create_random_tensor<double>("u", {100});
    auto   v      = create_random_tensor<double>("v", {100});
    double result = 0.0;
    cg::dot(&result, u, v);
    CHECK(result != 0.0);

    // Rank is not special.
    auto T = create_random_tensor<double>("T", {4, 4, 4, 4});
    auto U = create_random_tensor<double>("U", {4, 4, 4, 4});
    auto W = create_zero_tensor<double>("W", {4, 4, 4, 4});
    CHECK_NOTHROW(cg::einsum("ijkl;klmn->ijmn", &W, T, U));
}

TEST_CASE("tutorial - einsum: scratch and result inside one capture", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {7, 7});
    auto B = create_random_tensor<double>("B", {7, 7});

    auto t = create_zero_tensor<double>("t", {7, 7});
    auto o = create_zero_tensor<double>("o", {7, 7});
    cg::einsum("ik;kj->ij", &t, A, B);
    cg::einsum("ik;kj->ij", &o, t, A);

    cg::Graph graph("two steps");
    auto     &tmp = graph.create_runtime_tensor<double>("tmp", {7, 7});
    auto     &out = graph.create_runtime_tensor<double>("out", {7, 7}, /*intermediate=*/false);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &out, tmp, A);
    }
    graph.optimize();
    graph.execute();

    CHECK(out(0, 0) == o(0, 0));
}

// ── user/tutorial_linalg.rst ────────────────────────────────────────────────

TEST_CASE("tutorial - linalg: what RuntimeTensor supports", "[Docs][Tutorials]") {
    size_t const N = 4;
    auto         A = create_random_tensor<double>("A", {N, N});
    auto         B = create_random_tensor<double>("B", {N, N});
    auto         C = create_zero_tensor<double>("C", {N, N});
    auto         x = create_random_tensor<double>("x", {N});
    auto         y = create_random_tensor<double>("y", {N});

    CHECK_NOTHROW(la::scale(2.0, &C));
    CHECK_NOTHROW(la::axpy(1.0, A, &C));
    CHECK_NOTHROW((la::gemm<false, false>(1.0, A, B, 0.0, &C)));
    CHECK_NOTHROW(la::gemv<false>(1.0, A, x, 0.0, &y));
    CHECK_NOTHROW(la::ger(1.0, x, y, &C));
    CHECK_NOTHROW(la::dot(x, y));

    // The norm is named; there is no one-argument overload, which the page used to show.
    CHECK_NOTHROW(la::norm(la::Norm::FROBENIUS, A));

    auto M = create_random_definite<double>("M", static_cast<int>(N));
    auto w = create_zero_tensor<double>("w", {N});
    CHECK_NOTHROW(la::syev(&M, &w));
}

TEST_CASE("tutorial - linalg: the three routines that want a static rank", "[Docs][Tutorials]") {
    // det, svd and qr are constrained to MatrixConcept, a compile-time rank of two, so they do
    // not take a RuntimeTensor. The page says so; this is what keeps that true.
    auto A              = create_random_tensor<double>("A", 6, 4);
    auto [U, sigma, Vt] = la::svd(A);
    CHECK(sigma.dim(0) == 4);

    auto Aq     = create_random_tensor<double>("Aq", 6, 4);
    auto [Q, R] = la::qr(Aq);

    auto   Ad = create_random_tensor<double>("Ad", 3, 3);
    double d  = la::det(Ad);
    CHECK(std::isfinite(d));
}

TEST_CASE("tutorial - linalg: a captured gemm and scale", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", {64, 64});
    auto B = create_random_tensor<double>("B", {64, 64});

    cg::Graph graph("update");
    auto     &C = graph.create_runtime_tensor<double>("C", {64, 64}, /*intermediate=*/false);
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, A, B, 0.0, &C);
        cg::scale(0.5, &C);
    }
    graph.optimize();
    graph.execute();

    auto reference = create_zero_tensor<double>("reference", {64, 64});
    la::gemm<false, false>(1.0, A, B, 0.0, &reference);
    CHECK(C(0, 0) == Catch::Approx(0.5 * reference(0, 0)));
}

// ── user/tutorial_compute_graph.rst ─────────────────────────────────────────

TEST_CASE("tutorial - compute graph: capture, execute and replay", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    cg::Graph graph("matmul");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }
    // Capture records; nothing has run yet.
    CHECK(C(3, 4) == 0.0);

    graph.execute();
    auto reference = create_zero_tensor<double>("reference", 10, 8);
    la::gemm<false, false>(1.0, A, B, 0.0, &reference);
    CHECK(C(3, 4) == Catch::Approx(reference(3, 4)));

    // Replay reads A's new data.
    A.set_all(1.0);
    graph.execute();
    double column_sum = 0.0;
    for (size_t k = 0; k < 5; ++k)
        column_sum += B(k, 4);
    CHECK(C(3, 4) == Catch::Approx(column_sum));
}

TEST_CASE("tutorial - compute graph: a graph-owned intermediate", "[Docs][Tutorials]") {
    auto A = create_random_tensor<double>("A", 10, 5);
    auto B = create_random_tensor<double>("B", 5, 8);
    auto C = create_zero_tensor<double>("C", 10, 8);

    cg::Graph graph("pipeline");
    auto     &tmp = graph.create_zero_tensor<double, 2>("tmp", 10, 8);
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ij <- ik ; kj", &tmp, A, B);
        cg::scale(2.0, &tmp);
        cg::axpy(1.0, tmp, &C);
    }
    graph.execute();

    auto reference = create_zero_tensor<double>("reference", 10, 8);
    la::gemm<false, false>(2.0, A, B, 0.0, &reference);
    CHECK(C(9, 7) == Catch::Approx(reference(9, 7)));
}

TEST_CASE("tutorial - compute graph: a pipeline with a setup stage and a loop", "[Docs][Tutorials]") {
    auto H = create_random_tensor<double>("H", 6, 6);
    auto D = create_random_tensor<double>("D", 6, 6);
    auto F = create_zero_tensor<double>("F", 6, 6);

    size_t iterations = 0;

    cg::Pipeline pipeline("scf");
    pipeline.add_stage("setup", [&]() { cg::einsum("ij <- ik ; kj", &F, H, D); });
    pipeline.add_loop(
        "iterate", 100, [&](size_t iter) { return iter + 1 < 3; },
        [&]() {
            cg::einsum("ij <- ik ; kj", &F, H, D);
            cg::custom("count", [&]() { ++iterations; }, &F);
        });
    pipeline.execute();

    CHECK(iterations == 3);
    auto reference = create_zero_tensor<double>("reference", 6, 6);
    la::gemm<false, false>(1.0, H, D, 0.0, &reference);
    CHECK(F(2, 5) == Catch::Approx(reference(2, 5)));
}

TEST_CASE("tutorial - compute graph: parallel_for and parallel_reduce order around an assembly", "[Docs][Tutorials]") {
    size_t const N       = 6;
    size_t const n_pairs = N * N;

    auto H = create_random_tensor<double>("H", N, N);
    auto D = create_random_tensor<double>("D", N, N);
    auto J = create_zero_tensor<double>("J", N, N);
    auto K = create_zero_tensor<double>("K", N, N);
    auto F = create_zero_tensor<double>("F", N, N);

    double energy = 0.0;

    cg::Graph graph("fock_build");
    {
        cg::CaptureGuard guard(graph);

        cg::parallel_for(
            "integrals", 0, n_pairs,
            [&](size_t pair) {
                J(pair / N, pair % N) = 1.0;
                K(pair / N, pair % N) = 0.5;
            },
            &J, &K);

        cg::permute("ij <- ij", 0.0, &F, 1.0, H);
        cg::axpy(2.0, J, &F);
        cg::axpy(-1.0, K, &F);

        cg::parallel_reduce<double>(
            "energy", 0, N, &energy, []() { return 0.0; },
            [&](size_t i, double &acc) {
                for (size_t j = 0; j < N; ++j)
                    acc += D(i, j) * F(i, j);
            },
            [](double &g, double const &l) { g += l; }, &D, &F);
    }
    graph.execute();

    // F = H + 2J - K with J = 1 and K = 0.5 everywhere, so F = H + 1.5.
    CHECK(F(1, 2) == Catch::Approx(H(1, 2) + 1.5));
    double expected = 0.0;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            expected += D(i, j) * (H(i, j) + 1.5);
    CHECK(energy == Catch::Approx(expected));
}

TEST_CASE("tutorial - compute graph: a float contraction through the default passes", "[Docs][Tutorials]") {
    auto A = create_random_tensor<float>("A", 256, 256);
    auto B = create_random_tensor<float>("B", 256, 256);
    auto C = create_zero_tensor<float>("C", 256, 256);

    cg::Graph graph("my_computation");
    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }

    // The GPU passes join the default pipeline only when a backend is built in; the answer is the
    // same either way.
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    auto reference = create_zero_tensor<float>("reference", 256, 256);
    la::gemm<false, false>(1.0f, A, B, 0.0f, &reference);
    CHECK(C(17, 200) == Catch::Approx(reference(17, 200)).epsilon(1e-4));
}

TEST_CASE("tutorial - compute graph: a custom node feeding a pairwise contraction", "[Docs][Tutorials]") {
    size_t const nmo = 5;

    auto C    = create_random_tensor<double>("C", nmo, nmo);
    auto D    = create_random_tensor<double>("D", nmo, nmo);
    auto F    = create_zero_tensor<double>("F", nmo, nmo);
    auto F_mo = create_zero_tensor<double>("F_mo", nmo, nmo);

    cg::Graph graph("scf_iteration");
    auto     &CF = graph.create_zero_tensor<double, 2>("CF", nmo, nmo);
    {
        cg::CaptureGuard guard(graph);

        // Stands in for the page's build_fock_matrix; the page reads its inputs from disk first.
        // std::tie does not compile here: it yields non-const references, and the inputs tuple is
        // std::tuple<Inputs const &...>. The page shows make_tuple over cref and ref for that reason.
        cg::custom("build_fock", std::make_tuple(std::cref(D)), std::make_tuple(std::ref(F)),
                   [&]() { la::gemm<false, false>(1.0, D, D, 0.0, &F); });

        cg::einsum("pj <- pi ; ij", 0.0, &CF, 1.0, C, F);
        cg::einsum("pq <- pj ; jq", 0.0, &F_mo, 1.0, CF, C);
    }
    graph.execute();

    auto fock = create_zero_tensor<double>("fock", nmo, nmo);
    auto cf   = create_zero_tensor<double>("cf", nmo, nmo);
    auto ref  = create_zero_tensor<double>("ref", nmo, nmo);
    la::gemm<false, false>(1.0, D, D, 0.0, &fock);
    la::gemm<false, false>(1.0, C, fock, 0.0, &cf);
    la::gemm<false, false>(1.0, cf, C, 0.0, &ref);
    CHECK(F_mo(1, 3) == Catch::Approx(ref(1, 3)));
}

TEST_CASE("tutorial - compute graph: workspace tensors are declared, then materialized", "[Docs][Tutorials]") {
    size_t const nao = 4;

    cg::Workspace ws("calculation");
    auto         &eri = ws.declare_tensor<double, 4>("ERI", nao, nao, nao, nao);
    auto          D   = create_random_tensor<double>("D", nao, nao);

    cg::Pipeline scf("scf");
    scf.set_workspace(ws);

    auto &F = scf.declare_zero_tensor<double, 2>("F", nao, nao);

    {
        auto            &stage = scf.add_stage("compute");
        cg::CaptureGuard guard(stage);
        cg::einsum("ij <- ijkl ; kl", 0.0, &F, 1.0, eri, D);
    }

    auto pm = cg::PassManager::create_default();
    scf.apply(pm);
    CHECK_NOTHROW(scf.execute());
}
