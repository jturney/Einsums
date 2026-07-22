//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_StreamContractionFusion.cpp
/// @brief Tests for StreamContractionFusion: the Fock "J and K from one TEI
///        stream" idiom - distinct outputs, a shared accumulated output, the
///        interference guard, the size threshold, and complex elements.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <cmath>
#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {
constexpr int kN = 40; // TEI has kN^4 = 2.56M elements, above the pass's stream threshold
}

TEST_CASE("StreamContractionFusion - J and K from one TEI stream (distinct outputs)", "[ComputeGraph][Passes][StreamFusion]") {
    auto TEI = create_random_tensor<double>("TEI", kN, kN, kN, kN);
    auto D   = create_random_tensor<double>("D", kN, kN);

    // Eager reference.
    Tensor<double, 2> J_ref("J_ref", kN, kN), K_ref("K_ref", kN, kN);
    einsum(0.0, Indices{i, j}, &J_ref, 2.0, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
    einsum(0.0, Indices{i, j}, &K_ref, -1.0, Indices{i, k, j, l}, TEI, Indices{k, l}, D);

    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), K_rt("K", std::vector<size_t>{kN, kN});
    J_rt.zero();
    K_rt.zero();

    cg::Graph graph("stream_jk");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);
    REQUIRE(pass.num_eliminated() == 1);

    graph.execute();

    for (size_t ii = 0; ii < kN; ii++) {
        for (size_t jj = 0; jj < kN; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE_THAT(J_rt(idx), Catch::Matchers::WithinAbs(J_ref(ii, jj), 1e-9));
            REQUIRE_THAT(K_rt(idx), Catch::Matchers::WithinAbs(K_ref(ii, jj), 1e-9));
        }
    }
}

TEST_CASE("StreamContractionFusion - shared accumulated output (G = 2J - K)", "[ComputeGraph][Passes][StreamFusion]") {
    auto TEI = create_random_tensor<double>("TEI", kN, kN, kN, kN);
    auto D   = create_random_tensor<double>("D", kN, kN);

    Tensor<double, 2> G_ref("G_ref", kN, kN);
    einsum(0.0, Indices{i, j}, &G_ref, 2.0, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
    einsum(1.0, Indices{i, j}, &G_ref, -1.0, Indices{i, k, j, l}, TEI, Indices{k, l}, D);

    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> G_rt("G", std::vector<size_t>{kN, kN});
    G_rt.zero();

    cg::Graph graph("stream_g");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &G_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 1.0, &G_rt, -1.0, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t ii = 0; ii < kN; ii++) {
        for (size_t jj = 0; jj < kN; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE_THAT(G_rt(idx), Catch::Matchers::WithinAbs(G_ref(ii, jj), 1e-9));
        }
    }
}

TEST_CASE("StreamContractionFusion - dependency-pinned intervening writer blocks fusion", "[ComputeGraph][Passes][StreamFusion]") {
    auto TEI = create_random_tensor<double>("TEI", kN, kN, kN, kN);
    auto D   = create_random_tensor<double>("D", kN, kN);

    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), K_rt("K", std::vector<size_t>{kN, kN});
    J_rt.zero();
    K_rt.zero();

    cg::Graph graph("stream_hazard");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
        // Rewrites D from J between the members. The topological sort cannot
        // hoist K past it (K reads the new D) nor sink it (it reads J), so K
        // legitimately contracts a DIFFERENT D than J did - fusing would
        // compute K with the old D.
        cg::einsum("k,l <- k,i ; i,l", 0.0, &D_rt, 0.5, J_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_groups() == 0);
}

TEST_CASE("StreamContractionFusion - below the stream threshold stays unfused", "[ComputeGraph][Passes][StreamFusion]") {
    constexpr int n   = 7; // 2401 elements, below the 4096-element threshold
    auto          TEI = create_random_tensor<double>("TEI", n, n, n, n);
    auto          D   = create_random_tensor<double>("D", n, n);

    RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<double> J_rt("J", std::vector<size_t>{n, n}), K_rt("K", std::vector<size_t>{n, n});
    J_rt.zero();
    K_rt.zero();

    cg::Graph graph("stream_small");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_groups() == 0);
}

TEST_CASE("StreamContractionFusion - complex prefactors on complex tensors", "[ComputeGraph][Passes][StreamFusion]") {
    using T           = std::complex<double>;
    constexpr int n_c = 34; // 34^4 = 1.34M elements, above the threshold

    auto TEI = create_random_tensor<T>("TEI", n_c, n_c, n_c, n_c);
    auto D   = create_random_tensor<T>("D", n_c, n_c);
    auto J0  = create_random_tensor<T>("J0", n_c, n_c); // pre-existing J content, scaled by the complex c prefactor

    T const alpha_j{2.0, 0.5};
    T const alpha_k{-1.0, 0.25};
    T const c_j{0.5, -0.25};

    Tensor<T, 2> J_ref = J0;
    Tensor<T, 2> K_ref("K_ref", n_c, n_c);
    einsum(c_j, Indices{i, j}, &J_ref, alpha_j, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
    einsum(T{0.0}, Indices{i, j}, &K_ref, alpha_k, Indices{i, k, j, l}, TEI, Indices{k, l}, D);

    RuntimeTensor<T> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<T> J_rt(J0);
    RuntimeTensor<T> K_rt("K", std::vector<size_t>{n_c, n_c});
    K_rt.zero();

    cg::Graph graph("stream_complex_pf");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", c_j, &J_rt, alpha_j, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", T{0.0}, &K_rt, alpha_k, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);
    REQUIRE(pass.num_eliminated() == 1);

    graph.execute();

    for (size_t ii = 0; ii < n_c; ii++) {
        for (size_t jj = 0; jj < n_c; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(J_rt(idx) - J_ref(ii, jj)) < 1e-8);
            REQUIRE(std::abs(K_rt(idx) - K_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("StreamContractionFusion - profile-derived output cap", "[ComputeGraph][Passes][StreamFusion]") {
    auto TEI = create_random_tensor<double>("TEI", kN, kN, kN, kN);
    auto D   = create_random_tensor<double>("D", kN, kN);

    auto const capture = [&](cg::Graph &graph, RuntimeTensor<double> &TEI_rt, RuntimeTensor<double> &D_rt, RuntimeTensor<double> &J_rt,
                             RuntimeTensor<double> &K_rt) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K_rt, -1.0, TEI_rt, D_rt);
    };

    SECTION("over-cap outputs fuse via owner-computes chunking") {
        // llc/threads/8B lands at or below the 1024-element floor, so the
        // kN x kN = 1600-element outputs exceed the privatization cap - but
        // physical axis 0 carries output label i for BOTH members, so the
        // chunked kernel writes disjoint output slices directly.
        cg::HardwareProfile profile{};
        profile.cpu.caches = {cg::CacheLevel{.size_bytes = 4096}};

        Tensor<double, 2> J_ref("J_ref", kN, kN), K_ref("K_ref", kN, kN);
        einsum(0.0, Indices{i, j}, &J_ref, 2.0, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
        einsum(0.0, Indices{i, j}, &K_ref, -1.0, Indices{i, k, j, l}, TEI, Indices{k, l}, D);

        RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
        RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), K_rt("K", std::vector<size_t>{kN, kN});
        J_rt.zero();
        K_rt.zero();

        cg::Graph graph("stream_tiny_cache");
        capture(graph, TEI_rt, D_rt, J_rt, K_rt);

        cg::passes::StreamContractionFusion pass(profile);
        REQUIRE(pass.max_output_elems(sizeof(double)) < static_cast<size_t>(kN) * kN);
        REQUIRE(pass.run(graph));
        REQUIRE(pass.num_groups() == 1);

        graph.execute();
        for (size_t ii = 0; ii < kN; ii++) {
            for (size_t jj = 0; jj < kN; jj++) {
                std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
                REQUIRE_THAT(J_rt(idx), Catch::Matchers::WithinAbs(J_ref(ii, jj), 1e-9));
                REQUIRE_THAT(K_rt(idx), Catch::Matchers::WithinAbs(K_ref(ii, jj), 1e-9));
            }
        }
    }

    SECTION("two over-cap outputs with no shared axis stay unfused") {
        // J(i,j) needs a chunk axis from {i,j}, X(k,l) needs one from {k,l};
        // no axis covers both over-cap outputs, so both drop and no group
        // survives.
        cg::HardwareProfile profile{};
        profile.cpu.caches = {cg::CacheLevel{.size_bytes = 4096}};

        RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
        RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), X_rt("X", std::vector<size_t>{kN, kN});
        J_rt.zero();
        X_rt.zero();

        cg::Graph graph("stream_disjoint_axes");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
            cg::einsum("k,l <- i,j,k,l ; i,j", 0.0, &X_rt, 1.0, TEI_rt, D_rt);
        }

        cg::passes::StreamContractionFusion pass(profile);
        REQUIRE_FALSE(pass.run(graph));
        REQUIRE(pass.num_groups() == 0);
    }

    SECTION("mixed mode: chunked large output plus privatized small output") {
        // J(i,j) is over the tiny cap, so the group chunks one of J's axes
        // (the kernel picks the higher-stride j); Y(k) does not contain the
        // chunk label but is far under the cap, so it keeps thread-private
        // accumulators inside the same fused stream.
        cg::HardwareProfile profile{};
        profile.cpu.caches = {cg::CacheLevel{.size_bytes = 4096}};

        auto W3 = create_random_tensor<double>("W3", kN, kN, kN);

        Tensor<double, 2> J_ref("J_ref", kN, kN);
        Tensor<double, 1> Y_ref("Y_ref", kN);
        einsum(0.0, Indices{i, j}, &J_ref, 2.0, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
        einsum(0.0, Indices{k}, &Y_ref, 0.5, Indices{i, j, k, l}, TEI, Indices{i, j, l}, W3);

        RuntimeTensor<double> TEI_rt(TEI), D_rt(D), W3_rt(W3);
        RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), Y_rt("Y", std::vector<size_t>{kN});
        J_rt.zero();
        Y_rt.zero();

        cg::Graph graph("stream_mixed_mode");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J_rt, 2.0, TEI_rt, D_rt);
            cg::einsum("k <- i,j,k,l ; i,j,l", 0.0, &Y_rt, 0.5, TEI_rt, W3_rt);
        }

        cg::passes::StreamContractionFusion pass(profile);
        REQUIRE(pass.run(graph));
        REQUIRE(pass.num_groups() == 1);

        graph.execute();
        for (size_t ii = 0; ii < kN; ii++) {
            for (size_t jj = 0; jj < kN; jj++) {
                std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
                REQUIRE_THAT(J_rt(idx), Catch::Matchers::WithinAbs(J_ref(ii, jj), 1e-9));
            }
            std::vector<ptrdiff_t> const yidx{static_cast<ptrdiff_t>(ii)};
            REQUIRE_THAT(Y_rt(yidx), Catch::Matchers::WithinAbs(Y_ref(ii), 1e-8));
        }
    }

    SECTION("a generous cache fuses") {
        cg::HardwareProfile profile{};
        profile.cpu.caches = {cg::CacheLevel{.size_bytes = size_t{1} << 30}};

        RuntimeTensor<double> TEI_rt(TEI), D_rt(D);
        RuntimeTensor<double> J_rt("J", std::vector<size_t>{kN, kN}), K_rt("K", std::vector<size_t>{kN, kN});
        J_rt.zero();
        K_rt.zero();

        cg::Graph graph("stream_big_cache");
        capture(graph, TEI_rt, D_rt, J_rt, K_rt);

        cg::passes::StreamContractionFusion pass(profile);
        REQUIRE(pass.max_output_elems(sizeof(double)) >= static_cast<size_t>(kN) * kN);
        REQUIRE(pass.run(graph));
        REQUIRE(pass.num_groups() == 1);
    }

    SECTION("no cache data keeps the fallback cap") {
        cg::passes::StreamContractionFusion const with_empty_profile{cg::HardwareProfile{}};
        cg::passes::StreamContractionFusion const without_profile{};
        REQUIRE(with_empty_profile.max_output_elems(sizeof(double)) == without_profile.max_output_elems(sizeof(double)));
    }
}

TEST_CASE("StreamContractionFusion - complex elements with real prefactors", "[ComputeGraph][Passes][StreamFusion]") {
    using T           = std::complex<double>;
    constexpr int n_c = 34; // 34^4 = 1.34M elements, above the threshold

    auto TEI = create_random_tensor<T>("TEI", n_c, n_c, n_c, n_c);
    auto D   = create_random_tensor<T>("D", n_c, n_c);

    Tensor<T, 2> J_ref("J_ref", n_c, n_c), K_ref("K_ref", n_c, n_c);
    einsum(T{0.0}, Indices{i, j}, &J_ref, T{2.0}, Indices{i, j, k, l}, TEI, Indices{k, l}, D);
    einsum(T{0.0}, Indices{i, j}, &K_ref, T{-1.0}, Indices{i, k, j, l}, TEI, Indices{k, l}, D);

    RuntimeTensor<T> TEI_rt(TEI), D_rt(D);
    RuntimeTensor<T> J_rt("J", std::vector<size_t>{n_c, n_c}), K_rt("K", std::vector<size_t>{n_c, n_c});
    J_rt.zero();
    K_rt.zero();

    cg::Graph graph("stream_complex");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k,l ; k,l", T{0.0}, &J_rt, T{2.0}, TEI_rt, D_rt);
        cg::einsum("i,j <- i,k,j,l ; k,l", T{0.0}, &K_rt, T{-1.0}, TEI_rt, D_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t ii = 0; ii < n_c; ii++) {
        for (size_t jj = 0; jj < n_c; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(J_rt(idx) - J_ref(ii, jj)) < 1e-8);
            REQUIRE(std::abs(K_rt(idx) - K_ref(ii, jj)) < 1e-8);
        }
    }
}

// The remaining cases target the four branches of the SIMD inner kernel
// (StreamKernelBody.hpp) directly, over both real dtypes. Streamed tensors are
// column-major, so the innermost storage axis is physical axis 0 (label i); the
// (ds, dc, dw) triple for that axis then depends only on whether i appears in
// each member's output C and weight W:
//   i in C, not W  -> (1,1,0) scaled AXPY
//   i in C and W   -> (1,1,1) Hadamard FMA
//   i in W, not C  -> (1,0,1) dot reduction
//   i in neither   -> (1,0,0) scalar fallback (also the path complex takes)
// n^3 = 8000 clears the stream threshold; the n^2 / n outputs stay privatized.

namespace {
constexpr int kM = 20;

template <typename T>
RemoveComplexT<T> stream_tol() {
    using R = RemoveComplexT<T>;
    return std::is_same_v<R, float> ? R{1e-3} : R{1e-9};
}
} // namespace

TEMPLATE_TEST_CASE("StreamContractionFusion - scaled AXPY branch (1,1,0)", "[ComputeGraph][Passes][StreamFusion]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;
    auto S  = create_random_tensor<T>("S", kM, kM, kM);
    auto W1 = create_random_tensor<T>("W1", kM, kM);
    auto W2 = create_random_tensor<T>("W2", kM, kM);

    // C(i,j) = sum_k S(i,j,k) * W(j,k): i (unit-stride axis) is in C, not W.
    Tensor<T, 2> C1_ref("C1_ref", kM, kM), C2_ref("C2_ref", kM, kM);
    einsum(T{0}, Indices{i, j}, &C1_ref, T{1}, Indices{i, j, k}, S, Indices{j, k}, W1);
    einsum(T{0}, Indices{i, j}, &C2_ref, T{1}, Indices{i, j, k}, S, Indices{j, k}, W2);

    RuntimeTensor<T> S_rt(S), W1_rt(W1), W2_rt(W2);
    RuntimeTensor<T> C1_rt("C1", std::vector<size_t>{kM, kM}), C2_rt("C2", std::vector<size_t>{kM, kM});
    C1_rt.zero();
    C2_rt.zero();

    cg::Graph graph("stream_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k ; j,k", T{0}, &C1_rt, T{1}, S_rt, W1_rt);
        cg::einsum("i,j <- i,j,k ; j,k", T{0}, &C2_rt, T{1}, S_rt, W2_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t ii = 0; ii < kM; ii++) {
        for (size_t jj = 0; jj < kM; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(C1_rt(idx) - C1_ref(ii, jj)) < stream_tol<T>());
            REQUIRE(std::abs(C2_rt(idx) - C2_ref(ii, jj)) < stream_tol<T>());
        }
    }
}

TEMPLATE_TEST_CASE("StreamContractionFusion - Hadamard FMA branch (1,1,1)", "[ComputeGraph][Passes][StreamFusion]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;
    auto S  = create_random_tensor<T>("S", kM, kM, kM);
    auto W1 = create_random_tensor<T>("W1", kM, kM);
    auto W2 = create_random_tensor<T>("W2", kM, kM);

    // C(i,j) = W(i,j) * sum_k S(i,j,k): i is in both C and W. Hand-computed
    // reference (this pattern has no A-B link index, so the eager einsum path
    // is not a reliable oracle for it).
    Tensor<T, 2> C1_ref("C1_ref", kM, kM), C2_ref("C2_ref", kM, kM);
    for (int a = 0; a < kM; a++) {
        for (int b = 0; b < kM; b++) {
            T s{0};
            for (int c = 0; c < kM; c++) {
                s += S(a, b, c);
            }
            C1_ref(a, b) = s * W1(a, b);
            C2_ref(a, b) = s * W2(a, b);
        }
    }

    RuntimeTensor<T> S_rt(S), W1_rt(W1), W2_rt(W2);
    RuntimeTensor<T> C1_rt("C1", std::vector<size_t>{kM, kM}), C2_rt("C2", std::vector<size_t>{kM, kM});
    C1_rt.zero();
    C2_rt.zero();

    cg::Graph graph("stream_hadamard");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,j,k ; i,j", T{0}, &C1_rt, T{1}, S_rt, W1_rt);
        cg::einsum("i,j <- i,j,k ; i,j", T{0}, &C2_rt, T{1}, S_rt, W2_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t ii = 0; ii < kM; ii++) {
        for (size_t jj = 0; jj < kM; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(C1_rt(idx) - C1_ref(ii, jj)) < stream_tol<T>());
            REQUIRE(std::abs(C2_rt(idx) - C2_ref(ii, jj)) < stream_tol<T>());
        }
    }
}

TEMPLATE_TEST_CASE("StreamContractionFusion - dot-reduction branch (1,0,1)", "[ComputeGraph][Passes][StreamFusion]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;
    auto S  = create_random_tensor<T>("S", kM, kM, kM);
    auto W1 = create_random_tensor<T>("W1", kM, kM);
    auto W2 = create_random_tensor<T>("W2", kM, kM);

    // C(j) = sum_{i,k} S(i,j,k) * W(i,k): i (unit-stride axis) is in W, not C.
    Tensor<T, 1> C1_ref("C1_ref", kM), C2_ref("C2_ref", kM);
    einsum(T{0}, Indices{j}, &C1_ref, T{1}, Indices{i, j, k}, S, Indices{i, k}, W1);
    einsum(T{0}, Indices{j}, &C2_ref, T{1}, Indices{i, j, k}, S, Indices{i, k}, W2);

    RuntimeTensor<T> S_rt(S), W1_rt(W1), W2_rt(W2);
    RuntimeTensor<T> C1_rt("C1", std::vector<size_t>{kM}), C2_rt("C2", std::vector<size_t>{kM});
    C1_rt.zero();
    C2_rt.zero();

    cg::Graph graph("stream_reduction");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("j <- i,j,k ; i,k", T{0}, &C1_rt, T{1}, S_rt, W1_rt);
        cg::einsum("j <- i,j,k ; i,k", T{0}, &C2_rt, T{1}, S_rt, W2_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t jj = 0; jj < kM; jj++) {
        std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(jj)};
        REQUIRE(std::abs(C1_rt(idx) - C1_ref(jj)) < stream_tol<T>());
        REQUIRE(std::abs(C2_rt(idx) - C2_ref(jj)) < stream_tol<T>());
    }
}

TEMPLATE_TEST_CASE("StreamContractionFusion - scalar fallback branch (1,0,0)", "[ComputeGraph][Passes][StreamFusion]", float, double,
                   std::complex<float>, std::complex<double>) {
    using T = TestType;
    auto S  = create_random_tensor<T>("S", kM, kM, kM);
    auto W1 = create_random_tensor<T>("W1", kM, kM);
    auto W2 = create_random_tensor<T>("W2", kM, kM);

    // C(j,k) = W(j,k) * sum_i S(i,j,k): i (unit-stride axis) is in neither C nor
    // W, so the innermost axis matches no vectorized triple - the real scalar
    // fallback (the same path complex elements take) runs. Hand-computed
    // reference (no A-B link index, so the eager einsum is not a reliable
    // oracle here).
    Tensor<T, 2> C1_ref("C1_ref", kM, kM), C2_ref("C2_ref", kM, kM);
    for (int b = 0; b < kM; b++) {
        for (int c = 0; c < kM; c++) {
            T s{0};
            for (int a = 0; a < kM; a++) {
                s += S(a, b, c);
            }
            C1_ref(b, c) = s * W1(b, c);
            C2_ref(b, c) = s * W2(b, c);
        }
    }

    RuntimeTensor<T> S_rt(S), W1_rt(W1), W2_rt(W2);
    RuntimeTensor<T> C1_rt("C1", std::vector<size_t>{kM, kM}), C2_rt("C2", std::vector<size_t>{kM, kM});
    C1_rt.zero();
    C2_rt.zero();

    cg::Graph graph("stream_fallback");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("j,k <- i,j,k ; j,k", T{0}, &C1_rt, T{1}, S_rt, W1_rt);
        cg::einsum("j,k <- i,j,k ; j,k", T{0}, &C2_rt, T{1}, S_rt, W2_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t jj = 0; jj < kM; jj++) {
        for (size_t kk = 0; kk < kM; kk++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(jj), static_cast<ptrdiff_t>(kk)};
            REQUIRE(std::abs(C1_rt(idx) - C1_ref(jj, kk)) < stream_tol<T>());
            REQUIRE(std::abs(C2_rt(idx) - C2_ref(jj, kk)) < stream_tol<T>());
        }
    }
}
