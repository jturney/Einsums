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
