//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_LinearCombinationContractionFolding.cpp
/// @brief C++ tests for LCCF. The behavioral matrix lives in the Python
///        mirror (test_pass_linear_combination_contraction_folding_python.py);
///        this file pins two things Python cannot reach:
///        - the consumer-bearing topology, where the fused Custom node's
///          PLACEMENT is load-bearing (the node-position hazard - position is
///          program order in this IR), and
///        - statically-typed Tensor<T, Rank> captures, which the fused
///          executor must NOT touch (it casts the user operands to
///          GeneralRuntimeTensor<T>, so folding a typed capture was type
///          confusion and a segfault).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {
// out = 2*einsum("ij <- k ; kij") - einsum("ij <- k ; kji"), the 2J-K
// transpose-pair shape LCCF exists for, followed by D = out * E.
template <typename OutT, typename DT, typename AT, typename BT, typename ET>
void capture_program(cg::Graph &graph, OutT &out, DT &D, AT const &A, BT const &B, ET const &E) {
    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j <- k ; k,i,j", 0.0, &out, 2.0, A, B);
    cg::einsum("i,j <- k ; k,j,i", 1.0, &out, -1.0, A, B);
    cg::einsum("i,k <- i,j ; j,k", 0.0, &D, 1.0, out, E); // consumer of the folded output
}

void reference(Tensor<double, 2> &out_ref, Tensor<double, 2> &D_ref, Tensor<double, 1> const &A, Tensor<double, 3> const &B,
               Tensor<double, 2> const &E) {
    out_ref.zero();
    D_ref.zero();
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            for (size_t kk = 0; kk < 4; kk++) {
                out_ref(ii, jj) += 2.0 * A(kk) * B(kk, ii, jj) - A(kk) * B(kk, jj, ii);
            }
        }
    }
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            for (size_t kk = 0; kk < 3; kk++) {
                D_ref(ii, jj) += out_ref(ii, kk) * E(kk, jj);
            }
        }
    }
}
} // namespace

TEST_CASE("LCCF - folded output feeding a downstream consumer stays correct", "[ComputeGraph][Passes][LCCF]") {
    auto A = create_random_tensor<double>("A", 4);
    auto B = create_random_tensor<double>("B", 4, 3, 3);
    auto E = create_random_tensor<double>("E", 3, 3);

    Tensor<double, 2> out_ref("out_ref", 3, 3), D_ref("D_ref", 3, 3);
    reference(out_ref, D_ref, A, B, E);

    // Runtime-tensor captures: the shape LCCF's fused executor is built for.
    RuntimeTensor<double> A_rt(A), B_rt(B), E_rt(E);
    RuntimeTensor<double> out_rt("out", std::vector<size_t>{3, 3});
    RuntimeTensor<double> D_rt("D", std::vector<size_t>{3, 3});
    out_rt.zero();
    D_rt.zero();

    cg::Graph graph("lccf_consumer");
    capture_program(graph, out_rt, D_rt, A_rt, B_rt, E_rt);

    auto [modified, pass] = graph.apply<cg::passes::LinearCombinationContractionFolding>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(out_rt(idx) - out_ref(ii, jj)) < 1e-11);
            REQUIRE(std::abs(D_rt(idx) - D_ref(ii, jj)) < 1e-11);
        }
    }
}

TEST_CASE("LCCF - statically-typed captures are not folded (and stay correct)", "[ComputeGraph][Passes][LCCF]") {
    // Regression guard: the fused executor casts the user operands to
    // GeneralRuntimeTensor<T>. Folding a Tensor<T, Rank> capture was type
    // confusion (segfault in the fused axpy). The pass must skip these and
    // the unfused graph must still execute correctly.
    auto A   = create_random_tensor<double>("A", 4);
    auto B   = create_random_tensor<double>("B", 4, 3, 3);
    auto E   = create_random_tensor<double>("E", 3, 3);
    auto out = create_zero_tensor<double>("out", 3, 3);
    auto D   = create_zero_tensor<double>("D", 3, 3);

    Tensor<double, 2> out_ref("out_ref", 3, 3), D_ref("D_ref", 3, 3);
    reference(out_ref, D_ref, A, B, E);

    cg::Graph graph("lccf_typed");
    capture_program(graph, out, D, A, B, E);

    auto [modified, pass] = graph.apply<cg::passes::LinearCombinationContractionFolding>();
    CHECK_FALSE(modified);
    CHECK(pass.num_groups() == 0);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-11);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-11);
        }
    }
}
