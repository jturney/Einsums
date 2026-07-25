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
#include <complex>

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

TEST_CASE("LCCF - complex prefactors on complex tensors fold exactly", "[ComputeGraph][Passes][LCCF]") {
    using T = std::complex<double>;

    auto A    = create_random_tensor<T>("A", 4);
    auto B    = create_random_tensor<T>("B", 4, 3, 3);
    auto out0 = create_random_tensor<T>("out0", 3, 3); // pre-existing content scaled by the complex c prefactor

    T const a1{2.0, 0.5};
    T const a2{-1.0, 0.25};
    T const c0{0.5, -0.75};

    Tensor<T, 2> out_ref("out_ref", 3, 3);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            out_ref(ii, jj) = c0 * out0(ii, jj);
            for (size_t kk = 0; kk < 4; kk++) {
                out_ref(ii, jj) += a1 * A(kk) * B(kk, ii, jj) + a2 * A(kk) * B(kk, jj, ii);
            }
        }
    }

    RuntimeTensor<T> A_rt(A), B_rt(B), out_rt(out0);

    cg::Graph graph("lccf_complex_pf");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- k ; k,i,j", c0, &out_rt, a1, A_rt, B_rt);
        cg::einsum("i,j <- k ; k,j,i", T{1.0}, &out_rt, a2, A_rt, B_rt);
    }

    auto [modified, pass] = graph.apply<cg::passes::LinearCombinationContractionFolding>();
    REQUIRE(modified);
    REQUIRE(pass.num_groups() == 1);
    REQUIRE(pass.num_eliminated() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(ii), static_cast<ptrdiff_t>(jj)};
            REQUIRE(std::abs(out_rt(idx) - out_ref(ii, jj)) < 1e-10);
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

// LCCF emits the L construction as its OWN node, separate from the contraction.
// That is what lets LoopInvariantHoisting lift it: L depends only on the
// non-shared operand, so when that operand is loop-invariant -- the common case,
// it is an integral block from one-time setup -- L is built once before the loop
// instead of rebuilt on every replay. While the two halves lived in a single
// fused executor, nothing could see that the L half was invariant, because the
// combined node also read the varying shared operand.
TEST_CASE("LCCF - the L builder is a separate node that LIH hoists out of a loop", "[ComputeGraph][Passes][LCCF]") {
    size_t const o = 2, v = 3, niter = 4;

    auto g_typed = create_random_tensor<double>("g", o, v, v, v); // loop-INVARIANT integral
    auto d_typed = create_random_tensor<double>("d", o, v);       // per-iteration t1 increment

    RuntimeTensor<double> const g_rt(g_typed), d_rt(d_typed);
    RuntimeTensor<double>       t1("t1", std::vector<size_t>{o, v});
    RuntimeTensor<double>       Fae("Fae", std::vector<size_t>{v, v});
    t1.zero();
    Fae.zero();

    // The CCSD spin-adaptation pair: the same integral read with transposed
    // indices, against a shared operand that changes each iteration.
    cg::Graph graph("lccf_hoist");
    {
        auto                  &body = graph.add_loop("iter", niter, [niter](size_t it) { return it + 1 < niter; });
        cg::CaptureGuard const guard(body);
        cg::einsum("a,e <- m,f ; m,a,f,e", 1.0, &Fae, 2.0, t1, g_rt);
        cg::einsum("a,e <- m,f ; m,a,e,f", 1.0, &Fae, -1.0, t1, g_rt);
        cg::axpy(1.0, d_rt, &t1); // t1 varies -> the contraction is NOT invariant
    }

    // Must go through a PassManager, not Graph::apply<PassType>(): the latter
    // calls pass.run(graph) directly with no subgraph descent, so a pass that
    // needs to be handed the loop BODY never sees the pair.
    cg::PassManager pm_lccf;
    auto            lccf = std::make_shared<cg::passes::LinearCombinationContractionFolding>();
    pm_lccf.add(lccf);
    REQUIRE(graph.apply(pm_lccf));
    REQUIRE(lccf->num_groups() == 1);

    // Count body nodes of a given kind. Re-resolves the loop descriptor on every
    // call on purpose: it is a pointer into a Node's op_data, and any pass that
    // inserts into the parent's node vector (LIH hoisting does) reallocates it and
    // dangles a cached one.
    auto body_kind = [&graph](cg::OpKind want) -> size_t {
        for (auto const &n : graph.nodes()) {
            if (auto const *d = std::get_if<cg::LoopDescriptor>(&n.op_data); d != nullptr && d->body) {
                return static_cast<size_t>(std::ranges::count_if(d->body->nodes(), [want](cg::Node const &bn) { return bn.kind == want; }));
            }
        }
        return 0;
    };

    // Two nodes now stand where the pair was: the Custom L builder, and the
    // contraction as a REAL Einsum node (via Graph::make_einsum_node) so the
    // descriptor-reading passes can see it rather than skipping an opaque blob.
    CHECK(body_kind(cg::OpKind::Custom) == 1);
    CHECK(body_kind(cg::OpKind::Einsum) == 1);

    // LIH lifts the builder: invariant input, single writer, destination unread.
    cg::PassManager pm_lih;
    auto            lih = std::make_shared<cg::passes::LoopInvariantHoisting>();
    pm_lih.add(lih);
    CHECK(graph.apply(pm_lih));
    CHECK(lih->num_hoisted() == 1);
    // The builder left the body and now sits in the parent, before the loop; the
    // contraction stays behind as a visible Einsum.
    CHECK(body_kind(cg::OpKind::Custom) == 0);
    CHECK(body_kind(cg::OpKind::Einsum) == 1);
    CHECK(std::ranges::any_of(graph.nodes(), [](cg::Node const &n) { return n.kind == cg::OpKind::Custom; }));

    graph.execute();

    // Oracle: t1 gains d AFTER each iteration's contraction, so iteration k
    // contracts k*d.
    Tensor<double, 2> ref("ref", v, v);
    ref.zero();
    for (size_t it = 0; it < niter; ++it) {
        for (size_t a = 0; a < v; ++a) {
            for (size_t e = 0; e < v; ++e) {
                double acc = 0.0;
                for (size_t m = 0; m < o; ++m) {
                    for (size_t f = 0; f < v; ++f) {
                        double const t = static_cast<double>(it) * d_typed(m, f);
                        acc += t * (2.0 * g_typed(m, a, f, e) - g_typed(m, a, e, f));
                    }
                }
                ref(a, e) += acc;
            }
        }
    }

    for (size_t a = 0; a < v; ++a) {
        for (size_t e = 0; e < v; ++e) {
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(a), static_cast<ptrdiff_t>(e)};
            REQUIRE(std::abs(Fae(idx) - ref(a, e)) < 1e-11);
        }
    }
}
