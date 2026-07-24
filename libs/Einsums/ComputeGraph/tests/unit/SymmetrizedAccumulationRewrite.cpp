//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Level-1 rewrite of the SymmetrizedAccumulation pass: fold the symacc idiom by
// making the permute accumulate directly into r2 (r2 += s2*P(tmp)) and dropping
// the second axpby + the tmpP buffer. Fires only on runtime tensors, so this
// exercises the rewrite (the typed-capture matcher test covers detection). The
// fused graph must produce the same r2 as the un-fused one.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <memory>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// r2 += s*(tmp + P(tmp)), tmp = A (outer) B, as four captured nodes.
void capture_symacc(cg::Graph &graph, RuntimeTensor<double> &r2, RuntimeTensor<double> &tmp, RuntimeTensor<double> &tmpP,
                    RuntimeTensor<double> const &A, RuntimeTensor<double> const &B, double s) {
    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
    cg::axpby(s, tmp, 1.0, &r2);
    cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
    cg::axpby(s, tmpP, 1.0, &r2);
}

// Same idiom, but with a DAMPING step (r2 = X + damp*r2) landing inside the fold
// window, between the permute and the second axpby. Mixing/damping with
// beta not in {0, 1} is routine in SCF and DIIS-driven codes.
void capture_symacc_with_damping(cg::Graph &graph, RuntimeTensor<double> &r2, RuntimeTensor<double> &tmp, RuntimeTensor<double> &tmpP,
                                 RuntimeTensor<double> const &A, RuntimeTensor<double> const &B, RuntimeTensor<double> const &X, double s,
                                 double damp) {
    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
    cg::axpby(s, tmp, 1.0, &r2);
    cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
    cg::axpby(1.0, X, damp, &r2); // r2 = X + damp*r2  -- does NOT commute with the fold
    cg::axpby(s, tmpP, 1.0, &r2);
}

} // namespace

TEST_CASE("SymmetrizedAccumulation rewrite - fused result matches un-fused", "[ComputeGraph][SymmetrizedAccumulation]") {
    size_t const o       = 2;
    size_t const v       = 3;
    auto         A_typed = create_random_tensor<double>("A", o, v);
    auto         B_typed = create_random_tensor<double>("B", o, v);

    RuntimeTensor<double> const A(A_typed);
    RuntimeTensor<double> const B(B_typed);

    // Reference: capture + execute the un-fused idiom.
    RuntimeTensor<double> r2_ref("r2_ref", {o, o, v, v});
    RuntimeTensor<double> tmp_ref("tmp_ref", {o, o, v, v});
    RuntimeTensor<double> tmpP_ref("tmpP_ref", {o, o, v, v});
    r2_ref.zero();
    cg::Graph gref("symacc_ref");
    capture_symacc(gref, r2_ref, tmp_ref, tmpP_ref, A, B, 2.0);
    gref.execute();

    // Fused: same inputs, apply the pass, execute.
    RuntimeTensor<double> r2_fused("r2_fused", {o, o, v, v});
    RuntimeTensor<double> tmp_fused("tmp_fused", {o, o, v, v});
    RuntimeTensor<double> tmpP_fused("tmpP_fused", {o, o, v, v});
    r2_fused.zero();
    cg::Graph gfused("symacc_fused");
    capture_symacc(gfused, r2_fused, tmp_fused, tmpP_fused, A, B, 2.0);

    size_t const nodes_before = gfused.num_nodes();

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::SymmetrizedAccumulation>();
    pm.add(pass);
    gfused.apply(pm);

    CHECK(pass->num_matched() == 1);
    CHECK(pass->num_rewritten() == 1);
    CHECK(gfused.num_nodes() == nodes_before - 1); // one axpby folded away

    gfused.execute();

    REQUIRE(r2_fused.size() == r2_ref.size());
    for (size_t i = 0; i < r2_ref.size(); ++i) {
        CHECK(r2_fused.data()[i] == Catch::Approx(r2_ref.data()[i]));
    }
}

// Counters must report subtree TOTALS, not whatever the last-visited subgraph
// contributed. run() is invoked once per subgraph by the recursive driver, so a
// pass that zeroed its counters inside run() would report only the final
// subgraph -- silently right for a single loop (the body is visited last) and
// wrong for two sibling loops, which is what this builds. graph.explain()
// reads these getters.
TEST_CASE("SymmetrizedAccumulation - counters total across sibling subgraphs", "[ComputeGraph][SymmetrizedAccumulation]") {
    size_t const o       = 2;
    size_t const v       = 3;
    auto         A_typed = create_random_tensor<double>("A", o, v);
    auto         B_typed = create_random_tensor<double>("B", o, v);

    RuntimeTensor<double> const A(A_typed);
    RuntimeTensor<double> const B(B_typed);

    // Two sibling loops, one foldable site in each body.
    cg::Graph                          g("symacc_two_loops");
    std::vector<RuntimeTensor<double>> keep;
    keep.reserve(6);
    for (int k = 0; k < 2; ++k) {
        keep.emplace_back("r2", std::vector<size_t>{o, o, v, v});
        keep.emplace_back("tmp", std::vector<size_t>{o, o, v, v});
        keep.emplace_back("tmpP", std::vector<size_t>{o, o, v, v});
        auto &r2   = keep[keep.size() - 3];
        auto &tmp  = keep[keep.size() - 2];
        auto &tmpP = keep[keep.size() - 1];
        r2.zero();
        auto &body = g.add_loop(fmt::format("loop{}", k), 1, [](size_t) { return false; });
        capture_symacc(body, r2, tmp, tmpP, A, B, 1.0);
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::SymmetrizedAccumulation>();
    pm.add(pass);
    g.apply(pm);

    CHECK(pass->num_rewritten() == 2);
    CHECK(pass->num_matched() == 2);
    CHECK(pass->num_candidates() == 2);
}

// The interference guard admits an intervening node that reads AND writes r2 as
// an "additive accumulation that commutes with the fold". Folding moves the
// permuted contribution from the second axpby's position back to the permute's,
// which is only value-preserving when the intervening step leaves the existing
// r2 untouched -- i.e. beta == 1. A damping/mixing step (beta = 0.5) scales r2,
// so a contribution moved across it picks up a spurious factor of beta.
// beta == 0 is already safe: a pure overwrite does not list r2 as an input
// (Operations.hpp axpby), so the guard rejects it via touches_r2.
TEST_CASE("SymmetrizedAccumulation rewrite - damping in the window blocks the fold", "[ComputeGraph][SymmetrizedAccumulation]") {
    size_t const o       = 2;
    size_t const v       = 3;
    double const damp    = 0.5;
    auto         A_typed = create_random_tensor<double>("A", o, v);
    auto         B_typed = create_random_tensor<double>("B", o, v);
    auto         X_typed = create_random_tensor<double>("X", o, o, v, v);

    RuntimeTensor<double> const A(A_typed);
    RuntimeTensor<double> const B(B_typed);
    RuntimeTensor<double> const X(X_typed);

    RuntimeTensor<double> r2_ref("r2_ref", {o, o, v, v});
    RuntimeTensor<double> tmp_ref("tmp_ref", {o, o, v, v});
    RuntimeTensor<double> tmpP_ref("tmpP_ref", {o, o, v, v});
    r2_ref.zero();
    cg::Graph gref("symacc_damp_ref");
    capture_symacc_with_damping(gref, r2_ref, tmp_ref, tmpP_ref, A, B, X, 2.0, damp);
    gref.execute();

    RuntimeTensor<double> r2_fused("r2_fused", {o, o, v, v});
    RuntimeTensor<double> tmp_fused("tmp_fused", {o, o, v, v});
    RuntimeTensor<double> tmpP_fused("tmpP_fused", {o, o, v, v});
    r2_fused.zero();
    cg::Graph gfused("symacc_damp_fused");
    capture_symacc_with_damping(gfused, r2_fused, tmp_fused, tmpP_fused, A, B, X, 2.0, damp);

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::SymmetrizedAccumulation>();
    pm.add(pass);
    gfused.apply(pm);

    // The site is a structural match; the guard must refuse to rewrite it.
    CHECK(pass->num_rewritten() == 0);

    gfused.execute();

    REQUIRE(r2_fused.size() == r2_ref.size());
    for (size_t i = 0; i < r2_ref.size(); ++i) {
        CHECK(r2_fused.data()[i] == Catch::Approx(r2_ref.data()[i]));
    }
}
