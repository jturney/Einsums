//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Level-1 rewrite of the SymmetrizedAccumulation pass: fold the symacc idiom by
// making the permute accumulate directly into r2 (r2 += s2*alpha*P(tmp)) and
// dropping the second axpby + the tmpP buffer. The fused graph must produce the
// same r2 as the un-fused one, and the brute-force reference where one is given.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <complex>
#include <memory>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// r2 += s*(tmp + P(tmp)), tmp = A (outer) B, as four captured nodes.
//
// tmpP is graph-owned scratch, which is what the fold requires: it stops writing tmpP, so a tensor
// the caller holds would be left stale.
void capture_symacc(cg::Graph &graph, RuntimeTensor<double> &r2, RuntimeTensor<double> &tmp, RuntimeTensor<double> const &A,
                    RuntimeTensor<double> const &B, double s) {
    auto                  &tmpP = graph.create_zero_runtime_tensor<double>("tmpP", tmp.dims());
    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
    cg::axpby(s, tmp, 1.0, &r2);
    cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
    cg::axpby(s, tmpP, 1.0, &r2);
}

// Same idiom, but with a DAMPING step (r2 = X + damp*r2) landing inside the fold
// window, between the permute and the second axpby. Mixing/damping with
// beta not in {0, 1} is routine in SCF and DIIS-driven codes.
void capture_symacc_with_damping(cg::Graph &graph, RuntimeTensor<double> &r2, RuntimeTensor<double> &tmp, RuntimeTensor<double> const &A,
                                 RuntimeTensor<double> const &B, RuntimeTensor<double> const &X, double s, double damp) {
    auto                  &tmpP = graph.create_zero_runtime_tensor<double>("tmpP", tmp.dims());
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
    r2_ref.zero();
    cg::Graph gref("symacc_ref");
    capture_symacc(gref, r2_ref, tmp_ref, A, B, 2.0);
    gref.execute();

    // Fused: same inputs, apply the pass, execute.
    RuntimeTensor<double> r2_fused("r2_fused", {o, o, v, v});
    RuntimeTensor<double> tmp_fused("tmp_fused", {o, o, v, v});
    r2_fused.zero();
    cg::Graph gfused("symacc_fused");
    capture_symacc(gfused, r2_fused, tmp_fused, A, B, 2.0);

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
        auto &r2  = keep[keep.size() - 2];
        auto &tmp = keep[keep.size() - 1];
        r2.zero();
        auto &body = g.add_loop(fmt::format("loop{}", k), 1, [](size_t) { return false; });
        capture_symacc(body, r2, tmp, A, B, 1.0);
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
    r2_ref.zero();
    cg::Graph gref("symacc_damp_ref");
    capture_symacc_with_damping(gref, r2_ref, tmp_ref, A, B, X, 2.0, damp);
    gref.execute();

    RuntimeTensor<double> r2_fused("r2_fused", {o, o, v, v});
    RuntimeTensor<double> tmp_fused("tmp_fused", {o, o, v, v});
    r2_fused.zero();
    cg::Graph gfused("symacc_damp_fused");
    capture_symacc_with_damping(gfused, r2_fused, tmp_fused, A, B, X, 2.0, damp);

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

namespace {

/// @p r2 against s*(X + alpha*P(X)) with X = A (outer) B, from the brute-force reference.
void check_against_reference(Tensor<double, 4> const &r2, Tensor<double, 2> const &A, Tensor<double, 2> const &B, double s, double alpha) {
    auto X = create_zero_tensor<double>("X", A.dim(0), B.dim(0), A.dim(1), B.dim(1));
    einsums::testing::reference_einsum("i,j,a,b <- i,a ; j,b", 0.0, &X, 1.0, A, B);
    auto want = create_zero_tensor<double>("want", A.dim(0), B.dim(0), A.dim(1), B.dim(1));
    einsums::testing::reference_permute("i,j,a,b <- i,j,a,b", 0.0, &want, s, X);
    einsums::testing::reference_permute("j,i,b,a <- i,j,a,b", 1.0, &want, s * alpha, X);
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(r2.data()[i] == Catch::Approx(want.data()[i]).margin(1e-12));
    }
}

} // namespace

TEST_CASE("SymmetrizedAccumulation rewrite - typed tensors and a scaled permute fold", "[ComputeGraph][SymmetrizedAccumulation]") {
    // Typed Tensor<T, Rank> operands and a permute with alpha = 3. Both used to be declined: the
    // folded executor was hand-built and cast its operands to runtime tensors, and it had no
    // place for the permute's own scale. The folded node is now the library's own permute, read
    // through each operand's impl, carrying s2 * alpha.
    size_t const o = 2, v = 3;
    double const s = 0.5, alpha = 3.0;
    auto         A   = create_random_tensor<double>("A", o, v);
    auto         B   = create_random_tensor<double>("B", o, v);
    auto         tmp = create_zero_tensor<double>("tmp", o, o, v, v);
    auto         r2  = create_zero_tensor<double>("r2", o, o, v, v);

    cg::Graph graph("symacc_typed");
    auto     &tmpP = graph.create_zero_tensor<double, 4>("tmpP", o, o, v, v);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
        cg::axpby(s, tmp, 1.0, &r2);
        cg::permute("j,i,b,a <- i,j,a,b", 0.0, &tmpP, alpha, tmp);
        cg::axpby(s, tmpP, 1.0, &r2);
    }

    auto [modified, pass] = graph.apply<cg::passes::SymmetrizedAccumulation>();
    CHECK(modified);
    CHECK(pass.num_rewritten() == 1);

    graph.execute();
    check_against_reference(r2, A, B, s, alpha);
}

TEST_CASE("SymmetrizedAccumulation rewrite - a source read through a view folds", "[ComputeGraph][SymmetrizedAccumulation][Views]") {
    // tmp is a block of a larger scratch tensor. The folded permute reads it through the view's
    // own impl, strides and offset included.
    size_t const o = 2, v = 3;
    double const s  = 0.25;
    auto         A  = create_random_tensor<double>("A", o, v);
    auto         B  = create_random_tensor<double>("B", o, v);
    auto         r2 = create_zero_tensor<double>("r2", o, o, v, v);

    cg::Graph graph("symacc_view_source");
    auto     &big  = graph.create_zero_tensor<double, 4>("big", o + 1, o, v, v + 2);
    auto     &tmpP = graph.create_zero_tensor<double, 4>("tmpP", o, o, v, v);
    {
        cg::CaptureGuard const guard(graph);
        auto &tmp = cg::view(big, cg::ViewAxis::range(1, o + 1), cg::ViewAxis::full(), cg::ViewAxis::full(), cg::ViewAxis::range(1, v + 1));
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
        cg::axpby(s, tmp, 1.0, &r2);
        cg::permute("j,i,b,a <- i,j,a,b", 0.0, &tmpP, 1.0, tmp);
        cg::axpby(s, tmpP, 1.0, &r2);
    }

    auto [modified, pass] = graph.apply<cg::passes::SymmetrizedAccumulation>();
    CHECK(pass.num_rewritten() == 1);

    graph.execute();
    check_against_reference(r2, A, B, s, 1.0);
}

TEST_CASE("SymmetrizedAccumulation rewrite - a source laid over the output is not folded",
          "[ComputeGraph][SymmetrizedAccumulation][Views]") {
    // tmp is a whole view of r2. As captured, the transpose reads r2 into tmpP before the second
    // accumulate writes r2. Folded, one node would read r2 while writing its transpose into r2.
    size_t const o = 2, v = 3;
    double const s  = 0.5;
    auto         A  = create_random_tensor<double>("A", o, v);
    auto         B  = create_random_tensor<double>("B", o, v);
    auto         r2 = create_zero_tensor<double>("r2", o, o, v, v);

    cg::Graph graph("symacc_source_over_output");
    auto     &tmpP = graph.create_zero_tensor<double, 4>("tmpP", o, o, v, v);
    {
        cg::CaptureGuard const guard(graph);
        auto                  &tmp = cg::view(r2, cg::ViewAxis::full(), cg::ViewAxis::full(), cg::ViewAxis::full(), cg::ViewAxis::full());
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B); // r2 = X
        cg::axpby(s, tmp, 1.0, &r2);                    // r2 = (1 + s) X
        cg::permute("j,i,b,a <- i,j,a,b", 0.0, &tmpP, 1.0, tmp);
        cg::axpby(s, tmpP, 1.0, &r2); // r2 = (1 + s) (X + s P(X))
    }

    auto [modified, pass] = graph.apply<cg::passes::SymmetrizedAccumulation>();
    CHECK(pass.num_rewritten() == 0);

    graph.execute();
    check_against_reference(r2, A, B, 1.0 + s, s);
}

TEST_CASE("SymmetrizedAccumulation rewrite - a real output is not folded under imaginary factors",
          "[ComputeGraph][SymmetrizedAccumulation]") {
    // Both scalars of a float64 site set to i, as a loaded file can carry them. As captured, the
    // replay refuses the first one it meets (a real tensor cannot be scaled by i). Their product
    // is -1, so a fold that asked only about the product would replace that refusal with an
    // answer.
    size_t const          o = 2, v = 3;
    auto                  A_typed = create_random_tensor<double>("A", o, v);
    auto                  B_typed = create_random_tensor<double>("B", o, v);
    RuntimeTensor<double> A(A_typed), B(B_typed);
    RuntimeTensor<double> r2("r2", {o, o, v, v});
    RuntimeTensor<double> tmp("tmp", {o, o, v, v});
    r2.zero();

    cg::Graph graph("symacc_imaginary");
    capture_symacc(graph, r2, tmp, A, B, 1.0);
    std::complex<double> const i{0.0, 1.0};
    for (auto &node : graph.nodes()) {
        if (auto *pd = node.op_data.get_if<cg::PermuteDescriptor>(); pd != nullptr) {
            pd->alpha         = i;
            pd->params->alpha = cg::PrefactorScalar{i};
        }
        if (auto *ad = node.op_data.get_if<cg::AxpbyDescriptor>(); ad != nullptr && graph.find_tensor(node.inputs[0])->name == "tmpP") {
            ad->alpha         = cg::PrefactorScalar{i};
            ad->params->alpha = cg::PrefactorScalar{i};
        }
    }

    auto [modified, pass] = graph.apply<cg::passes::SymmetrizedAccumulation>();
    CHECK(pass.num_matched() == 1);
    CHECK(pass.num_rewritten() == 0);
    CHECK_THROWS(graph.execute());
}
