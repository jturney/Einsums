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
