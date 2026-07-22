//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Matcher-only slice for the SymmetrizedAccumulation pass: assert it DETECTS the
// CCSD `symacc` idiom (einsum -> tmp; axpby(tmp -> r2); permute(jiba<-ijab);
// axpby(tmpP -> r2)) as a foldable site, on typed captures, before any rewrite
// exists. See docs/symmetrized_accumulation_design.md.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>
#include <string>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

// Capture one symacc site into `graph`: r2 += (tmp + P(tmp)), tmp = A (outer) B.
// `tag` keeps the intermediate names unique across sites in one graph.
void capture_symacc_site(cg::Graph &graph, Tensor<double, 4> &r2, Tensor<double, 2> const &A, Tensor<double, 2> const &B,
                         std::string const &tag) {
    auto &tmp  = graph.declare_tensor<double, 4>("tmp" + tag, A.dim(0), B.dim(0), A.dim(1), B.dim(1));
    auto &tmpP = graph.declare_tensor<double, 4>("tmpP" + tag, B.dim(0), A.dim(0), B.dim(1), A.dim(1));

    cg::CaptureGuard const guard(graph);
    cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B); // outer product -> tmp
    cg::axpby(1.0, tmp, 1.0, &r2);                  // r2 += tmp
    cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);  // tmpP = P(tmp)  (involution)
    cg::axpby(1.0, tmpP, 1.0, &r2);                 // r2 += P(tmp)
}

std::shared_ptr<cg::passes::SymmetrizedAccumulation> match(cg::Graph &graph) {
    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::SymmetrizedAccumulation>();
    pm.add(pass);
    graph.apply(pm);
    return pass;
}

} // namespace

TEST_CASE("SymmetrizedAccumulation matcher - single symacc site", "[ComputeGraph][SymmetrizedAccumulation]") {
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-one");
    capture_symacc_site(graph, r2, A, B, "0");

    auto pass = match(graph);
    CHECK(pass->num_candidates() == 1);
    CHECK(pass->num_matched() == 1);
}

TEST_CASE("SymmetrizedAccumulation matcher - multiple sites into one output", "[ComputeGraph][SymmetrizedAccumulation]") {
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-many");
    capture_symacc_site(graph, r2, A, B, "0");
    capture_symacc_site(graph, r2, A, B, "1");
    capture_symacc_site(graph, r2, A, B, "2");

    auto pass = match(graph);
    CHECK(pass->num_candidates() == 3);
    CHECK(pass->num_matched() == 3);
}

TEST_CASE("SymmetrizedAccumulation matcher - a lone permute+axpby is not a site", "[ComputeGraph][SymmetrizedAccumulation]") {
    // Only the permuted branch (no sibling axpby over the un-permuted tmp): not
    // the symmetrization idiom, so nothing matches.
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-negative");
    auto     &tmp  = graph.declare_tensor<double, 4>("tmp", 2, 2, 3, 3);
    auto     &tmpP = graph.declare_tensor<double, 4>("tmpP", 2, 2, 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
        cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
        cg::axpby(1.0, tmpP, 1.0, &r2); // only the permuted branch
    }

    auto pass = match(graph);
    CHECK(pass->num_candidates() == 0);
    CHECK(pass->num_matched() == 0);
}
