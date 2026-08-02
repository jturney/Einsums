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

TEST_CASE("SymmetrizedAccumulation matcher - sites sharing one scratch pair match per generation",
          "[ComputeGraph][SymmetrizedAccumulation]") {
    // The CCSD body's actual shape: ONE tmp/tmpP recycled by every site. Each
    // overwrite of tmpP starts a new generation, and the matcher pairs each
    // permute with the consumer of ITS generation instead of demanding
    // whole-graph writer/reader uniqueness (which rejected every site).
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-shared-scratch");
    auto     &tmp  = graph.declare_tensor<double, 4>("tmp", 2, 2, 3, 3);
    auto     &tmpP = graph.declare_tensor<double, 4>("tmpP", 2, 2, 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        for (int site = 0; site < 3; ++site) {
            cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
            cg::axpby(1.0, tmp, 1.0, &r2);
            cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
            cg::axpby(1.0, tmpP, 1.0, &r2);
        }
    }

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

// `r2 += tmp` is an axpy in every other library, so cg::axpy is the spelling a
// user reaches for first - and for a long time it could never match, because it
// recorded a separate opaque node kind and the fold needs to read the
// accumulate's scalar. cg::axpy now records an Axpby with beta == 1, so both
// spellings match identically. Regression guard for the spelling, not the
// algebra: the rewrite itself is covered next door.
TEST_CASE("SymmetrizedAccumulation matcher - the axpy spelling matches like axpby", "[ComputeGraph][SymmetrizedAccumulation]") {
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-axpy-spelling");
    auto     &tmp  = graph.declare_tensor<double, 4>("tmp", 2, 2, 3, 3);
    auto     &tmpP = graph.declare_tensor<double, 4>("tmpP", 2, 2, 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
        cg::axpy(1.0, tmp, &r2);                       // the natural spelling
        cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp); // involution
        cg::axpy(1.0, tmpP, &r2);                      // the natural spelling
    }

    auto pass = match(graph);
    CHECK(pass->num_candidates() == 1);
    CHECK(pass->num_matched() == 1);
}

// The tally is per-apply state: a second apply() must not report the first
// apply's rejections on top of its own. Uses a shape the pass declines (only
// the permuted half accumulates) so there is something to count.
TEST_CASE("SymmetrizedAccumulation - skip reasons reset between applies", "[ComputeGraph][SymmetrizedAccumulation]") {
    auto A  = create_random_tensor<double>("A", 2, 3);
    auto B  = create_random_tensor<double>("B", 2, 3);
    auto r2 = create_zero_tensor<double>("r2", 2, 2, 3, 3);

    cg::Graph graph("symacc-reset");
    auto     &tmp  = graph.declare_tensor<double, 4>("tmp", 2, 2, 3, 3);
    auto     &tmpP = graph.declare_tensor<double, 4>("tmpP", 2, 2, 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j,a,b <- i,a ; j,b", &tmp, A, B);
        cg::permute("j,i,b,a <- i,j,a,b", &tmpP, tmp);
        cg::axpby(1.0, tmpP, 1.0, &r2); // no sibling accumulate of the un-permuted tmp
    }

    cg::PassManager pm;
    auto            pass = std::make_shared<cg::passes::SymmetrizedAccumulation>();
    pm.add(pass);

    graph.apply(pm);
    auto const first = pass->skip_reasons();
    REQUIRE_FALSE(first.empty());

    graph.apply(pm);
    auto const second = pass->skip_reasons();
    REQUIRE(second.size() == first.size());
    for (size_t n = 0; n < first.size(); n++) {
        CHECK(second[n].first == first[n].first);
        CHECK(second[n].second == first[n].second); // counts, not accumulated totals
    }
}

// The descriptor must not be a snapshot the executor disagrees with. axpy's
// executor reads the shared params, so rewriting alpha there changes what
// replay computes - the property that makes the scalar safe for a pass to fold
// into, and the thing a baked-in lambda constant silently broke.
TEST_CASE("axpy - descriptor scalars are live, and beta != 1 is honored", "[ComputeGraph][Operations]") {
    auto X = create_random_tensor<double>("X", 4);
    auto Y = create_zero_tensor<double>("Y", 4);

    cg::Graph graph("axpy-live-params");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(2.0, X, &Y);
    }

    REQUIRE(graph.num_nodes() == 1);
    auto *desc = std::get_if<cg::AxpbyDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->params != nullptr);
    CHECK(cg::is_one(desc->params->beta));

    graph.execute();
    for (size_t n = 0; n < 4; n++) {
        REQUIRE(std::abs(Y(n) - 2.0 * X(n)) < 1e-12);
    }

    // Rewrite alpha through the params, as ScaleAbsorption/CSE do.
    desc->params->alpha = cg::PrefactorScalar{5.0};
    Y.zero();
    graph.execute();
    for (size_t n = 0; n < 4; n++) {
        REQUIRE(std::abs(Y(n) - 5.0 * X(n)) < 1e-12);
    }

    // Rewriting beta turns it into a genuine axpby; the executor must follow
    // rather than keep calling the beta == 1 fast path.
    desc->params->beta = cg::PrefactorScalar{2.0};
    for (size_t n = 0; n < 4; n++) {
        Y(n) = 1.0;
    }
    graph.execute(); // Y = 5*X + 2*1
    for (size_t n = 0; n < 4; n++) {
        REQUIRE(std::abs(Y(n) - (5.0 * X(n) + 2.0)) < 1e-12);
    }
}
