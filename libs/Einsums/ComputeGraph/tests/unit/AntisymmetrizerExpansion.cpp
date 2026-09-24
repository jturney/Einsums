//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/AntisymmetrizerExpansion.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <memory>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Count the nodes of one kind, which is how the rewrite is observed: the pass
/// turns one einsum into an einsum plus one permute per term.
std::size_t count_kind(cg::Graph const &graph, cg::OpKind kind) {
    std::size_t n = 0;
    for (auto const &node : graph.nodes()) {
        if (node.kind == kind) {
            ++n;
        }
    }
    return n;
}

} // namespace

TEST_CASE("AntisymmetrizerExpansion - P(ij)P(ab) lowers to a contraction plus four accumulations",
          "[ComputeGraph][AntisymmetrizerExpansion]") {
    size_t const no = 3, nv = 4;
    auto         t2 = create_random_tensor<double>("t2", no, no, nv, nv);
    auto         F  = create_random_tensor<double>("F", no, no);
    auto         C  = create_zero_tensor<double>("C", no, no, nv, nv);

    cg::Graph graph("asym");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,b ; m,j", 0.0, &C, 1.0, t2, F);
    }
    REQUIRE(count_kind(graph, cg::OpKind::Einsum) == 1);
    REQUIRE(count_kind(graph, cg::OpKind::Permute) == 0);

    auto            pass = std::make_shared<cg::passes::AntisymmetrizerExpansion>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    CHECK(pass->num_sites() == 1);
    CHECK(pass->num_expanded() == 1);
    CHECK(pass->num_terms() == 4);
    CHECK(count_kind(graph, cg::OpKind::Einsum) == 1);
    CHECK(count_kind(graph, cg::OpKind::Permute) == 4);

    // The lowered einsum must no longer carry the operator, or the executor
    // would expand it a second time and the answer would be the square of the
    // antisymmetrizer rather than the antisymmetrizer.
    for (auto const &node : graph.nodes()) {
        if (node.kind != cg::OpKind::Einsum) {
            continue;
        }
        auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
        REQUIRE(desc != nullptr);
        CHECK(desc->operators.empty());
        if (desc->indices != nullptr) {
            CHECK(desc->indices->spec.operators.empty());
        }
    }
}

TEST_CASE("AntisymmetrizerExpansion - the lowered graph computes the same numbers", "[ComputeGraph][AntisymmetrizerExpansion]") {
    size_t const no = 3, nv = 4;
    auto         t2 = create_random_tensor<double>("t2", no, no, nv, nv);
    auto         W  = create_random_tensor<double>("W", no, nv, nv, no);
    auto         C0 = create_random_tensor<double>("C0", no, no, nv, nv);

    // Accumulating into a non-zero C, with a non-unit product prefactor, so the
    // test would catch the prefactor landing once per term rather than once.
    double const c_pf = 0.5, ab_pf = -1.5;

    auto unlowered = create_zero_tensor<double>("unlowered", no, no, nv, nv);
    unlowered      = C0;
    cg::Graph plain("plain");
    {
        cg::CaptureGuard const capture(plain);
        cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,e ; m,b,e,j", c_pf, &unlowered, ab_pf, t2, W);
    }
    plain.execute();

    auto lowered = create_zero_tensor<double>("lowered", no, no, nv, nv);
    lowered      = C0;
    cg::Graph expanded("expanded");
    {
        cg::CaptureGuard const capture(expanded);
        cg::einsum("i,j,a,b <- P(ij) P(ab) i,m,a,e ; m,b,e,j", c_pf, &lowered, ab_pf, t2, W);
    }
    auto            pass = std::make_shared<cg::passes::AntisymmetrizerExpansion>();
    cg::PassManager manager;
    manager.add(pass);
    expanded.apply(manager);
    REQUIRE(pass->num_expanded() == 1);
    expanded.execute();

    // BitwiseExact is the declared tier: the lowered form runs the same kernels
    // on the same values in the same order, so this is an exact comparison and
    // not a tolerance.
    for (size_t ii = 0; ii < no; ii++) {
        for (size_t jj = 0; jj < no; jj++) {
            for (size_t aa = 0; aa < nv; aa++) {
                for (size_t bb = 0; bb < nv; bb++) {
                    REQUIRE(lowered(ii, jj, aa, bb) == unlowered(ii, jj, aa, bb));
                }
            }
        }
    }
}

TEST_CASE("AntisymmetrizerExpansion - the default pipeline lowers and still computes", "[ComputeGraph][AntisymmetrizerExpansion]") {
    size_t const no = 3, nv = 4;
    auto         t2 = create_random_tensor<double>("t2", no, no, nv, nv);
    auto         F  = create_random_tensor<double>("F", no, no);

    auto eager = create_zero_tensor<double>("eager", no, no, nv, nv);
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("i,j,a,b <- P(i/j) i,m,a,b ; m,j", 0.0, &eager, 1.0, t2, F);

    auto      replayed = create_zero_tensor<double>("replayed", no, no, nv, nv);
    cg::Graph graph("default_pipeline");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,a,b <- P(i/j) i,m,a,b ; m,j", 0.0, &replayed, 1.0, t2, F);
    }
    cg::PassManager pipeline = cg::PassManager::create_default();
    graph.apply(pipeline);
    graph.execute();

    for (size_t ii = 0; ii < no; ii++) {
        for (size_t jj = 0; jj < no; jj++) {
            for (size_t aa = 0; aa < nv; aa++) {
                for (size_t bb = 0; bb < nv; bb++) {
                    REQUIRE_THAT(replayed(ii, jj, aa, bb), Catch::Matchers::WithinAbs(eager(ii, jj, aa, bb), 1e-12));
                }
            }
        }
    }
}

TEST_CASE("AntisymmetrizerExpansion - a permute's operator is left alone", "[ComputeGraph][AntisymmetrizerExpansion]") {
    // Lowering a permute would replace one node with N doing the same work in the
    // same order, and the N share one source and one destination so there is
    // nothing downstream to gain from seeing them apart. Declining is the
    // behaviour, not an omission.
    size_t const n = 3;
    auto         X = create_random_tensor<double>("X", n, n, n, n, n, n);
    auto         C = create_zero_tensor<double>("C", n, n, n, n, n, n);

    cg::Graph graph("permute_operator");
    {
        cg::CaptureGuard const capture(graph);
        cg::permute("i,j,k,a,b,c <- P(i/jk) P(a/bc) i,j,k,a,b,c", 0.0, &C, 1.0, X);
    }
    auto            pass = std::make_shared<cg::passes::AntisymmetrizerExpansion>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    CHECK(pass->num_sites() == 0);
    CHECK(pass->num_expanded() == 0);
    CHECK(count_kind(graph, cg::OpKind::Permute) == 1);
}

TEST_CASE("AntisymmetrizerExpansion - a graph with no operator is untouched", "[ComputeGraph][AntisymmetrizerExpansion]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("no_operator");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }
    std::size_t const before = graph.num_nodes();

    auto            pass = std::make_shared<cg::passes::AntisymmetrizerExpansion>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    CHECK(pass->num_sites() == 0);
    CHECK(pass->num_expanded() == 0);
    CHECK(graph.num_nodes() == before);
    CHECK(pass->explain().empty());
}

TEST_CASE("AntisymmetrizerExpansion - declines a contraction whose operands differ in element type",
          "[ComputeGraph][AntisymmetrizerExpansion][MixedPrecision]") {
    // A mixed-precision einsum with an operator is refused where it is built, so the only way here
    // is a pass that edits the operator into a mixed node, or a loaded graph. The expansion's
    // temporary takes C's type and its terms go through the same-type permute, so it declines.
    size_t const no = 3, nv = 4;
    auto         t2 = create_random_tensor<float>("t2", no, no, nv, nv);
    auto         F  = create_random_tensor<double>("F", no, no);
    auto         C  = create_zero_tensor<double>("C", no, no, nv, nv);

    cg::Graph graph("asym_mixed");
    {
        cg::CaptureGuard const capture(graph);
        cg::einsum("i,j,a,b <- i,m,a,b ; m,j", 0.0, &C, 1.0, t2, F);
    }
    auto const operators = cg::parse_einsum_spec("i,j,a,b <- P(ij) P(ab) i,m,a,b ; m,j").value().operators;
    for (auto &node : graph.nodes()) {
        if (auto *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data); desc != nullptr) {
            desc->indices->spec.operators = operators;
        }
    }

    auto            pass = std::make_shared<cg::passes::AntisymmetrizerExpansion>();
    cg::PassManager manager;
    manager.add(pass);
    graph.apply(manager);

    CHECK(pass->num_expanded() == 0);
    CHECK(count_kind(graph, cg::OpKind::Permute) == 0);
    auto const reasons = pass->skip_reasons();
    REQUIRE(reasons.size() == 1);
    CHECK(reasons[0].first.find("different element types") != std::string::npos);
}
