//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The commit rules of Graph::edit: what a NodeEdit does for a pass when it ends, so the pass only
// has to describe its change.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t n = 4;

/// The position of the first node of @p kind.
std::size_t position_of(cg::Graph const &graph, cg::OpKind kind) {
    for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
        if (graph.nodes()[i].kind == kind) {
            return i;
        }
    }
    FAIL("no node of the kind asked for");
    return 0;
}

void check_equal(Tensor<double, 2> const &got, Tensor<double, 2> const &want) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            CHECK_THAT(got(i, j), Catch::Matchers::WithinRel(want(i, j), 1e-12));
        }
    }
}

/// C = A B and D = A2 B, so the second einsum names A2 for the first to be repointed at.
struct TwoProducts {
    Tensor<double, 2> A  = create_random_tensor<double>("A", n, n);
    Tensor<double, 2> A2 = create_random_tensor<double>("A2", n, n);
    Tensor<double, 2> B  = create_random_tensor<double>("B", n, n);
    Tensor<double, 2> C  = create_zero_tensor<double>("C", n, n);
    Tensor<double, 2> D  = create_zero_tensor<double>("D", n, n);
    cg::Graph         graph{"two_products"};

    TwoProducts() {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &D, 1.0, A2, B);
    }

    [[nodiscard]] cg::TensorId a2_id() const { return graph.nodes()[1].inputs[0]; }

    [[nodiscard]] Tensor<double, 2> expected(Tensor<double, 2> const &left) const {
        Tensor<double, 2> want("want", n, n);
        testing::reference_einsum("ij <- ik ; kj", 0.0, &want, 1.0, left, B);
        return want;
    }
};

} // namespace

TEST_CASE("NodeEdit - a repointed input is the one the replay reads", "[ComputeGraph][NodeEdit]") {
    // The 2026-09-17 fold bug and the CSE slot bug: an operand list rewritten under an executor
    // that still closed over the old operand. The commit rebuilds the executor from the fields.
    TwoProducts p;
    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });
    p.graph.execute();
    check_equal(p.C, p.expected(p.A2));
}

TEST_CASE("NodeEdit - usage sees the edit", "[ComputeGraph][NodeEdit]") {
    TwoProducts        p;
    cg::TensorId const a = p.graph.nodes()[0].inputs[0];
    REQUIRE(p.graph.usage().find_owner(a) != nullptr);
    REQUIRE(p.graph.usage().find_owner(a)->first_use() == 0);

    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });

    auto const *use = p.graph.usage().find_owner(a);
    CHECK((use == nullptr || use->first_use() == cg::TensorUsage::npos));
    CHECK(p.graph.usage().find_owner(p.a2_id())->first_use() == 0);
}

TEST_CASE("NodeEdit - a replaced descriptor is the one the replay applies", "[ComputeGraph][NodeEdit]") {
    // A fresh descriptor with no live parameters: only a rebuild from it can apply the new factor.
    auto              A  = create_random_tensor<double>("A", n, n);
    Tensor<double, 2> A0 = A;
    cg::Graph         graph("rescale");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }
    graph.edit(position_of(graph, cg::OpKind::Scale), [](cg::NodeEdit &e) {
        e->op_data = cg::OpData{cg::ScaleDescriptor{.factor = cg::PrefactorScalar{5.0}, .params = nullptr}};
    });
    graph.execute();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            CHECK(A(i, j) == 5.0 * A0(i, j));
        }
    }
}

TEST_CASE("NodeEdit - the destination is listed again when the node reads it", "[ComputeGraph][NodeEdit]") {
    auto const A = create_random_tensor<double>("A", n, n);
    auto const B = create_random_tensor<double>("B", n, n);
    auto       C = create_random_tensor<double>("C", n, n);
    cg::Graph  graph("accumulate");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B); // reads C
    }
    std::size_t const at = position_of(graph, cg::OpKind::Einsum);
    REQUIRE(graph.nodes()[at].inputs.size() == 3);
    graph.edit(at, [](cg::NodeEdit &e) { e->inputs.pop_back(); });
    CHECK(graph.nodes()[at].inputs.size() == 3);
    CHECK(graph.nodes()[at].inputs.back() == graph.nodes()[at].outputs.front());
}

TEST_CASE("NodeEdit - a pass-written executor is kept until the operands change", "[ComputeGraph][NodeEdit]") {
    TwoProducts p;
    int         runs = 0;
    p.graph.edit(0, [&](cg::NodeEdit &e) { e.set_executor([&runs] { ++runs; }); });
    p.graph.execute();
    CHECK(runs == 1);

    // An edit that leaves the operands alone keeps it.
    p.graph.edit(0, [](cg::NodeEdit &e) { e->label = "renamed"; });
    p.graph.execute();
    CHECK(runs == 2);

    // One that changes them without a new executor is refused, and the node is left as it was.
    std::vector<cg::TensorId> const inputs = p.graph.nodes()[0].inputs;
    CHECK_THROWS_AS(p.graph.edit(0,
                                 [&](cg::NodeEdit &e) {
                                     e->label     = "repointed";
                                     e->inputs[0] = p.a2_id();
                                 }),
                    std::logic_error);
    CHECK(p.graph.nodes()[0].inputs == inputs);
    CHECK(p.graph.nodes()[0].label == "renamed");
    p.graph.execute();
    CHECK(runs == 3);

    // With a new one it goes through.
    int other = 0;
    p.graph.edit(0, [&](cg::NodeEdit &e) {
        e->inputs[0] = p.a2_id();
        e.set_executor([&other] { ++other; });
    });
    p.graph.execute();
    CHECK(runs == 3);
    CHECK(other == 1);
}

TEST_CASE("NodeEdit - assigning the executor directly is refused", "[ComputeGraph][NodeEdit]") {
    TwoProducts p;
    CHECK_THROWS_AS(p.graph.edit(0,
                                 [](cg::NodeEdit &e) {
                                     e->label   = "changed";
                                     e->execute = [] {};
                                 }),
                    std::logic_error);
    CHECK(p.graph.nodes()[0].label != "changed");
    p.graph.execute();
    check_equal(p.C, p.expected(p.A));
}

TEST_CASE("NodeEdit - a change that throws leaves the node as it was", "[ComputeGraph][NodeEdit]") {
    TwoProducts p;
    CHECK_THROWS_AS(p.graph.edit(0,
                                 [&](cg::NodeEdit &e) {
                                     e->inputs[0] = p.a2_id();
                                     throw std::runtime_error("changed its mind");
                                 }),
                    std::runtime_error);
    p.graph.execute();
    check_equal(p.C, p.expected(p.A));
    CHECK_THROWS_AS(p.graph.edit(99, [](cg::NodeEdit &) {}), std::out_of_range);
}

TEST_CASE("NodeEdit - a node with no executor the builder can make needs one given", "[ComputeGraph][NodeEdit]") {
    cg::Graph graph("hand_written");
    cg::Node  node;
    node.kind    = cg::OpKind::Custom;
    node.label   = "hand-written";
    node.execute = [] {};
    graph.add_node(std::move(node));
    CHECK_THROWS_WITH(graph.edit(0, [](cg::NodeEdit &e) { e->label = "renamed"; }), Catch::Matchers::ContainsSubstring("set_executor"));
    CHECK(graph.nodes()[0].label == "hand-written");
    CHECK(static_cast<bool>(graph.nodes()[0].execute));

    // Marked as its pass's own, it edits like any other.
    int runs = 0;
    graph.edit(0, [&](cg::NodeEdit &e) { e.set_executor([&runs] { ++runs; }); });
    graph.edit(0, [](cg::NodeEdit &e) { e->label = "renamed"; });
    graph.execute();
    CHECK(runs == 1);
}

TEST_CASE("NodeEdit - a folded node stays inert through later edits", "[ComputeGraph][NodeEdit]") {
    TwoProducts p;
    p.C.zero();
    p.graph.edit(0, [](cg::NodeEdit &e) { e.fold(); });
    p.graph.execute();
    CHECK(p.C(0, 0) == 0.0);

    // A later pass repoints it; it is still folded, so it still does nothing.
    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });
    p.graph.execute();
    CHECK(p.C(0, 0) == 0.0);
    CHECK(p.graph.nodes()[0].folded);
}

TEST_CASE("NodeEdit - pre-run hooks run first, in order, through later edits", "[ComputeGraph][NodeEdit]") {
    TwoProducts      p;
    std::vector<int> order;
    p.graph.edit(0, [&](cg::NodeEdit &e) {
        e.add_pre_run([&order] { order.push_back(1); });
        e.add_pre_run([&order] { order.push_back(2); });
    });
    p.graph.execute();
    CHECK(order == std::vector<int>{1, 2});
    check_equal(p.C, p.expected(p.A));

    // A later edit rebuilds the executor and keeps the hooks ahead of it, once each.
    order.clear();
    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });
    p.graph.execute();
    CHECK(order == std::vector<int>{1, 2});
    check_equal(p.C, p.expected(p.A2));
}

TEST_CASE("NodeEdit - a placed node's CPU fallback follows the rebuilt executor", "[ComputeGraph][NodeEdit]") {
    // GPUPlacement copies the executor into cpu_fallback; a commit that rebuilt only execute would
    // leave the fallback running the old closure.
    TwoProducts p;
    {
        // What placement does, without a device to place on.
        auto &node        = const_cast<cg::Node &>(p.graph.nodes()[0]);
        node.cpu_fallback = node.execute;
    }
    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });
    p.C.zero();
    p.graph.nodes()[0].cpu_fallback();
    check_equal(p.C, p.expected(p.A2));
}

TEST_CASE("NodeEdit - a pass-written executor keeps its own state through edits", "[ComputeGraph][NodeEdit]") {
    // cg::custom and its kin hold the caller's functor by value in a mutable lambda. A commit that
    // copied the executor would start each edit's copy from the state it was given in.
    TwoProducts p;
    int         seen = 0;
    p.graph.edit(0, [&](cg::NodeEdit &e) { e.set_executor([count = 0, &seen]() mutable { seen = ++count; }); });
    p.graph.execute();
    CHECK(seen == 1);
    p.graph.edit(0, [](cg::NodeEdit &e) { e->label = "renamed"; });
    p.graph.execute();
    CHECK(seen == 2);
    p.graph.edit(0, [](cg::NodeEdit &e) { e.add_pre_run([] {}); });
    p.graph.execute();
    CHECK(seen == 3);
}

TEST_CASE("NodeEdit - a captured executor the builder cannot make is kept as the node's own", "[ComputeGraph][NodeEdit][Capture]") {
    // cg::custom records a Custom node with the caller's functor and no codec. Capture marks it as
    // the capture site's own, so an edit keeps it, state and all, rather than refusing the node.
    auto      T    = create_zero_tensor<double>("T", n, n);
    int       seen = 0;
    cg::Graph graph("captured_custom");
    {
        cg::CaptureGuard const guard(graph);
        cg::custom("count", [count = 0, &seen]() mutable { seen = ++count; }, &T);
    }
    std::size_t const at = position_of(graph, cg::OpKind::Custom);
    REQUIRE(graph.nodes()[at].custom_execute != nullptr);
    graph.execute();
    CHECK(seen == 1);
    graph.edit(at, [](cg::NodeEdit &e) { e->label = "renamed"; });
    graph.execute();
    CHECK(seen == 2);
}

TEST_CASE("NodeEdit - a captured executor the builder can make is not marked", "[ComputeGraph][NodeEdit][Capture]") {
    TwoProducts p;
    CHECK(p.graph.nodes()[0].custom_execute == nullptr);
}
