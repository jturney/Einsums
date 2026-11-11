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
#include <cstdint>
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

TEST_CASE("NodeEdit - keep_executor keeps a pass-written executor through a rename", "[ComputeGraph][NodeEdit]") {
    // What a pass that renamed an operand behind Graph::redirect_slot says: the executor resolves
    // the old operand's slot, so it already reads what the new lists say. The same executor, with
    // its own state, keeps running.
    TwoProducts p;
    int         runs = 0;
    p.graph.edit(0, [&](cg::NodeEdit &e) { e.set_executor([runs]() mutable { ++runs; }); });
    auto const held = p.graph.nodes()[0].custom_execute;

    p.graph.edit(0, [&](cg::NodeEdit &e) {
        e->inputs[0] = p.a2_id();
        e.keep_executor();
    });
    CHECK(p.graph.nodes()[0].inputs[0] == p.a2_id());
    CHECK(p.graph.nodes()[0].custom_execute == held);

    // On a node the builder makes it changes nothing: the commit rebuilds from the fields.
    p.graph.edit(1, [&](cg::NodeEdit &e) {
        e->inputs[0] = p.graph.find_tensor_id_by_ptr(&p.A);
        e.keep_executor();
    });
    CHECK(p.graph.nodes()[1].custom_execute == nullptr);
    p.graph.execute();
    check_equal(p.D, p.expected(p.A));
}

TEST_CASE("NodeEdit - replace puts a pass-built node in the position, keeping its id", "[ComputeGraph][NodeEdit]") {
    TwoProducts        p;
    cg::NodeId const   id   = p.graph.nodes()[0].id;
    auto const        &spec = p.graph.nodes()[0].op_data.get_if<cg::EinsumDescriptor>()->indices->spec;
    cg::TensorId const c    = p.graph.nodes()[0].outputs[0];
    cg::TensorId const b    = p.graph.nodes()[0].inputs[1];

    // A node the builder makes is rebuilt.
    cg::Node fused = p.graph.make_einsum_node(p.a2_id(), b, c, spec, 0.0, 1.0, false, false, "fused");
    p.graph.edit(0, [&](cg::NodeEdit &e) { e.replace(std::move(fused)); });
    CHECK(p.graph.nodes()[0].id == id);
    CHECK(p.graph.nodes()[0].label == "fused");
    p.graph.execute();
    check_equal(p.C, p.expected(p.A2));

    // One its pass wrote keeps that executor, though the operands are new.
    int      runs = 0;
    cg::Node own;
    own.kind    = cg::OpKind::Custom;
    own.label   = "own";
    own.inputs  = {b};
    own.outputs = {c};
    cg::use_custom_executor(own, [&runs] { ++runs; });
    p.graph.edit(0, [&](cg::NodeEdit &e) { e.replace(std::move(own)); });
    CHECK(p.graph.nodes()[0].id == id);
    p.graph.execute();
    CHECK(runs == 1);
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
    p.graph.edit(0, [](cg::NodeEdit &e) { e.place_on(cg::Target::GPU); });
    REQUIRE(p.graph.nodes()[0].cpu_fallback);
    p.graph.edit(0, [&](cg::NodeEdit &e) { e->inputs[0] = p.a2_id(); });
    p.C.zero();
    p.graph.nodes()[0].cpu_fallback();
    check_equal(p.C, p.expected(p.A2));
}

TEST_CASE("NodeEdit - placing a node back on the CPU drops its fallback", "[ComputeGraph][NodeEdit]") {
    TwoProducts p;
    p.graph.edit(0, [](cg::NodeEdit &e) { e.place_on(cg::Target::GPU); });
    CHECK(p.graph.nodes()[0].target == cg::Target::GPU);
    p.graph.edit(0, [](cg::NodeEdit &e) { e.place_on(cg::Target::CPU); });
    CHECK(p.graph.nodes()[0].target == cg::Target::CPU);
    CHECK_FALSE(p.graph.nodes()[0].cpu_fallback);
}

TEST_CASE("NodeEdit - changing the target directly is refused", "[ComputeGraph][NodeEdit]") {
    // A target set without place_on would leave a GPU node with no CPU fallback to decline to.
    TwoProducts p;
    CHECK_THROWS_AS(p.graph.edit(0, [](cg::NodeEdit &e) { e->target = cg::Target::GPU; }), std::logic_error);
    CHECK(p.graph.nodes()[0].target == cg::Target::CPU);
    CHECK_FALSE(p.graph.nodes()[0].cpu_fallback);
}

TEST_CASE("Graph - set_schedule sets the schedule and nothing else", "[ComputeGraph][NodeEdit][Schedule]") {
    // A planner trying a plan per replay sets every node's width each time; that must not count
    // as an edit, or every replay would rebuild the usage and the dependency schedule.
    TwoProducts         p;
    std::uint64_t const version = p.graph.analysis_version();

    cg::NodeSchedule schedule   = cg::schedule_of(p.graph.nodes()[1]);
    schedule.thread_width       = 3;
    schedule.admission_priority = 42;
    schedule.stream_id          = 1;
    p.graph.set_schedule(1, schedule);

    cg::Node const &node = p.graph.nodes()[1];
    CHECK(node.thread_width == 3);
    CHECK(node.admission_priority == 42);
    CHECK(node.stream_id == 1);
    CHECK(p.graph.analysis_version() == version);
    CHECK_THROWS_AS(p.graph.set_schedule(2, schedule), std::out_of_range);

    p.graph.execute();
    check_equal(p.D, p.expected(p.A2));
}

TEST_CASE("Graph - for_each_child visits one node's bodies", "[ComputeGraph][NodeEdit][Walk]") {
    cg::Graph graph("walk");
    auto [then_g, else_g] = graph.add_conditional("branch", [] { return true; });
    cg::Graph &setup      = graph.add_setup("once");

    std::size_t const branch = position_of(graph, cg::OpKind::Conditional);
    std::size_t const once   = position_of(graph, cg::OpKind::Setup);

    std::vector<cg::Graph const *> seen;
    auto const                     record = [&](cg::Graph &body) { seen.push_back(&body); };

    graph.for_each_child(branch, record);
    CHECK(seen == std::vector<cg::Graph const *>{&then_g, &else_g});

    seen.clear();
    graph.for_each_child(once, record);
    CHECK(seen == std::vector<cg::Graph const *>{&setup});

    seen.clear();
    graph.for_each_child(once, record, /*include_setup=*/false);
    CHECK(seen.empty());

    seen.clear();
    graph.for_each_subgraph(record, /*include_setup=*/false);
    CHECK(seen == std::vector<cg::Graph const *>{&then_g, &else_g});

    CHECK_THROWS_AS(graph.for_each_child(graph.num_nodes(), record), std::out_of_range);
}

TEST_CASE("Graph - for_each_child survives a visitor that grows the parent", "[ComputeGraph][NodeEdit][Walk]") {
    // IOPrefetch inserts into the parent between a conditional's two branches; the node vector
    // may move, and a walk still reading the branch pointers from the descriptor would read
    // freed memory for the second.
    cg::Graph graph("walk_grow");
    auto [then_g, else_g]    = graph.add_conditional("branch", [] { return true; });
    std::size_t const branch = position_of(graph, cg::OpKind::Conditional);

    std::vector<cg::Graph const *> seen;
    graph.for_each_child(branch, [&](cg::Graph &body) {
        seen.push_back(&body);
        for (int k = 0; k < 64; ++k) {
            (void)graph.add_loop("filler", 1, [](std::size_t it) { return it < 1; });
        }
    });
    CHECK(seen == std::vector<cg::Graph const *>{&then_g, &else_g});
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

// ── Node order ──────────────────────────────────────────────────────────────

namespace {

/// Three independent scales, so any order of them is a valid one.
struct ThreeScales {
    Tensor<double, 2> A = create_random_tensor<double>("A", n, n);
    Tensor<double, 2> B = create_random_tensor<double>("B", n, n);
    Tensor<double, 2> C = create_random_tensor<double>("C", n, n);
    cg::Graph         graph{"three_scales"};

    ThreeScales() {
        {
            cg::CaptureGuard const guard(graph);
            cg::scale(2.0, &A);
            cg::scale(3.0, &B);
            cg::scale(4.0, &C);
        }
        for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
            graph.edit(i, [i](cg::NodeEdit &e) { e->label = std::string(1, static_cast<char>('a' + i)); });
        }
    }

    [[nodiscard]] std::string labels() const {
        std::string out;
        for (auto const &node : graph.nodes()) {
            out += node.label;
        }
        return out;
    }
};

} // namespace

TEST_CASE("Graph - reorder puts the nodes in the order given", "[ComputeGraph][NodeEdit][Order]") {
    ThreeScales s;
    REQUIRE(s.labels() == "abc");
    cg::NodeId const               first = s.graph.nodes()[0].id;
    std::vector<std::size_t> const order{2, 0, 1};
    s.graph.reorder(order);
    CHECK(s.labels() == "cab");
    CHECK(s.graph.nodes()[1].id == first); // a node keeps its id when it moves
    s.graph.execute();

    // Position-keyed answers follow the new order.
    cg::TensorId const c = s.graph.nodes()[0].outputs.front();
    CHECK(s.graph.usage().find_owner(c)->first_use() == 0);

    CHECK_THROWS_AS(s.graph.reorder(std::vector<std::size_t>{0, 0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(s.graph.reorder(std::vector<std::size_t>{0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(s.graph.reorder(std::vector<std::size_t>{0, 1, 3}), std::invalid_argument);
    CHECK(s.labels() == "cab");
}

TEST_CASE("Graph - move_node moves one node and keeps the rest in order", "[ComputeGraph][NodeEdit][Order]") {
    ThreeScales s;
    s.graph.move_node(0, 2);
    CHECK(s.labels() == "bca");
    s.graph.move_node(2, 0);
    CHECK(s.labels() == "abc");
    s.graph.move_node(1, 1);
    CHECK(s.labels() == "abc");
    CHECK_THROWS_AS(s.graph.move_node(0, 3), std::out_of_range);
}

// ── Hoisting out of a body ──────────────────────────────────────────────────

TEST_CASE("Graph - hoist moves a body node out, rebuilt against the parent", "[ComputeGraph][NodeEdit][Hoist]") {
    // W = A B is the same at every iteration, so it can run once, before the loop. Its scratch is
    // the body's deferred intermediate, so its declaration moves out with it, and Materialization
    // run afterwards gives it a lifecycle in the parent.
    auto const A   = create_random_tensor<double>("A", n, n);
    auto const B   = create_random_tensor<double>("B", n, n);
    auto       acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph  graph("hoist");
    cg::Graph *body = &graph.add_loop("iter", 2, [](std::size_t it) { return it < 2; });
    void      *w    = nullptr;
    {
        cg::CaptureGuard const guard(*body);
        auto                  &W = body->scratch_zero<double, 2>("W", n, n);
        w                        = &W;
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, B);   // invariant
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A); // acc += W A
    }
    std::size_t const loop = position_of(graph, cg::OpKind::Loop);
    REQUIRE(body->nodes().size() == 2);
    cg::TensorId const w_in_body = body->nodes()[0].outputs.front();

    std::vector<std::size_t> const picks{0};
    auto const                     moved = graph.hoist(*body, picks, loop);
    REQUIRE(moved == std::vector<std::size_t>{loop});
    REQUIRE(body->nodes().size() == 1);

    cg::Node const &lifted = graph.nodes()[loop];
    CHECK(lifted.custom_execute == nullptr);
    CHECK(lifted.id != cg::unassigned_node_id);
    cg::TensorId const w_here = lifted.outputs.front();
    CHECK(w_here == graph.find_tensor_id_by_ptr(w));
    CHECK(lifted.inputs[0] == graph.find_tensor_id_by_ptr(&A));

    // The declaration moved with the writer.
    CHECK(graph.tensor(w_here).is_intermediate);
    CHECK_FALSE(body->tensor(w_in_body).is_intermediate);

    graph.apply<cg::passes::Materialization>();
    graph.execute();

    Tensor<double, 2> AB("AB", n, n);
    Tensor<double, 2> want("want", n, n);
    testing::reference_einsum("ij <- ik ; kj", 0.0, &AB, 1.0, A, B);
    testing::reference_einsum("ij <- ik ; kj", 0.0, &want, 2.0, AB, A); // two iterations
    check_equal(acc, want);
}

TEST_CASE("Graph - a hoisted node its pass wrote keeps its executor and re-seats the body first", "[ComputeGraph][NodeEdit][Hoist]") {
    auto       T    = create_zero_tensor<double>("T", n, n);
    int        runs = 0;
    cg::Graph  graph("hoist_custom");
    cg::Graph *body = &graph.add_loop("iter", 1, [](std::size_t it) { return it < 1; });
    {
        cg::CaptureGuard const guard(*body);
        cg::custom("count", [&runs]() { ++runs; }, &T);
    }
    std::size_t const              loop = position_of(graph, cg::OpKind::Loop);
    std::vector<std::size_t> const picks{0};
    (void)graph.hoist(*body, picks, loop);
    cg::Node const &lifted = graph.nodes()[loop];
    CHECK(lifted.custom_execute != nullptr);
    CHECK(lifted.pre_run.size() == 1);
    graph.execute();
    CHECK(runs == 1);
}

TEST_CASE("Graph - hoist refuses what it cannot do and changes nothing", "[ComputeGraph][NodeEdit][Hoist]") {
    TwoProducts p; // a graph with no body at all
    cg::Graph   other("other");
    auto       &body = other.add_loop("iter", 1, [](std::size_t it) { return it < 1; });
    {
        cg::Node node;
        node.kind    = cg::OpKind::Custom;
        node.label   = "hand-written";
        node.execute = [] {};
        body.add_node(std::move(node));
        cg::Node second;
        second.kind  = cg::OpKind::Custom;
        second.label = "second";
        use_custom_executor(second, [] {});
        body.add_node(std::move(second));
    }
    std::vector<std::size_t> const first{0};
    std::vector<std::size_t> const backwards{1, 0};
    CHECK_THROWS_AS(p.graph.hoist(body, first, 0), std::invalid_argument);   // not p's body
    CHECK_THROWS_AS(other.hoist(body, backwards, 0), std::invalid_argument); // not ascending
    CHECK_THROWS_AS(other.hoist(body, first, 5), std::invalid_argument);     // past the end
    CHECK_THROWS_AS(other.hoist(body, first, 0), std::logic_error);          // no executor to give it
    CHECK(body.nodes().size() == 2);
    CHECK(other.nodes().size() == 1);

    // The one its pass marked moves.
    std::vector<std::size_t> const second{1};
    CHECK(other.hoist(body, second, 0) == std::vector<std::size_t>{0});
    CHECK(body.nodes().size() == 1);
    CHECK(other.nodes().size() == 2);
}
