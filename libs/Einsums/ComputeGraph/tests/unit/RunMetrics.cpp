//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// Graph::last_run: what a replay measured as it ran. The memory and node counts are asserted
/// exactly, on graphs built so the answer is known.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t n     = 16;
constexpr std::size_t bytes = n * n * sizeof(double);

RuntimeTensor<double> filled(std::string name, double scale) {
    RuntimeTensor<double> out(std::move(name), std::vector<std::size_t>{n, n});
    for (std::size_t k = 0; k < out.size(); ++k) {
        out.data()[k] = scale * static_cast<double>(k % 7);
    }
    return out;
}

/// out = ((A B) B) A through three graph-owned intermediates X, Y, Z, each dead once the next is
/// computed: under FreeInsertion at most two are held at once.
struct Chain {
    RuntimeTensor<double> a{filled("A", 0.25)};
    RuntimeTensor<double> b{filled("B", 0.5)};
    RuntimeTensor<double> out{"out", std::vector<std::size_t>{n, n}};
    cg::Graph             graph{"chain"};

    /// @p eager: the intermediates are allocated when declared rather than by a Materialize node,
    /// so FreeInsertion frees tensors no lifecycle node allocated.
    explicit Chain(bool eager = false) {
        auto const make = [&](std::string const &name) -> RuntimeTensor<double> & {
            return eager ? graph.create_runtime_tensor<double>(name, {n, n}, true)
                         : graph.declare_runtime_tensor<double>(name, {n, n}, true);
        };
        auto                  &x = make("X");
        auto                  &y = make("Y");
        auto                  &z = make("Z");
        cg::CaptureGuard const guard(graph);
        cg::einsum("i,j <- i,k ; k,j", &x, a, b);
        cg::einsum("i,j <- i,k ; k,j", &y, x, b);
        cg::einsum("i,j <- i,k ; k,j", &z, y, a);
        cg::einsum("i,j <- i,k ; k,j", &out, z, a);
    }

    std::shared_ptr<cg::passes::MemoryPlanning> apply(bool arena) {
        auto            memory = std::make_shared<cg::passes::MemoryPlanning>(arena);
        cg::PassManager manager;
        manager.add(std::make_shared<cg::passes::Materialization>());
        manager.add(std::make_shared<cg::passes::FreeInsertion>(0)); // these tensors are under its default floor
        manager.add(memory);
        graph.apply(manager);
        return memory;
    }
};

std::size_t total_runs(cg::RunMetrics const &run, cg::OpKind kind) {
    std::size_t runs = 0;
    for (auto const &node : run.nodes) {
        if (node.kind == kind) {
            runs += node.runs;
        }
    }
    return runs;
}

} // namespace

TEST_CASE("last_run is empty before the first replay", "[ComputeGraph][RunMetrics]") {
    Chain       chain;
    auto const &run = chain.graph.last_run();
    CHECK(run.nodes.empty());
    CHECK(run.backings.empty());
    CHECK(run.peak_bytes == 0);
}

TEST_CASE("last_run counts the most graph-managed memory held at once", "[ComputeGraph][RunMetrics]") {
    Chain chain;
    SECTION("tensors freed after their last use: two of the three at once, at every replay") {
        chain.apply(false);
        for (int replay = 0; replay < 2; ++replay) {
            chain.graph.execute();
            auto const &run = chain.graph.last_run();
            CHECK(run.peak_bytes == 2 * bytes);
            CHECK(run.arena_bytes == 0);
            CHECK(total_runs(run, cg::OpKind::Einsum) == 4);
            CHECK(total_runs(run, cg::OpKind::Materialize) == 3);
            CHECK(total_runs(run, cg::OpKind::Free) == 3);
        }
    }
    SECTION("tensors placed in an arena: the arena, held for the whole replay") {
        auto const memory = chain.apply(true);
        REQUIRE(memory->num_planned() > 0);
        chain.graph.execute();
        auto const &run = chain.graph.last_run();
        CHECK(run.arena_bytes == memory->planned_arena_bytes());
        CHECK(run.peak_bytes == run.arena_bytes);
    }
}

// A tensor allocated when it was declared, then freed by a Free node, was released without ever
// being counted as allocated: the live count went below zero and the reported peak wrapped to
// about 1.8e19 bytes (seen on the CCSD example with every intermediate freeable).
TEST_CASE("last_run counts tensors allocated outside a lifecycle node", "[ComputeGraph][RunMetrics]") {
    Chain chain(/*eager=*/true);
    chain.apply(false);
    for (int replay = 0; replay < 3; ++replay) {
        INFO("replay " << replay);
        chain.graph.execute();
        auto const &run = chain.graph.last_run();
        CHECK(total_runs(run, cg::OpKind::Free) == 3);
        CHECK(run.peak_bytes >= 2 * bytes);
        CHECK(run.peak_bytes <= 3 * bytes);
    }
}

TEST_CASE("last_run reports a loop body's nodes summed over its iterations", "[ComputeGraph][RunMetrics]") {
    auto      a = filled("A", 0.25);
    auto      c = filled("C", 1.0);
    cg::Graph graph("looped");
    auto     &body = graph.add_loop("triple", 3, [](std::size_t iteration) { return iteration < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &c);
    }
    graph.execute();
    auto const &run  = graph.last_run();
    auto const  loop = std::ranges::find_if(run.nodes, [](cg::NodeRun const &node) { return node.kind == cg::OpKind::Loop; });
    REQUIRE(loop != run.nodes.end());
    CHECK(loop->graph == "looped");
    CHECK(loop->runs == 1);
    auto const scale = std::ranges::find_if(run.nodes, [](cg::NodeRun const &node) { return node.graph == "triple/body"; });
    REQUIRE(scale != run.nodes.end());
    CHECK(scale->runs == 3);
    CHECK(scale->ms <= loop->ms);
    CHECK(run.wall_ms >= loop->ms);
}

TEST_CASE("last_run is the same under every executor", "[ComputeGraph][RunMetrics]") {
    Chain chain;
    chain.apply(false);
    std::vector<std::shared_ptr<cg::Executor>> executors{std::make_shared<cg::SequentialExecutor>(), std::make_shared<cg::OpenMPExecutor>(),
                                                         std::make_shared<cg::DataflowExecutor>()};
    for (auto const &executor : executors) {
        INFO("executor " << executor->name());
        chain.graph.execute(*executor);
        auto const &run = chain.graph.last_run();
        CHECK(total_runs(run, cg::OpKind::Einsum) == 4);
        CHECK(total_runs(run, cg::OpKind::Materialize) == 3);
        CHECK(total_runs(run, cg::OpKind::Free) == 3);
        // Two at once in list order; a parallel executor may also hold all three, never more.
        CHECK(run.peak_bytes >= 2 * bytes);
        CHECK(run.peak_bytes <= 3 * bytes);
    }
}

TEST_CASE("A replay that throws leaves the record of the one before", "[ComputeGraph][RunMetrics]") {
    Chain chain;
    chain.apply(false);
    bool     fail = false;
    cg::Node node;
    node.kind    = cg::OpKind::Custom;
    node.label   = "fails on demand";
    node.execute = [&fail]() {
        if (fail) {
            throw std::runtime_error("asked to fail");
        }
    };
    chain.graph.add_node(std::move(node));
    chain.graph.execute();
    auto const wall  = chain.graph.last_run().wall_ms;
    auto const nodes = chain.graph.last_run().nodes.size();

    fail = true;
    CHECK_THROWS(chain.graph.execute());
    CHECK(chain.graph.last_run().wall_ms == wall);
    CHECK(chain.graph.last_run().nodes.size() == nodes);

    // And the tree is no longer marked inside a replay: the next one is recorded.
    fail = false;
    chain.graph.execute();
    CHECK(chain.graph.last_run().nodes.size() == nodes);
}
