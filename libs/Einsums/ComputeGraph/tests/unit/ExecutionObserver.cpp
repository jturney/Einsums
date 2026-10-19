//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// Graph::set_observer: what an observer hears of a replay, under every executor.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t n = 16;

RuntimeTensor<double> filled(std::string name, double scale) {
    RuntimeTensor<double> out(std::move(name), std::vector<std::size_t>{n, n});
    for (std::size_t k = 0; k < out.size(); ++k) {
        out.data()[k] = scale * static_cast<double>(k % 7);
    }
    return out;
}

/// out = ((A B) B) A through three graph-owned intermediates, freed after their last use.
struct Chain {
    RuntimeTensor<double> a{filled("A", 0.25)};
    RuntimeTensor<double> b{filled("B", 0.5)};
    RuntimeTensor<double> out{"out", std::vector<std::size_t>{n, n}};
    cg::Graph             graph{"chain"};

    Chain() {
        auto &x = graph.declare_runtime_tensor<double>("X", {n, n}, true);
        auto &y = graph.declare_runtime_tensor<double>("Y", {n, n}, true);
        auto &z = graph.declare_runtime_tensor<double>("Z", {n, n}, true);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,k ; k,j", &x, a, b);
            cg::einsum("i,j <- i,k ; k,j", &y, x, b);
            cg::einsum("i,j <- i,k ; k,j", &z, y, a);
            cg::einsum("i,j <- i,k ; k,j", &out, z, a);
        }
        cg::PassManager manager;
        manager.add(std::make_shared<cg::passes::Materialization>());
        manager.add(std::make_shared<cg::passes::FreeInsertion>(0));
        graph.apply(manager);
    }
};

/// Counts what it hears and checks every node end has its start, on the same thread.
class Counting : public cg::ExecutionObserver {
  public:
    void on_replay_start(cg::ExecutionEvent const &) override {
        std::scoped_lock const lock(mutex);
        ++replay_starts;
    }
    void on_node_start(cg::ExecutionEvent const &event) override {
        std::scoped_lock const lock(mutex);
        ++starts;
        open[{event.graph, event.node}] = event.thread;
    }
    void on_node_end(cg::ExecutionEvent const &event) override {
        std::scoped_lock const lock(mutex);
        ++ends;
        auto const hit = open.find({event.graph, event.node});
        if (hit == open.end() || hit->second != event.thread || event.ms < 0.0) {
            ++unpaired;
        } else {
            open.erase(hit);
        }
        graphs.push_back(event.graph);
    }
    void on_replay_end(cg::ExecutionEvent const &, cg::RunMetrics const &run) override {
        std::scoped_lock const lock(mutex);
        ++replay_ends;
        nodes_in_run = run.nodes.size();
    }

    std::mutex                                                  mutex;
    std::size_t                                                 replay_starts{0}, replay_ends{0}, starts{0}, ends{0}, unpaired{0};
    std::size_t                                                 nodes_in_run{0};
    std::map<std::pair<std::string, cg::NodeId>, std::uint64_t> open;
    std::vector<std::string>                                    graphs;
};

class Throwing : public cg::ExecutionObserver {
  public:
    void on_node_start(cg::ExecutionEvent const &) override {
        ++calls;
        throw std::runtime_error("an observer that fails");
    }
    std::size_t calls{0};
};

} // namespace

TEST_CASE("An observer hears every node start and end, paired, under every executor", "[ComputeGraph][Observer]") {
    Chain                                      chain;
    std::vector<std::shared_ptr<cg::Executor>> executors{std::make_shared<cg::SequentialExecutor>(), std::make_shared<cg::OpenMPExecutor>(),
                                                         std::make_shared<cg::DataflowExecutor>()};
    SECTION("the built-in loop") {
        auto counting = std::make_shared<Counting>();
        chain.graph.set_observer(counting);
        chain.graph.execute();
        CHECK(counting->replay_starts == 1);
        CHECK(counting->replay_ends == 1);
        CHECK(counting->starts == chain.graph.num_nodes());
        CHECK(counting->ends == counting->starts);
        CHECK(counting->unpaired == 0);
        CHECK(counting->open.empty());
        CHECK(counting->nodes_in_run == chain.graph.last_run().nodes.size());
    }
    for (auto const &executor : executors) {
        DYNAMIC_SECTION("executor " << executor->name()) {
            auto counting = std::make_shared<Counting>();
            chain.graph.set_observer(counting);
            chain.graph.execute(*executor);
            CHECK(counting->replay_starts == 1);
            CHECK(counting->replay_ends == 1);
            CHECK(counting->starts == chain.graph.num_nodes());
            CHECK(counting->ends == counting->starts);
            CHECK(counting->unpaired == 0);
        }
    }
}

TEST_CASE("Without an observer nothing is heard, and detaching stops it", "[ComputeGraph][Observer]") {
    Chain chain;
    auto  counting = std::make_shared<Counting>();
    chain.graph.execute();
    chain.graph.set_observer(counting);
    chain.graph.set_observer(nullptr);
    chain.graph.execute();
    CHECK(counting->starts == 0);
    CHECK(counting->replay_starts == 0);
}

TEST_CASE("An observer that throws is detached and the replay is unaffected", "[ComputeGraph][Observer]") {
    Chain reference;
    reference.graph.execute();

    Chain chain;
    auto  throwing = std::make_shared<Throwing>();
    chain.graph.set_observer(throwing);
    CHECK_NOTHROW(chain.graph.execute());
    CHECK(throwing->calls == 1);
    CHECK(std::equal(chain.out.data(), chain.out.data() + chain.out.size(), reference.out.data()));
    CHECK_NOTHROW(chain.graph.execute());
    CHECK(throwing->calls == 1);
}

TEST_CASE("An observer hears a loop body's nodes under the body's name", "[ComputeGraph][Observer]") {
    auto      c = filled("C", 1.0);
    cg::Graph graph("looped");
    auto     &body = graph.add_loop("triple", 3, [](std::size_t iteration) { return iteration < 2; });
    {
        cg::CaptureGuard const guard(body);
        cg::scale(0.5, &c);
    }
    auto counting = std::make_shared<Counting>();
    graph.set_observer(counting);
    graph.execute();
    CHECK(counting->replay_starts == 1);
    CHECK(std::ranges::count(counting->graphs, std::string{"triple/body"}) == 3);
    CHECK(counting->unpaired == 0);
}
