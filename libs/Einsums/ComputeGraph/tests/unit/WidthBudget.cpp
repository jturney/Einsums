//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Admission: a node planned wide is only started once the machine has room for
// it, and the sum of the widths running at any instant never exceeds the
// machine. The interesting content here is the deadlock-freedom argument, which
// is written out as tests rather than as a comment: every width is clamped to
// something the machine can satisfy, a drained budget admits anything, a task
// that fails or that blocks in a nested replay gives its width back, and the
// strict-priority rule never lets a narrow task jump a wide one.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;
using einsums::task_pool::WidthBudget;
using namespace std::chrono_literals;

namespace {

/// The machine the budget rations. Read on the calling thread, which for a
/// Catch2 case is the main thread and so is not one of the pool workers that
/// pin themselves to a single thread.
unsigned machine_threads() {
    int const threads = hardware::get_max_threads();
    return threads > 0 ? static_cast<unsigned>(threads) : 1U;
}

/// Start order and peak concurrency, as seen from inside the kernels.
struct Trace {
    mutable std::mutex       mutex;
    std::vector<std::string> order;
    std::atomic<int>         live{0};
    std::atomic<int>         most_live{0};

    void enter(std::string const &label) {
        {
            std::scoped_lock const lk(mutex);
            order.push_back(label);
        }
        int const now  = live.fetch_add(1) + 1;
        int       seen = most_live.load();
        while (seen < now && !most_live.compare_exchange_weak(seen, now)) {
        }
    }

    void leave() { live.fetch_sub(1); }

    /// Where @p label started, or -1 if it never did.
    [[nodiscard]] long position(std::string const &label) const {
        std::scoped_lock const lk(mutex);
        for (size_t i = 0; i < order.size(); i++) {
            if (order[i] == label) {
                return static_cast<long>(i);
            }
        }
        return -1;
    }

    [[nodiscard]] size_t started() const {
        std::scoped_lock const lk(mutex);
        return order.size();
    }
};

/// A node whose entire effect is to be observed running.
cg::Node work_node(std::string label, unsigned width, Trace *trace, std::chrono::milliseconds work = 0ms) {
    cg::Node node;
    node.kind         = cg::OpKind::Custom;
    node.label        = label;
    node.thread_width = static_cast<std::uint16_t>(width);
    node.execute      = [trace, label, work]() {
        trace->enter(label);
        if (work > 0ms) {
            std::this_thread::sleep_for(work);
        }
        trace->leave();
    };
    return node;
}

/// Position of the node with this label, after the graph has been sorted.
long node_position(cg::Graph &graph, std::string const &label) {
    auto const &nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].label == label) {
            return static_cast<long>(i);
        }
    }
    return -1;
}

} // namespace

TEST_CASE("WidthBudget - an unplanned graph never touches the budget", "[ComputeGraph][Executor][WidthBudget]") {
    // The floor this protects is the per-node submission cost of a graph nobody
    // planned, which is every graph until a planning pass runs. Not one unit is
    // charged, which is the observable half of "no atomic on the hot path".
    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    Trace     trace;
    cg::Graph graph("unplanned");
    for (size_t i = 0; i < 40; i++) {
        graph.add_node(work_node("plain_" + std::to_string(i), 0, &trace));
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(trace.started() == 40);
    REQUIRE(budget.peak_in_use() == 0);
    REQUIRE(budget.parked() == 0);
}

TEST_CASE("WidthBudget - a width beyond the machine is clamped, not stranded", "[ComputeGraph][Executor][WidthBudget]") {
    // A plan is a tuning artifact, so a width the machine cannot satisfy is a
    // stale number rather than a requirement. Clamping is what makes the
    // head-of-line rule deadlock-free: the head of the queue always fits once
    // the budget drains. Refusing, or parking it as-is, would wedge the run -
    // and this test would hang rather than fail.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    Trace     trace;
    cg::Graph graph("impossible_width");
    graph.add_node(work_node("greedy", 4 * p + 7, &trace));

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(trace.position("greedy") == 0);
    REQUIRE(budget.total() == p);
    REQUIRE(budget.peak_in_use() == p);
    REQUIRE(budget.in_use() == 0);
}

TEST_CASE("WidthBudget - a drained budget admits a machine-wide node", "[ComputeGraph][Executor][WidthBudget]") {
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    Trace     trace;
    cg::Graph graph("drain_to_admit");
    // Narrow nodes first so the wide one cannot be admitted on arrival; it runs
    // only because the budget drains completely for it.
    for (size_t i = 0; i < 8; i++) {
        graph.add_node(work_node("narrow_" + std::to_string(i), 1, &trace, 1ms));
    }
    graph.add_node(work_node("machine_wide", p, &trace));

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(trace.started() == 9);
    REQUIRE(trace.position("machine_wide") >= 0);
    REQUIRE(budget.peak_in_use() <= p);
    REQUIRE(budget.in_use() == 0);
}

TEST_CASE("WidthBudget - a parked wide node blocks the head of the line", "[ComputeGraph][Executor][WidthBudget]") {
    // Strict priority: while the most urgent ready task does not fit, nothing
    // narrower is admitted ahead of it, so the machine drains for it. Priority
    // is the longest remaining path to a sink, so the wide node is given a
    // chain of successors and the narrow ones are sinks.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    auto  chained = create_zero_tensor<double>("chained", 4, 4);
    Trace trace;

    cg::Graph graph("head_of_line");
    graph.add_node(work_node("holder", 1, &trace, 40ms));
    {
        // Three scales of one tensor are a three-node chain, which is all this
        // needs from them; the kernels are replaced below.
        cg::CaptureGuard const guard(graph);
        cg::scale(1.0, &chained);
        cg::scale(1.0, &chained);
        cg::scale(1.0, &chained);
    }
    size_t const narrow_count = 6;
    for (size_t i = 0; i < narrow_count; i++) {
        graph.add_node(work_node("narrow_" + std::to_string(i), 1, &trace));
    }

    graph.topological_sort();

    // The scenario needs the holder admitted before the wide node is offered,
    // and the wide node offered before the narrow ones. Roots are seeded in
    // node order, so that is a fact about the sorted graph, and it is asserted
    // rather than assumed: a sort that reordered these would silently turn this
    // into a different test.
    long const holder_at = node_position(graph, "holder");
    long const wide_at   = holder_at + 1;
    REQUIRE(holder_at == 0);
    REQUIRE(graph.nodes()[static_cast<size_t>(wide_at)].kind == cg::OpKind::Scale);
    REQUIRE(node_position(graph, "narrow_0") > wide_at + 2);

    auto &wide_node        = graph.nodes()[static_cast<size_t>(wide_at)];
    wide_node.label        = "wide";
    wide_node.thread_width = static_cast<std::uint16_t>(p);
    wide_node.execute      = [&trace]() {
        trace.enter("wide");
        std::this_thread::sleep_for(5ms);
        trace.leave();
    };

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(trace.position("holder") == 0);
    REQUIRE(trace.position("wide") == 1);
    for (size_t i = 0; i < narrow_count; i++) {
        // The whole point: these fit the whole time the wide node was parked,
        // and none of them was allowed to take the room it was waiting for.
        REQUIRE(trace.position("narrow_" + std::to_string(i)) > trace.position("wide"));
    }
    REQUIRE(budget.peak_in_use() <= p);
    REQUIRE(budget.in_use() == 0);
}

TEST_CASE("WidthBudget - an all-wide graph runs one node at a time", "[ComputeGraph][Executor][WidthBudget]") {
    // Every node claims the whole machine, so the only schedule that respects
    // the budget is a serial one. It must be found without a deadlock and
    // without oversubscribing.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    Trace     trace;
    cg::Graph graph("all_wide");
    for (size_t i = 0; i < 12; i++) {
        graph.add_node(work_node("wide_" + std::to_string(i), p, &trace, 1ms));
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(trace.started() == 12);
    REQUIRE(trace.most_live.load() == 1);
    REQUIRE(budget.peak_in_use() == p);
    REQUIRE(budget.in_use() == 0);
}

TEST_CASE("WidthBudget - a throwing wide kernel returns its width", "[ComputeGraph][Executor][WidthBudget]") {
    // A width leaked by an exception is not a slow run, it is a budget that
    // never drains again: every later replay in the process would park forever.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    cg::Graph failing("wide_throws");
    for (size_t i = 0; i < 8; i++) {
        cg::Node node;
        node.kind         = cg::OpKind::Custom;
        node.label        = "thrower_" + std::to_string(i);
        node.thread_width = static_cast<std::uint16_t>(p);
        node.execute      = []() { throw std::runtime_error("kernel failed while wide"); };
        failing.add_node(std::move(node));
    }

    cg::DataflowExecutor df;
    REQUIRE_THROWS(failing.execute(df));

    // Every node completed (help_until only returns at completed == n, so
    // reaching here at all proves the counting held) and every unit came back.
    REQUIRE(budget.in_use() == 0);
    REQUIRE(budget.parked() == 0);

    // And the budget still works, which is the thing a leak would have broken.
    Trace     trace;
    cg::Graph after("after_throw");
    for (size_t i = 0; i < 4; i++) {
        after.add_node(work_node("wide_" + std::to_string(i), p, &trace));
    }
    after.execute(df);

    REQUIRE(trace.started() == 4);
    REQUIRE(budget.in_use() == 0);
}

TEST_CASE("WidthBudget - a loop body acquires from the same budget", "[ComputeGraph][Executor][WidthBudget]") {
    // A control-flow node replays its body through a nested run whose nodes
    // acquire from this same budget. The container holds one unit, and hands it
    // back for as long as it is waiting on that run - without which a body node
    // planned at the machine width could never be admitted while its own
    // ancestor held a unit of the machine, and the run would wedge.
    unsigned const p          = machine_threads();
    size_t const   iterations = 3;

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    Trace     trace;
    cg::Graph outer("nested_widths");
    outer.add_node(work_node("parent_wide_0", p, &trace, 2ms));
    outer.add_node(work_node("parent_wide_1", p, &trace, 2ms));
    outer.add_node(work_node("parent_narrow", 1, &trace, 2ms));

    auto &body = outer.add_loop("loop", iterations, nullptr);
    body.set_executor(std::make_shared<cg::DataflowExecutor>());
    for (size_t i = 0; i < 4; i++) {
        body.add_node(work_node("body_wide_" + std::to_string(i), p, &trace, 1ms));
    }

    cg::DataflowExecutor df;
    outer.execute(df);

    REQUIRE(trace.started() == 3 + (iterations * 4));
    // The invariant, stated as an invariant rather than hunted for as a race:
    // admission never let more than the machine's worth of width run at once,
    // counting the body's nodes and the parent's together.
    REQUIRE(budget.peak_in_use() <= p);
    REQUIRE(budget.in_use() == 0);
    REQUIRE(budget.parked() == 0);
}

TEST_CASE("WidthBudget - memory parking and width parking compose", "[ComputeGraph][Executor][WidthBudget]") {
    // A Materialize can be parked for bytes and then for width, in that order.
    // The two queues are independent and neither one may strand a node in the
    // other.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    auto first  = std::make_unique<RuntimeTensor<double>>("first", std::vector<size_t>{2, 2});
    auto second = std::make_unique<RuntimeTensor<double>>("second", std::vector<size_t>{2, 2});

    Trace              trace;
    cg::Graph          graph("bytes_then_width");
    cg::TensorId const first_id  = graph.register_tensor(cg::make_handle(*first, 0));
    cg::TensorId const second_id = graph.register_tensor(cg::make_handle(*second, 0));

    {
        cg::Node node;
        node.kind            = cg::OpKind::Materialize;
        node.label           = "materialize_first";
        node.thread_width    = static_cast<std::uint16_t>(p);
        node.estimated_bytes = 100;
        node.outputs         = {first_id};
        node.execute         = [&trace]() {
            trace.enter("materialize_first");
            std::this_thread::sleep_for(2ms);
            trace.leave();
        };
        graph.add_node(std::move(node));
    }
    {
        cg::Node node;
        node.kind            = cg::OpKind::Free;
        node.label           = "free_first";
        node.estimated_bytes = 100;
        node.inputs          = {first_id};
        node.execute         = [&trace]() {
            trace.enter("free_first");
            trace.leave();
        };
        graph.add_node(std::move(node));
    }
    {
        cg::Node node;
        node.kind            = cg::OpKind::Materialize;
        node.label           = "materialize_second";
        node.thread_width    = static_cast<std::uint16_t>(p);
        node.estimated_bytes = 100;
        node.outputs         = {second_id};
        node.execute         = [&trace]() {
            trace.enter("materialize_second");
            trace.leave();
        };
        graph.add_node(std::move(node));
    }

    cg::DataflowExecutor df;
    df.set_memory_budget(100); // room for exactly one of the two materializations
    graph.execute(df);

    REQUIRE(trace.started() == 3);
    // The second materialization waited for the bytes the Free returned, and
    // then for the width, and ran once it had both.
    REQUIRE(trace.position("materialize_second") == 2);
    REQUIRE(budget.peak_in_use() <= p);
    REQUIRE(budget.in_use() == 0);
    REQUIRE(budget.parked() == 0);
}

TEST_CASE("WidthBudget - a plan for another machine is not used", "[ComputeGraph][Executor][WidthBudget]") {
    // Widths divide a KNOWN number of threads between the nodes, so the same
    // widths against a different thread count are not a worse plan, they are a
    // wrong one. The executor runs every node at width 1 instead, which is the
    // one plan that is right on every machine.
    unsigned const p = machine_threads();

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();
    cg::reset_stale_thread_plan_fallbacks();

    // A width no thread in the process already has, so observing it can only
    // mean the plan was honored.
    unsigned const   telltale = p == 4 ? 3 : 4;
    std::atomic<int> observed{-1};

    cg::Graph graph("stale_plan");
    {
        cg::Node node;
        node.kind         = cg::OpKind::Custom;
        node.label        = "planned";
        node.thread_width = static_cast<std::uint16_t>(telltale);
        node.execute      = [&observed]() { observed.store(hardware::get_max_threads(), std::memory_order_relaxed); };
        graph.add_node(std::move(node));
    }

    cg::DataflowExecutor df;

    // The plan claims a machine that is not this one.
    graph.set_planned_thread_count(p + 7);
    graph.execute(df);

    REQUIRE(cg::stale_thread_plan_fallbacks() == 1);
    REQUIRE(budget.peak_in_use() == 0); // not admitted, because not planned
    REQUIRE(observed.load() != static_cast<int>(telltale));

    // The same graph, honestly labelled, is honored.
    REQUIRE(budget.total() == p);
    graph.set_planned_thread_count(p);
    observed.store(-1);
    graph.execute(df);

    REQUIRE(cg::stale_thread_plan_fallbacks() == 1);
    REQUIRE(observed.load() == static_cast<int>(telltale));
    REQUIRE(budget.peak_in_use() == telltale);

    // ... and so is a graph that never recorded a machine at all, which is what
    // a hand-set width looks like.
    graph.set_planned_thread_count(0);
    observed.store(-1);
    graph.execute(df);

    REQUIRE(cg::stale_thread_plan_fallbacks() == 1);
    REQUIRE(observed.load() == static_cast<int>(telltale));
}

TEST_CASE("WidthBudget - concurrent replays share one machine", "[ComputeGraph][Executor][WidthBudget]") {
    // Several graphs replaying at once through separate executors, which is the
    // case a per-scheduler budget would get wrong: each of them would believe
    // it had the whole machine. They contend on one counter and one parked
    // queue, so this is also where a lost wakeup or a mislaid unit shows up as
    // a hang rather than as a wrong answer.
    unsigned const   p       = machine_threads();
    constexpr size_t runners = 4;
    constexpr size_t replays = 20;

    auto &budget = WidthBudget::get_singleton();
    budget.reset_peak();

    std::array<Trace, runners> traces;
    std::vector<std::thread>   threads;
    threads.reserve(runners);

    for (size_t r = 0; r < runners; r++) {
        threads.emplace_back([&, r]() {
            cg::Graph graph("concurrent_" + std::to_string(r));
            for (size_t i = 0; i < 10; i++) {
                // A spread of widths, including the machine-wide one that can
                // only run when every other replay has drained.
                unsigned const width = (i % 3 == 0) ? p : (i % 3 == 1) ? 2U : 1U;
                graph.add_node(work_node("n" + std::to_string(r) + "_" + std::to_string(i), width, &traces[r]));
            }

            cg::DataflowExecutor df;
            for (size_t k = 0; k < replays; k++) {
                graph.execute(df);
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (size_t r = 0; r < runners; r++) {
        REQUIRE(traces[r].started() == 10 * replays);
    }
    REQUIRE(budget.peak_in_use() <= p);
    REQUIRE(budget.in_use() == 0);
    REQUIRE(budget.parked() == 0);
}

TEST_CASE("WidthBudget - mixed widths replay bit-identically", "[ComputeGraph][Executor][WidthBudget]") {
    // A node's arithmetic depends on its inputs and its width, both of which
    // are fixed here, so the order admission happens to choose must not move a
    // bit. This is the tripwire for the whole moldable design: if it fails,
    // widths have made replays nondeterministic.
    unsigned const p = machine_threads();

    constexpr size_t chains = 4;
    constexpr size_t dim    = 24;

    std::vector<Tensor<double, 2>> a, b, c;
    for (size_t i = 0; i < chains; i++) {
        a.push_back(create_random_tensor<double>("a", dim, dim));
        b.push_back(create_random_tensor<double>("b", dim, dim));
        c.push_back(create_zero_tensor<double>("c", dim, dim));
    }

    cg::Graph graph("mixed_widths");
    {
        cg::CaptureGuard const guard(graph);
        for (size_t i = 0; i < chains; i++) {
            cg::einsum("ik;kj->ij", &c[i], a[i], b[i]);
            cg::scale(3.0, &c[i]);
        }
    }

    // Every rung the machine has, so the run parks and drains rather than
    // admitting everything at once.
    unsigned const widths[] = {1U, std::max(2U, p / 2), p, 2U};
    for (size_t i = 0; i < graph.nodes().size(); i++) {
        graph.nodes()[i].thread_width = static_cast<std::uint16_t>(widths[i % 4]);
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    std::vector<Tensor<double, 2>> first;
    for (size_t i = 0; i < chains; i++) {
        first.push_back(Tensor<double, 2>(c[i]));
    }

    graph.execute(df);

    for (size_t i = 0; i < chains; i++) {
        REQUIRE(std::memcmp(c[i].data(), first[i].data(), dim * dim * sizeof(double)) == 0);
    }
}
