//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkMoldableWidths.cpp
/// @brief The case moldable widths exist for: one level holding two fat nodes
///        and many thin ones, where a node-per-thread schedule leaves most of
///        the machine idle because most of the nodes are trivial.
///
/// The plan is asserted; the times are reported. A width assertion is a
/// statement about the planner and holds on any machine with threads to divide,
/// while a timing assertion on a shared machine tests the machine.
///
/// Both levels here are built from kernels einsums threads itself - an
/// element-wise transform and a rank-3 permute - which fork from the OpenMP ICV
/// the executor sets and are therefore moldable whatever BLAS was linked. A
/// contraction would be the more natural fat node and is exactly the wrong one
/// for this benchmark: on a vendor without per-thread control the planner
/// correctly refuses to widen a BLAS route, and the case would assert nothing.
///
/// The two levels differ in what limits them, which is the whole difference
/// between a width that buys time and one that only buys threads. The transform
/// level is compute-bound and scales with width; the permute level saturates
/// memory bandwidth at two threads and cannot.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::performance;
namespace cg = einsums::compute_graph;

namespace {

constexpr int kFatNodes  = 2;
constexpr int kThinNodes = 12;
constexpr int kReps      = 5;

/// Bytes above which a node counts as fat here. Both levels put three orders of
/// magnitude between their fat and thin working sets, so this never has to be
/// retuned with the extents.
constexpr size_t kFatBytes = 4u << 20;

/// The machine the width budget rations, which is the `P` a plan is made for.
unsigned machine_threads() {
    auto &budget = task_pool::WidthBudget::get_singleton();
    budget.sync_machine_width();
    return std::max(1U, budget.total());
}

/// Bytes the graph believes a node touches, which is what the planner prices it
/// by, and so the honest way to tell a fat node from a thin one without leaning
/// on capture order surviving the topological sort.
size_t node_bytes(cg::Graph const &graph, cg::Node const &node) {
    auto const &map   = graph.tensors_map();
    size_t      bytes = 0;
    auto        add   = [&](cg::TensorId raw) {
        if (auto it = map.find(graph.resolve_alias(raw)); it != map.end()) {
            bytes += it->second.total_bytes();
        }
    };
    for (cg::TensorId const tid : node.inputs) {
        add(tid);
    }
    for (cg::TensorId const tid : node.outputs) {
        add(tid);
    }
    return bytes;
}

/// Plan @p graph for @p p threads, assert the shape of the plan, print it, and
/// hand back the widths so a caller can put them back after running without
/// them.
///
/// Run as a standalone pass rather than through `plan_threads()` so the
/// estimates are readable and no warmup re-plan can move the widths out from
/// under the assertions.
std::vector<std::uint16_t> plan_and_assert(cg::Graph &graph, unsigned p, char const *what) {
    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    std::vector<std::uint16_t> planned;
    std::vector<unsigned>      fat_widths, thin_widths;
    unsigned                   fat_width_sum = 0;
    for (auto const &node : graph.nodes()) {
        planned.push_back(node.thread_width);
        if (node_bytes(graph, node) >= kFatBytes) {
            fat_widths.push_back(node.thread_width);
            fat_width_sum += node.thread_width;
        } else {
            thin_widths.push_back(node.thread_width);
        }
    }

    fmt::println("[MoldableWidths {}] P={}  {} fat + {} thin  moldable BLAS route: {}", what, p, kFatNodes, kThinNodes,
                 cg::blas_route_is_moldable());
    fmt::println("  plan: fat widths [{}] (sum {}), thin widths [{}]", fmt::join(fat_widths, ","), fat_width_sum,
                 fmt::join(thin_widths, ","));
    fmt::println("  estimate: makespan {:.2f} ms -> {:.2f} ms (cp {:.2f} ms, area {:.2f} ms)", planner.makespan_before_us() * 1e-3,
                 planner.makespan_after_us() * 1e-3, planner.critical_path_after_us() * 1e-3, planner.area_after_us() * 1e-3);

    REQUIRE(fat_widths.size() == static_cast<size_t>(kFatNodes));
    REQUIRE(thin_widths.size() == static_cast<size_t>(kThinNodes));
    REQUIRE(planner.num_widened() == static_cast<std::size_t>(kFatNodes));
    REQUIRE(planner.makespan_after_us() < planner.makespan_before_us());

    for (auto const &node : graph.nodes()) {
        // Every planned node carries an admission priority; a zero would mean
        // the executor is ordering this level by hop count instead of the plan.
        REQUIRE(node.admission_priority > 0);
    }
    for (unsigned const width : fat_widths) {
        REQUIRE(width > 1);
        REQUIRE(width <= p);
    }
    for (unsigned const width : thin_widths) {
        // Under the serial floor, so narrow whatever the machine has spare.
        REQUIRE(width == 1);
    }

    // The machine bound the planner enforces is the AREA, not the sum of the
    // widths on a level: `sum(w_i * t_i(w_i)) / P` inside the makespan. The sum
    // is deliberately not asserted, because a plan is allowed to hand out more
    // width than the machine has at one instant - the process-wide WidthBudget
    // rations that at run time, admitting the wide nodes in priority order
    // rather than all at once. It is printed above so an oversubscribed level
    // is visible when reading a run.
    double const tolerance = 1e-6 * std::max(1.0, planner.makespan_after_us());
    REQUIRE(planner.area_after_us() <= planner.makespan_after_us() + tolerance);

    return planned;
}

/// Median wall time of @p reps replays in milliseconds, after one untimed
/// warmup. The median rather than the mean because a benchmark machine is
/// shared and one preempted replay should not move the number it reports.
double median_replay_ms(cg::Graph &graph, cg::Executor &executor, int reps) {
    graph.execute(executor);

    std::vector<double> times;
    times.reserve(static_cast<size_t>(reps));
    for (int r = 0; r < reps; r++) {
        auto const t0 = Clock::now();
        graph.execute(executor);
        auto const t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

/// Check and then time the same captured graph three ways: the plan under the
/// dataflow executor, the same graph with every width forced to 1 (what the
/// dataflow executor did before widths existed), and the OpenMP executor, which
/// ignores widths entirely.
///
/// @p reset restores the inputs, and @p deviation reports the worst
/// disagreement with the sequential reference. Correctness first and once per
/// executor: a replay that races is not a faster replay.
void run_three_ways(cg::Graph &graph, std::vector<std::uint16_t> const &planned, char const *what, std::function<void()> const &reset,
                    std::function<double()> const &deviation) {
    cg::DataflowExecutor df;
    cg::OpenMPExecutor   omp;

    reset();
    graph.execute(df);
    REQUIRE(deviation() < 1e-12);
    double const t_planned = median_replay_ms(graph, df, kReps);

    for (auto &node : graph.nodes()) {
        node.thread_width = 1;
    }
    reset();
    graph.execute(df);
    REQUIRE(deviation() < 1e-12);
    double const t_narrow = median_replay_ms(graph, df, kReps);

    reset();
    graph.execute(omp);
    REQUIRE(deviation() < 1e-12);
    double const t_omp = median_replay_ms(graph, omp, kReps);

    for (size_t i = 0; i < graph.nodes().size(); i++) {
        graph.nodes()[i].thread_width = planned[i];
    }

    fmt::println("  median of {} replays: dataflow+plan {:.2f} ms   dataflow width-1 {:.2f} ms   openmp {:.2f} ms", kReps, t_planned,
                 t_narrow, t_omp);
    fmt::println("  planned speedup: {:.2f}x over width-1 dataflow, {:.2f}x over openmp", t_narrow / t_planned, t_omp / t_planned);

    ProfileAnnotate("topology", what);
    ProfileAnnotate("nodes", int64_t(kFatNodes + kThinNodes));
}

/// A transcendental with a bounded orbit: x in (0, 1] maps into (0.36, 1], so
/// replaying the same graph over its own output neither overflows nor drifts
/// into denormals, and every replay costs the same. Expensive enough per
/// element that the kernel is compute-bound rather than bandwidth-bound, which
/// is the regime a width can actually buy time in.
double bounded_transcendental(double x) {
    return std::exp(-x * x);
}

} // namespace

EINSUMS_TEST_CASE("Bench Moldable: two fat plus many thin, compute-bound", "[ComputeGraph][ThreadPlanning][benchmark]") {
    LabeledSection0();

    unsigned const p = machine_threads();
    if (p < 4) {
        // Below four threads two fat nodes already saturate the machine and the
        // plan this case asserts is the wrong one.
        SKIP("plan needs at least four threads to have anything to divide");
    }

    // 200^3 doubles is 64 MB, far above the planner's serial floor at any
    // bandwidth; 32^3 is a quarter of a megabyte, which is tens of microseconds
    // and therefore under it - a fork costs more than the node contains.
    constexpr size_t kFatExtent  = 200;
    constexpr size_t kThinExtent = 32;

    // Storage first and whole: capture takes the addresses of these tensors, so
    // nothing may reallocate afterwards.
    std::vector<Tensor<double, 3>> fat, thin;
    fat.reserve(kFatNodes);
    thin.reserve(kThinNodes);
    for (int i = 0; i < kFatNodes; i++) {
        fat.emplace_back(create_random_tensor<double>(fmt::format("fat_{}", i), kFatExtent, kFatExtent, kFatExtent));
    }
    for (int i = 0; i < kThinNodes; i++) {
        thin.emplace_back(create_random_tensor<double>(fmt::format("thin_{}", i), kThinExtent, kThinExtent, kThinExtent));
    }

    // One dependency level: every node transforms its own buffer in place, so
    // the executors differ only in how they spend the machine on that level.
    cg::Graph graph("two_fat_many_thin_transform");
    {
        cg::CaptureGuard const guard(graph);
        for (auto &tensor : fat) {
            cg::element_transform(&tensor, bounded_transcendental);
        }
        for (auto &tensor : thin) {
            cg::element_transform(&tensor, bounded_transcendental);
        }
    }

    auto flatten = [&]() {
        std::vector<double> out;
        for (auto const &t : fat) {
            out.insert(out.end(), t.data(), t.data() + t.size());
        }
        for (auto const &t : thin) {
            out.insert(out.end(), t.data(), t.data() + t.size());
        }
        return out;
    };

    // In place, so the inputs have to be put back before every checked replay.
    std::vector<double> const inputs = flatten();
    auto                      reset  = [&]() {
        double const *src = inputs.data();
        for (auto &t : fat) {
            std::copy(src, src + t.size(), t.data());
            src += t.size();
        }
        for (auto &t : thin) {
            std::copy(src, src + t.size(), t.data());
            src += t.size();
        }
    };

    // The reference every executor is checked against, taken before any width
    // exists so it cannot have been produced by the thing under test.
    cg::SequentialExecutor seq;
    graph.execute(seq);
    std::vector<double> const reference = flatten();

    auto deviation = [&]() {
        std::vector<double> const current = flatten();
        double                    worst   = 0.0;
        for (size_t i = 0; i < reference.size(); i++) {
            worst = std::max(worst, std::abs(current[i] - reference[i]));
        }
        return worst;
    };

    auto const planned = plan_and_assert(graph, p, "transform");
    run_three_ways(graph, planned, "two_fat_many_thin_transform", reset, deviation);
}

EINSUMS_TEST_CASE("Bench Moldable: two fat plus many thin, bandwidth-bound", "[ComputeGraph][ThreadPlanning][benchmark]") {
    LabeledSection0();

    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    // The same level shape over a kernel that is pure traffic. The planner
    // widens the fat nodes here too - correctly, by its own model - and the
    // measurement is what says how much that is worth: a rank-3 permute of
    // 128 MB saturates the memory system long before it saturates the cores,
    // so the extra threads move the same bytes.
    constexpr size_t kFatExtent  = 200;
    constexpr size_t kThinExtent = 32;

    std::vector<Tensor<double, 3>> fat_in, fat_out, thin_in, thin_out;
    fat_in.reserve(kFatNodes);
    fat_out.reserve(kFatNodes);
    thin_in.reserve(kThinNodes);
    thin_out.reserve(kThinNodes);
    for (int i = 0; i < kFatNodes; i++) {
        fat_in.emplace_back(create_random_tensor<double>(fmt::format("fat_in_{}", i), kFatExtent, kFatExtent, kFatExtent));
        fat_out.emplace_back(create_zero_tensor<double>(fmt::format("fat_out_{}", i), kFatExtent, kFatExtent, kFatExtent));
    }
    for (int i = 0; i < kThinNodes; i++) {
        thin_in.emplace_back(create_random_tensor<double>(fmt::format("thin_in_{}", i), kThinExtent, kThinExtent, kThinExtent));
        thin_out.emplace_back(create_zero_tensor<double>(fmt::format("thin_out_{}", i), kThinExtent, kThinExtent, kThinExtent));
    }

    cg::Graph graph("two_fat_many_thin_permute");
    {
        cg::CaptureGuard const guard(graph);
        for (int i = 0; i < kFatNodes; i++) {
            cg::permute("kij <- ijk", 0.0, &fat_out[i], 1.0, fat_in[i]);
        }
        for (int i = 0; i < kThinNodes; i++) {
            cg::permute("kij <- ijk", 0.0, &thin_out[i], 1.0, thin_in[i]);
        }
    }

    auto reset = [&]() {
        for (auto &t : fat_out) {
            t.zero();
        }
        for (auto &t : thin_out) {
            t.zero();
        }
    };

    auto flatten = [&]() {
        std::vector<double> out;
        for (auto const &t : fat_out) {
            out.insert(out.end(), t.data(), t.data() + t.size());
        }
        for (auto const &t : thin_out) {
            out.insert(out.end(), t.data(), t.data() + t.size());
        }
        return out;
    };

    cg::SequentialExecutor seq;
    reset();
    graph.execute(seq);
    std::vector<double> const reference = flatten();

    auto deviation = [&]() {
        std::vector<double> const current = flatten();
        double                    worst   = 0.0;
        for (size_t i = 0; i < reference.size(); i++) {
            worst = std::max(worst, std::abs(current[i] - reference[i]));
        }
        return worst;
    };

    auto const planned = plan_and_assert(graph, p, "permute");
    run_three_ways(graph, planned, "two_fat_many_thin_permute", reset, deviation);
}
