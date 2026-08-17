//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The width PLAN, not the wall clock. Every case here asserts what the planner
// decided - which nodes were widened, which were vetoed, what the area came to -
// because a timing assertion on a shared machine tests the machine. The two
// cases that do replay assert the answer rather than speed, which is the one
// run-time property the plan is supposed to guarantee: bit-identity where the
// widths are the same on both sides, and a last-place bound where they are not.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The machine the budget rations, which is the `P` a plan is made for.
unsigned machine_threads() {
    auto &budget = task_pool::WidthBudget::get_singleton();
    budget.sync_machine_width();
    return std::max(1U, budget.total());
}

/// A square deferred tensor of @p extent, and its id. Deferred means metadata
/// and no storage, which is exactly what a planner reads and is what lets these
/// cases declare working sets of hundreds of megabytes for free. Nothing here
/// ever materializes or executes them.
cg::TensorId shell(cg::Graph &graph, std::string const &name, size_t extent) {
    auto &tensor = graph.declare_runtime_tensor<double>(name, {extent, extent});
    return graph.find_tensor_id_by_ptr(&tensor);
}

/// A node that reads one buffer and writes another, and does nothing.
cg::Node work_node(std::string label, cg::TensorId in, cg::TensorId out, cg::OpKind kind = cg::OpKind::Custom) {
    cg::Node node;
    node.kind    = kind;
    node.label   = std::move(label);
    node.inputs  = {in};
    node.outputs = {out};
    node.execute = []() {};
    return node;
}

/// Extent whose square, at 8 bytes an element and two buffers a node, is far
/// past any plausible serial floor: 2 x 4096^2 x 8 B is 256 MB, which is
/// milliseconds at any memory bandwidth a CPU has.
constexpr size_t kFatExtent = 4096;

/// ...and one that is L1-resident, so it is under the floor at any bandwidth
/// AND has an efficiency curve of exactly 1/w, which is the second reason it
/// can never be widened.
constexpr size_t kThinExtent = 24;

/// The width of node @p position, as the plan left it.
unsigned width_of(cg::Graph const &graph, size_t position) {
    return graph.nodes()[position].thread_width;
}

/// A byte-exact fingerprint of a result buffer. Two replays of ONE plan on one
/// machine run the same kernels at the same widths, so their bits match; that
/// is what the determinism cases assert.
std::vector<unsigned char> bytes_of(Tensor<double, 2> const &tensor) {
    std::vector<unsigned char> out(tensor.size() * sizeof(double));
    std::memcpy(out.data(), tensor.data(), out.size());
    return out;
}

/// The values, for a comparison that spans two different widths.
std::vector<double> values_of(Tensor<double, 2> const &tensor) {
    return {tensor.data(), tensor.data() + tensor.size()};
}

/// The worst elementwise difference between two replays, relative to the
/// element but never divided by less than one.
///
/// Widths are not a rounding-neutral choice: VendorWidthFence presents a width
/// of one to the vendor from inside a moldable scope, so a node that is planned
/// wide reaches an OpenMP-threaded OpenBLAS single-threaded while the same node
/// unplanned reaches it at the machine width. Those two accumulate a dot
/// product in different orders. The difference that leaves is a last-digit one;
/// the corruption these suites exist to catch is not, which is why a bound
/// still separates them.
///
/// The floor of one is the part worth explaining. These buffers are GEMMs of
/// random data, so among a hundred thousand outputs some land near zero through
/// cancellation, and dividing a last-bit difference by such an element reports a
/// large relative error for an answer that is not wrong. Every comparison in
/// this suite family measures the same way for the same reason - see `compare`
/// in MixedWidthDifferential and MixedWidthVendorSafety.
double worst_difference(std::vector<double> const &lhs, std::vector<double> const &rhs) {
    REQUIRE(lhs.size() == rhs.size());
    double worst = 0.0;
    for (size_t i = 0; i < lhs.size(); i++) {
        worst = std::max(worst, std::abs(lhs[i] - rhs[i]) / std::max(1.0, std::abs(lhs[i])));
    }
    return worst;
}

/// Two fat contractions plus enough thin ones to keep the graph honest, with
/// real storage so it can be replayed.
struct FatGraph {
    Tensor<double, 2> a{create_random_tensor<double>("a", 384, 384)};
    Tensor<double, 2> b{create_random_tensor<double>("b", 384, 384)};
    Tensor<double, 2> c1{create_zero_tensor<double>("c1", 384, 384)};
    Tensor<double, 2> c2{create_zero_tensor<double>("c2", 384, 384)};
    Tensor<double, 2> small{create_random_tensor<double>("small", 16, 16)};
    Tensor<double, 2> s1{create_zero_tensor<double>("s1", 16, 16)};
    Tensor<double, 2> s2{create_zero_tensor<double>("s2", 16, 16)};

    void capture(cg::Graph &graph) {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &c1, a, b);
        cg::einsum("ik;kj->ij", &c2, b, a);
        cg::einsum("ik;kj->ij", &s1, small, small);
        cg::einsum("ik;kj->ij", &s2, small, small);
    }
};

} // namespace

TEST_CASE("ThreadPlanning - fat nodes are widened and thin ones are not", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        // Below four threads two fat nodes already saturate the machine, and
        // the correct plan is the one this case is asserting is wrong.
        SKIP("plan needs at least four threads to have anything to divide");
    }

    cg::Graph graph("two_fat_many_thin");
    for (int i = 0; i < 2; i++) {
        auto const in  = shell(graph, fmt::format("fat_in_{}", i), kFatExtent);
        auto const out = shell(graph, fmt::format("fat_out_{}", i), kFatExtent);
        graph.add_node(work_node(fmt::format("fat_{}", i), in, out));
    }
    for (int i = 0; i < 8; i++) {
        auto const in  = shell(graph, fmt::format("thin_in_{}", i), kThinExtent);
        auto const out = shell(graph, fmt::format("thin_out_{}", i), kThinExtent);
        graph.add_node(work_node(fmt::format("thin_{}", i), in, out));
    }

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    REQUIRE(planner.num_widened() == 2);
    REQUIRE(planner.max_width() > 1);
    REQUIRE(planner.makespan_after_us() < planner.makespan_before_us());

    // The plan, node by node. Positions survive because nothing here creates a
    // hazard between two nodes, so the topological sort is the identity.
    for (auto const &node : graph.nodes()) {
        unsigned const width = node.thread_width;
        if (node.label.rfind("fat_", 0) == 0) {
            REQUIRE(width > 1);
            REQUIRE(width <= p);
        } else {
            REQUIRE(width == 1);
        }
        // Every planned node carries an admission priority, which is what tells
        // the executor to use the plan's ordering instead of hop counts.
        REQUIRE(node.admission_priority > 0);
    }

    REQUIRE(graph.planned_thread_count() == p);
}

TEST_CASE("ThreadPlanning - a node under the serial floor stays at width 1", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 2) {
        SKIP("a one-thread machine plans nothing");
    }

    // Alone on its level, with the whole machine free, and still width 1: the
    // floor is not a scheduling decision, it is a statement that a fork costs
    // more than this node contains.
    cg::Graph  graph("under_the_floor");
    auto const in  = shell(graph, "in", kThinExtent);
    auto const out = shell(graph, "out", kThinExtent);
    graph.add_node(work_node("thin", in, out));

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    REQUIRE(planner.num_widened() == 0);
    REQUIRE(width_of(graph, 0) == 1);
    REQUIRE(planner.makespan_before_us() < cg::passes::ThreadPlanning::serial_floor_us());
}

TEST_CASE("ThreadPlanning - the grouped scalar ops are width 1 by contract", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    // One graph per kind, all three the same shape and the same size, so the
    // only thing that differs between the vetoed ones and the control is the
    // kind. Sharing one graph would not prove anything: a vetoed node sits on
    // the critical path at full width and holds the makespan up, so the control
    // beside it would be left narrow on the merits.
    auto plan_one = [p](cg::OpKind kind) {
        cg::Graph  graph(fmt::format("kind_{}", static_cast<int>(kind)));
        auto const in  = shell(graph, "in", kFatExtent);
        auto const out = shell(graph, "out", kFatExtent);
        graph.add_node(work_node("node", in, out, kind));

        cg::passes::ThreadPlanning planner(p);
        planner.run(graph);
        return width_of(graph, 0);
    };

    REQUIRE(plan_one(cg::OpKind::GroupedDot) == 1);
    REQUIRE(plan_one(cg::OpKind::GroupedAxpby) == 1);
    // The control proves the size, not the kind, is what the others were
    // rejected for.
    REQUIRE(plan_one(cg::OpKind::Custom) > 1);
}

TEST_CASE("ThreadPlanning - a BLAS route follows the vendor's moldability", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    cg::Graph  graph("blas_route");
    auto const in  = shell(graph, "in", kFatExtent);
    auto const out = shell(graph, "out", kFatExtent);
    graph.add_node(work_node("gemm", in, out, cg::OpKind::Gemm));

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    if (cg::blas_route_is_moldable()) {
        REQUIRE(width_of(graph, 0) > 1);
    } else {
        // Accelerate and reference BLAS: the width would be a number nothing
        // honors. An OpenMP-built OpenBLAS would honor it but cannot take
        // concurrent callers at different widths without corrupting, so it
        // lands here too. Either way the planner does not write one.
        REQUIRE(width_of(graph, 0) == 1);
    }
}

TEST_CASE("ThreadPlanning - a container is width 1 and its body is planned", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    cg::Graph  graph("loop_body");
    cg::Graph &body = graph.add_loop("loop", 4, [](size_t iter) { return iter < 3; });
    {
        auto const in  = shell(body, "body_in", kFatExtent);
        auto const out = shell(body, "body_out", kFatExtent);
        body.add_node(work_node("body_fat", in, out));
    }

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    REQUIRE(graph.num_nodes() == 1);
    REQUIRE(width_of(graph, 0) == 1); // the Loop itself holds no width
    REQUIRE(width_of(body, 0) > 1);   // its body was planned on its own
    REQUIRE(body.planned_thread_count() == p);
    // The container is priced by what it runs, so it is not mistaken for a
    // no-op when it sits on a critical path.
    REQUIRE(graph.nodes()[0].admission_priority > 1);
}

TEST_CASE("ThreadPlanning - the area never exceeds the makespan estimate", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    // A chain of stages, each a handful of independent nodes of assorted sizes
    // feeding the next stage, so the graph has both a real critical path and
    // real width. Sizes are a fixed pattern rather than random: a planner test
    // that fails one run in twenty is worse than no test.
    static constexpr size_t kExtents[] = {4096, 24, 2048, 64, 3072, 32, 1024, 48};

    cg::Graph    graph("mixed");
    cg::TensorId previous = shell(graph, "seed", 512);
    for (int stage = 0; stage < 4; stage++) {
        std::vector<cg::TensorId> produced;
        for (int lane = 0; lane < 4; lane++) {
            size_t const extent = kExtents[(stage * 4 + lane) % 8];
            auto const   out    = shell(graph, fmt::format("s{}_l{}", stage, lane), extent);
            graph.add_node(work_node(fmt::format("n_s{}_l{}", stage, lane), previous, out));
            produced.push_back(out);
        }
        previous = produced.front();
    }

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    double const tolerance = 1e-6 * std::max(1.0, planner.makespan_after_us());
    REQUIRE(planner.area_after_us() <= planner.makespan_after_us() + tolerance);
    REQUIRE(planner.makespan_after_us() >= planner.critical_path_after_us() - tolerance);
    REQUIRE(planner.makespan_after_us() <= planner.makespan_before_us() + tolerance);

    for (auto const &node : graph.nodes()) {
        REQUIRE(node.thread_width >= 1);
        REQUIRE(node.thread_width <= p);
    }
}

TEST_CASE("ThreadPlanning - a cold plan re-plans once, then freezes", "[ComputeGraph][ThreadPlanning]") {
    FatGraph  fixture;
    cg::Graph graph("cold_then_warm");
    fixture.capture(graph);

    graph.plan_threads();
    // Nothing has replayed, so the plan came from the model and one re-plan is
    // owed to the first real timings.
    REQUIRE(graph.thread_replan_armed());

    cg::DataflowExecutor df;
    graph.execute(df);

    // Fired at the end of that replay, and disarmed.
    REQUIRE_FALSE(graph.thread_replan_armed());

    std::vector<unsigned> const widths_after_replan = [&]() {
        std::vector<unsigned> w;
        for (auto const &node : graph.nodes()) {
            w.push_back(node.thread_width);
        }
        return w;
    }();

    graph.execute(df);
    auto const first = bytes_of(fixture.c1);
    REQUIRE_FALSE(graph.thread_replan_armed());

    graph.execute(df);
    auto const second = bytes_of(fixture.c1);

    // Widths frozen from here on, and with them the bits.
    for (size_t i = 0; i < graph.nodes().size(); i++) {
        REQUIRE(graph.nodes()[i].thread_width == widths_after_replan[i]);
    }
    REQUIRE(first == second);
}

TEST_CASE("ThreadPlanning - freeze never re-plans", "[ComputeGraph][ThreadPlanning]") {
    FatGraph  fixture;
    cg::Graph graph("frozen");
    fixture.capture(graph);

    graph.plan_threads(/*freeze=*/true);
    REQUIRE_FALSE(graph.thread_replan_armed());

    std::vector<unsigned> planned;
    for (auto const &node : graph.nodes()) {
        planned.push_back(node.thread_width);
    }

    cg::DataflowExecutor df;
    graph.execute(df);
    REQUIRE_FALSE(graph.thread_replan_armed());
    auto const first = bytes_of(fixture.c1);

    graph.execute(df);
    auto const second = bytes_of(fixture.c1);

    for (size_t i = 0; i < graph.nodes().size(); i++) {
        REQUIRE(graph.nodes()[i].thread_width == planned[i]);
    }
    REQUIRE(first == second);
}

TEST_CASE("ThreadPlanning - a plan for another machine is not run", "[ComputeGraph][ThreadPlanning]") {
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    FatGraph  fixture;
    cg::Graph graph("stale_plan");
    fixture.capture(graph);
    graph.plan_threads(/*freeze=*/true);

    bool any_wide = false;
    for (auto const &node : graph.nodes()) {
        any_wide = any_wide || node.thread_width > 1;
    }
    if (!any_wide) {
        SKIP("the cost model gave this graph no width, so there is no stale plan to reject");
    }

    // The widths are unchanged; only the machine they claim to be for moves.
    graph.set_planned_thread_count(p + 1);

    cg::reset_stale_thread_plan_fallbacks();
    cg::DataflowExecutor df;
    graph.execute(df);

    REQUIRE(cg::stale_thread_plan_fallbacks() == 1);
    // Still correct, just narrow: the fallback is today's unplanned behavior.
    auto const stale = values_of(fixture.c1);

    graph.set_planned_thread_count(p);
    graph.execute(df);
    REQUIRE(cg::stale_thread_plan_fallbacks() == 1);
    // Not bit-for-bit: this replay honors widths and the rejected one did not,
    // and the vendor fence makes that a different summation order rather than
    // the same one. What must hold is that rejecting a plan cost accuracy
    // nothing.
    auto const honored = worst_difference(values_of(fixture.c1), stale);
    INFO("stale-plan replay against the honored one, worst difference " << honored);
    REQUIRE(honored < 1e-12);
}

TEST_CASE("ThreadPlanning - a planned replay is deterministic at fixed P", "[ComputeGraph][ThreadPlanning]") {
    FatGraph  fixture;
    cg::Graph graph("determinism");
    fixture.capture(graph);
    graph.plan_threads(/*freeze=*/true);

    cg::DataflowExecutor df;
    graph.execute(df);
    auto const first_c1 = bytes_of(fixture.c1);
    auto const first_c2 = bytes_of(fixture.c2);

    graph.execute(df);
    REQUIRE(bytes_of(fixture.c1) == first_c1);
    REQUIRE(bytes_of(fixture.c2) == first_c2);
}

TEST_CASE("ThreadPlanning - the triples contractions are widened", "[ComputeGraph][ThreadPlanning]") {
    // The shape the moldable plan exists for: a rank-3 contraction against a
    // rank-2 factor, accumulated through a permute, six permutations deep. Both
    // kinds are einsums-threaded now - the contraction because PackedGemm keeps
    // it in its own packed loops under a width the wrappers would clamp - so
    // both must come back above 1 on a machine with threads to divide.
    unsigned const p = machine_threads();
    if (p < 4) {
        SKIP("plan needs at least four threads to have anything to divide");
    }

    constexpr size_t n = 220;

    cg::Graph  graph("triples_shaped");
    auto      &scratch    = graph.declare_runtime_tensor<double>("scratch", {n, n, n});
    auto      &w          = graph.declare_runtime_tensor<double>("W", {n, n, n});
    auto      &k          = graph.declare_runtime_tensor<double>("K_xvvv", {n, n, n});
    auto      &t          = graph.declare_runtime_tensor<double>("t_kj", {n, n});
    auto const scratch_id = graph.find_tensor_id_by_ptr(&scratch);
    auto const w_id       = graph.find_tensor_id_by_ptr(&w);
    auto const k_id       = graph.find_tensor_id_by_ptr(&k);
    auto const t_id       = graph.find_tensor_id_by_ptr(&t);

    for (int perm = 0; perm < 6; perm++) {
        cg::Node contract;
        contract.kind    = cg::OpKind::Einsum;
        contract.label   = fmt::format("abc <- abd ; cd [{}]", perm);
        contract.inputs  = {k_id, t_id};
        contract.outputs = {scratch_id};
        contract.execute = []() {};
        graph.add_node(std::move(contract));

        cg::Node accumulate;
        accumulate.kind    = cg::OpKind::Permute;
        accumulate.label   = fmt::format("W <- scratch [{}]", perm);
        accumulate.inputs  = {scratch_id};
        accumulate.outputs = {w_id};
        accumulate.execute = []() {};
        graph.add_node(std::move(accumulate));
    }

    cg::passes::ThreadPlanning planner(p);
    planner.run(graph);

    std::vector<std::string> widths;
    unsigned                 wide_contractions = 0;
    unsigned                 wide_permutes     = 0;
    for (auto const &node : graph.nodes()) {
        widths.push_back(fmt::format("{} = {}", node.label, node.thread_width));
        if (node.kind == cg::OpKind::Einsum && node.thread_width > 1) {
            wide_contractions++;
        }
        if (node.kind == cg::OpKind::Permute && node.thread_width > 1) {
            wide_permutes++;
        }
    }
    INFO(fmt::format("planned widths on {} threads: {}", p, fmt::join(widths, ", ")));
    REQUIRE(wide_contractions > 0);
    REQUIRE(wide_permutes > 0);
}
