//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A loop body that slices its operands through parameter-bound views is the
// shape every blocked algorithm takes: write_param advances the induction
// variable, the views re-resolve, the body works on one block. The dependency
// between the write and the slice lives in the ParamTable and touches no
// tensor, so every pass that reasons from TensorIds alone used to get it wrong:
//
//   ConstantFolding       folded the write_param (no tensor inputs, therefore
//                         "all inputs constant") and replaced it with a no-op
//   LoopInvariantHoisting hoisted it out of the loop for the same reason
//   Reorder               floated a view ahead of the write that positions it,
//                         throwing "parameter 'j' is not set" on iteration one
//
// Each is checked below in isolation, then under the full default pipeline.
// See param_writes / param_reads / has_runtime_view_bounds in Node.hpp.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>
#include <Einsums/ComputeGraph/PredExpr.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// A[row] for row = 0, 1, 2 summed into `total`, one row per iteration, with
/// the row index carried by a parameter. Correct execution leaves
/// total = sum of every element of A.
struct RowSumLoop {
    RuntimeTensor<double> A{"A", {3UL, 3UL}};
    RuntimeTensor<double> row{"row", {3UL}};
    RuntimeTensor<double> total{"total", {3UL}};
    RuntimeTensor<double> big{"big", {3UL, 4096UL}};
    RuntimeTensor<double> bulk{"bulk", {4096UL}};
    cg::Graph             graph{"param_loop"};
    size_t                iter{0};

    RowSumLoop() {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                A(i, j) = static_cast<double>(1 + (3 * i) + j); // 1..9
            }
        }
        total.zero();

        auto                  &body = graph.add_loop("rows", 3, [this](size_t) {
            ++iter;
            return iter < 3;
        });
        cg::CaptureGuard const capture(body);
        // Two parameters, and a big scratch tensor whose liveness gives the
        // memory-aware reschedule something to want to move: enough for an
        // unguarded Reorder to float a slice ahead of the write that positions
        // it (the second parameter is the one it reached for in practice).
        cg::write_param("r", std::function<std::int64_t()>([this] { return static_cast<std::int64_t>(iter); }));
        cg::write_param("c", std::function<std::int64_t()>([this] { return static_cast<std::int64_t>(iter); }));
        auto &slice = cg::view_runtime(A, {cg::ViewAxis::drop("r"), cg::ViewAxis::full()});
        auto &wide  = cg::view_runtime(big, {cg::ViewAxis::drop("c"), cg::ViewAxis::full()});
        cg::axpby(1.0, slice, 0.0, &row);
        cg::axpby(1.0, wide, 0.0, &bulk);
        cg::scale(0.0, &bulk);
        cg::axpby(1.0, row, 1.0, &total);
    }

    double run() {
        iter = 0;
        total.zero();
        graph.execute();
        return total(0) + total(1) + total(2);
    }

    [[nodiscard]] cg::Graph const &body_graph() const {
        for (auto const &node : graph.nodes()) {
            if (auto const *loop = node.op_data.get_if<cg::LoopDescriptor>(); loop != nullptr && loop->body) {
                return *loop->body;
            }
        }
        throw std::logic_error("no loop node");
    }

    /// Nodes still inside the loop body (the hoist moves them to the parent).
    [[nodiscard]] size_t body_nodes() const { return body_graph().num_nodes(); }
};

double run_with(std::shared_ptr<cg::OptimizerPass> pass) {
    RowSumLoop      loop;
    cg::PassManager pm;
    if (pass) {
        pm.add(std::move(pass));
    }
    loop.graph.apply(pm);
    return loop.run();
}

} // namespace

TEST_CASE("Parametric loop - unoptimized baseline", "[ComputeGraph][View][WriteParam]") {
    RowSumLoop loop;
    REQUIRE(loop.run() == Catch::Approx(45.0)); // 1 + 2 + ... + 9
    // Replays must restart the sweep, not continue it.
    REQUIRE(loop.run() == Catch::Approx(45.0));
}

TEST_CASE("ConstantFolding leaves a write_param alone", "[ComputeGraph][Pass][ConstantFolding][WriteParam]") {
    REQUIRE(run_with(std::make_shared<cg::passes::ConstantFolding>()) == Catch::Approx(45.0));
}

TEST_CASE("LoopInvariantHoisting keeps a write_param in the body", "[ComputeGraph][Pass][LoopInvariantHoisting][WriteParam]") {
    RowSumLoop      loop;
    size_t const    before = loop.body_nodes();
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::LoopInvariantHoisting>());
    loop.graph.apply(pm);
    // Neither the parameter write nor the slice it positions may leave the body.
    REQUIRE(loop.body_nodes() == before);
    REQUIRE(loop.run() == Catch::Approx(45.0));
}

TEST_CASE("Reorder keeps a parameter write ahead of the slice that reads it", "[ComputeGraph][Pass][Reorder][WriteParam]") {
    RowSumLoop      loop;
    cg::PassManager pm;
    pm.add(std::make_shared<cg::passes::Reorder>());
    loop.graph.apply(pm);

    // Structural, not just numeric: assert the schedule itself. Whether this
    // particular body tempts the memory-aware reschedule depends on its
    // heuristic (the blocked-triples body in CG_TriplesBlocking did, and threw
    // "parameter 'j' is not set" on iteration one), so pin the invariant rather
    // than one heuristic's current output.
    auto const &body = loop.body_graph();
    for (size_t reader = 0; reader < body.nodes().size(); ++reader) {
        for (auto const &name : cg::param_reads(body.nodes()[reader])) {
            bool written_earlier = false;
            for (size_t writer = 0; writer < reader; ++writer) {
                auto const written = cg::param_writes(body.nodes()[writer]);
                written_earlier    = written_earlier || std::ranges::find(written, name) != written.end();
            }
            INFO("parameter '" << name << "' is read at position " << reader << " with no earlier write");
            REQUIRE(written_earlier);
        }
    }
    REQUIRE(loop.run() == Catch::Approx(45.0));
}

TEST_CASE("Parametric loop survives the default pipeline", "[ComputeGraph][Pass][View][WriteParam]") {
    RowSumLoop loop;
    auto       pm = cg::PassManager::create_default();
    loop.graph.apply(pm);
    REQUIRE(loop.run() == Catch::Approx(45.0));
    REQUIRE(loop.run() == Catch::Approx(45.0));
}

namespace {

/// A 3x3 matrix holding 1..9 by rows.
RuntimeTensor<double> one_to_nine() {
    RuntimeTensor<double> A{"A", {3UL, 3UL}};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(1 + (3 * i) + j);
        }
    }
    return A;
}

} // namespace

// A loop body is a scope of its graph, not a separate one: a parameter written
// before the loop is the one the body's slices resolve against. A captured body
// used to hold a table of its own and threw "parameter 'r' is not set", while
// the same graph saved and loaded shared the root's and ran.
TEST_CASE("Parametric loop - the body reads a parameter written before the loop", "[ComputeGraph][View][WriteParam]") {
    auto                  A = one_to_nine();
    RuntimeTensor<double> total{"total", {3UL}};
    total.zero();

    cg::Graph graph("param_before_loop");
    {
        cg::CaptureGuard const capture(graph);
        cg::write_param("r", cg::BoundExpr(1));
    }
    auto &body = graph.add_loop("rows", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const capture(body);
        auto                  &slice = cg::view_runtime(A, {cg::ViewAxis::drop("r"), cg::ViewAxis::full()});
        cg::axpby(1.0, slice, 1.0, &total);
    }

    REQUIRE_NOTHROW(graph.execute());
    CHECK(total(0) + total(1) + total(2) == 45.0); // 3 iterations of 4 + 5 + 6
}

// The loop's condition and its body read one table: a condition on a parameter
// the body writes sees each write. The condition used to read the parent's
// table while the body wrote its own, so the parameter was never set there.
TEST_CASE("Parametric loop - the condition reads a parameter the body writes", "[ComputeGraph][WriteParam]") {
    RuntimeTensor<double> one{"one", {1UL}};
    RuntimeTensor<double> count{"count", {1UL}};
    one(0) = 1.0;
    count.zero();
    std::int64_t runs = 0;

    cg::Graph graph("param_condition");
    auto     &body = graph.add_loop("until_k", 10, cg::PredExpr::compare("k", cg::CmpOp::Lt, 3));
    {
        cg::CaptureGuard const capture(body);
        cg::write_param("k", std::function<std::int64_t()>([&runs] { return ++runs; }));
        cg::axpy(1.0, one, &count);
    }

    REQUIRE_NOTHROW(graph.execute());
    CHECK(count(0) == 3.0);
}

// A conditional's branches are scopes of the same graph too; they held tables
// of their own as loop bodies did.
TEST_CASE("Parametric loop - a branch reads a parameter written before the conditional", "[ComputeGraph][View][WriteParam]") {
    auto                  A = one_to_nine();
    RuntimeTensor<double> row{"row", {3UL}};
    row.zero();

    cg::Graph graph("param_branch");
    {
        cg::CaptureGuard const capture(graph);
        cg::write_param("r", cg::BoundExpr(2));
    }
    auto [then_branch, else_branch] = graph.add_conditional("always", cg::PredExpr::always(true));
    {
        cg::CaptureGuard const capture(then_branch);
        auto                  &slice = cg::view_runtime(A, {cg::ViewAxis::drop("r"), cg::ViewAxis::full()});
        cg::axpby(1.0, slice, 0.0, &row);
    }

    REQUIRE_NOTHROW(graph.execute());
    CHECK(row(0) + row(1) + row(2) == 24.0); // 7 + 8 + 9
}
