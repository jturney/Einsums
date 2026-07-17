//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file RecursionPlumbing.cpp
/// @brief Tests for the sub-graph recursion infrastructure shared by
///        optimization passes, ``Graph::for_each_subgraph`` and
///        ``PassManager::run`` opt-in recursion via
///        ``OptimizerPass::recurse_into_subgraphs()``.
///
/// Behavior change is *opt-in per pass*; this file asserts the plumbing
/// itself works so that subsequent per-pass fixes can rely on it without
/// re-litigating the dispatch logic.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Records every (pass-name, graph-name) pair it sees. Used to assert
/// which graphs PassManager actually visited.
struct TraceEntry {
    std::string graph_name;
    size_t      num_nodes_at_call;
};

class TracingPass : public cg::OptimizerPass {
  public:
    TracingPass(std::vector<TraceEntry> *log, bool recurse) : _log(log), _recurse(recurse) {}

    [[nodiscard]] std::string name() const override { return "Tracing"; }

    bool run(cg::Graph &graph) override {
        _log->push_back({graph.name(), graph.num_nodes()});
        return false;
    }

    [[nodiscard]] bool recurse_into_subgraphs() const override { return _recurse; }

  private:
    std::vector<TraceEntry> *_log;
    bool                     _recurse;
};

// Builds a parent graph with one Loop and one Conditional child. Sub-graph
// names are the auto-generated ``<label>/body``, ``<label>/then``,
// ``<label>/else`` that ``Graph::add_loop`` / ``add_conditional`` assign at
// construction time, there's no ``set_name`` on Graph, the name is bound
// to the label passed in.
cg::Graph build_loop_and_conditional(Tensor<double, 2> &A, Tensor<double, 2> &B, Tensor<double, 2> &C) {
    cg::Graph g("parent");

    // ``CaptureGuard`` is non-nesting, every capture block runs against
    // exactly one graph and must exit before the next one starts. So we
    // (1) capture parent ops, (2) declare the loop/conditional nodes on
    // the parent (no capture needed for ``add_loop``/``add_conditional``
    // themselves), then (3) enter each child graph in its own scope.
    {
        cg::CaptureGuard const guard(g);
        cg::scale(2.0, &A);
    }

    auto &body            = g.add_loop("body", /*max_iterations=*/3, [&](size_t i) { return i < 2; });
    auto [then_g, else_g] = g.add_conditional("cond", [] { return true; });

    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ij;jk->ik", 0.0, &C, 1.0, A, B);
    }
    {
        cg::CaptureGuard const guard(then_g);
        cg::scale(1.5, &A);
    }
    {
        cg::CaptureGuard const guard(else_g);
        cg::scale(0.5, &B);
    }

    return g;
}

} // namespace

TEST_CASE("Graph::for_each_subgraph visits loop bodies", "[ComputeGraph][Recursion]") {
    auto A = create_zero_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph g("with_loop");
    {
        auto                  &body = g.add_loop("body", 5, [](size_t i) { return i < 1; });
        cg::CaptureGuard const guard(body);
        cg::einsum("ij;jk->ik", 0.0, &C, 1.0, A, B);
    }

    std::vector<std::string> visited;
    g.for_each_subgraph([&](cg::Graph const &sub) { visited.push_back(sub.name()); });

    REQUIRE(visited.size() == 1);
    CHECK(visited[0] == "body/body");
}

TEST_CASE("Graph::for_each_subgraph visits both conditional branches", "[ComputeGraph][Recursion]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    cg::Graph g("with_cond");
    {
        auto [then_g, else_g] = g.add_conditional("cond", [] { return true; });
        {
            cg::CaptureGuard const guard(then_g);
            cg::scale(1.0, &A);
        }
        {
            cg::CaptureGuard const guard(else_g);
            cg::scale(1.0, &B);
        }
    }

    std::vector<std::string> visited;
    g.for_each_subgraph([&](cg::Graph const &sub) { visited.push_back(sub.name()); });

    REQUIRE(visited.size() == 2);
    CHECK(visited[0] == "cond/then");
    CHECK(visited[1] == "cond/else");
}

TEST_CASE("Graph::for_each_subgraph yields only direct children (no auto-recursion)", "[ComputeGraph][Recursion]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    // ``CaptureGuard`` doesn't nest, so the build is three separate
    // captures: parent (adds loop), outer body (adds inner loop), inner
    // body (adds the actual op).
    cg::Graph g("outer");
    auto     &body  = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &inner = body.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const inner_guard(inner);
        cg::scale(2.0, &A);
    }

    std::vector<std::string> visited;
    g.for_each_subgraph([&](cg::Graph const &sub) { visited.push_back(sub.name()); });

    REQUIRE(visited.size() == 1);
    CHECK(visited[0] == "outer/body");
}

TEST_CASE("PassManager recurses for opt-in passes", "[ComputeGraph][Recursion][PassManager]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    auto g = build_loop_and_conditional(A, B, C);

    std::vector<TraceEntry> log;
    cg::PassManager         pm;
    pm.add<TracingPass>(&log, /*recurse=*/true);

    pm.run(g);

    // Expect: parent, then each direct child of parent, order matches
    // node iteration. The exact order in build_loop_and_conditional is:
    //   parent → body/body → cond/then → cond/else.
    REQUIRE(log.size() == 4);
    CHECK(log[0].graph_name == "parent");
    CHECK(log[1].graph_name == "body/body");
    CHECK(log[2].graph_name == "cond/then");
    CHECK(log[3].graph_name == "cond/else");
}

TEST_CASE("PassManager does NOT recurse for opt-out passes", "[ComputeGraph][Recursion][PassManager]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    auto g = build_loop_and_conditional(A, B, C);

    std::vector<TraceEntry> log;
    cg::PassManager         pm;
    pm.add<TracingPass>(&log, /*recurse=*/false);

    pm.run(g);

    REQUIRE(log.size() == 1);
    CHECK(log[0].graph_name == "parent");
}

TEST_CASE("PassManager recurses into nested loops for opt-in passes", "[ComputeGraph][Recursion][PassManager]") {
    auto A = create_zero_tensor<double>("A", 2, 2);

    cg::Graph g("outer");
    auto     &body  = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &inner = body.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const inner_guard(inner);
        cg::scale(2.0, &A);
    }

    std::vector<TraceEntry> log;
    cg::PassManager         pm;
    pm.add<TracingPass>(&log, /*recurse=*/true);

    pm.run(g);

    REQUIRE(log.size() == 3);
    CHECK(log[0].graph_name == "outer");
    CHECK(log[1].graph_name == "outer/body");
    CHECK(log[2].graph_name == "inner/body");
}

TEST_CASE("OptimizerPass::recurse_into_subgraphs() default is false", "[ComputeGraph][Recursion]") {
    // Sanity check on the base-class default so a future override that
    // accidentally flips the contract is loud.
    std::vector<TraceEntry> log;
    TracingPass const       flat(&log, /*recurse=*/false);
    CHECK_FALSE(flat.recurse_into_subgraphs());

    TracingPass const deep(&log, /*recurse=*/true);
    CHECK(deep.recurse_into_subgraphs());
}

TEST_CASE("LoopInvariantHoisting still opts out (manages its own descent)", "[ComputeGraph][Recursion]") {
    // LIH is the one pre-existing loop-aware pass. It walks
    // LoopDescriptor::body itself; the PassManager must not double-walk
    // by also recursing, or LIH would see its hoisted body twice.
    cg::passes::LoopInvariantHoisting const lih;
    CHECK_FALSE(lih.recurse_into_subgraphs());
}

// ── Per-pass recursion-policy contract ───────────────────────────────────
// Locks in which passes opt into PassManager auto-recursion. A pass that
// flips this accidentally (or a new pass that picks the wrong default)
// trips this test.
// pass.

TEST_CASE("Recursion policy - safe local-rewrite passes opt in", "[ComputeGraph][Recursion][Policy]") {
    CHECK(cg::passes::ScaleAbsorption{}.recurse_into_subgraphs());
    CHECK(cg::passes::ElementWiseFusion{}.recurse_into_subgraphs());
    CHECK(cg::passes::PermuteFusion{}.recurse_into_subgraphs());
    CHECK(cg::passes::DistributiveFactoring{}.recurse_into_subgraphs());
    CHECK(cg::passes::GEMMBatching{}.recurse_into_subgraphs());
    CHECK(cg::passes::StreamAssignment{}.recurse_into_subgraphs());
    // GPU transfer transforms (per-graph; self-contained body transfers).
    CHECK(cg::passes::TransferInsertion{}.recurse_into_subgraphs());
    CHECK(cg::passes::TransferElimination{}.recurse_into_subgraphs());
    // DeadNodeElimination opts in now that it treats tensors referenced by
    // a child sub-graph as live (Graph::collect_subtree_referenced_ptrs).
    CHECK(cg::passes::DeadNodeElimination{}.recurse_into_subgraphs());
    // ConstantFolding opts in now that it only folds nodes whose tensors
    // are materialized at pass time (skips deferred body tensors).
    CHECK(cg::passes::ConstantFolding{}.recurse_into_subgraphs());
    // SymmetryPropagation opts in: it only tags single-writer, not-used-by-
    // a-child tensors whose symmetry is structurally guaranteed.
    CHECK(cg::passes::SymmetryPropagation{}.recurse_into_subgraphs());
    // ContractionPlanning opts in: chain restructuring is associativity-
    // equivalent and its intermediates are eager (no Materialization dep).
    CHECK(cg::passes::ContractionPlanning{}.recurse_into_subgraphs());
    // Reorder opts in as of 2026-05-26: run() now encodes all three hazard
    // classes (RAW/WAW/WAR) with view-alias resolution + effective_io, so the
    // memory-aware sort no longer drops the WAR edges that broke the SCF
    // body's snapshot+recompute sequence. Guarded by the fuzzer's loop-body
    // "Reorder bait" patterns.
    CHECK(cg::passes::Reorder{}.recurse_into_subgraphs());
}

TEST_CASE("Recursion policy - hoisting / aggregation passes stay opt-out", "[ComputeGraph][Recursion][Policy]") {
    // Hoisting passes walk children themselves inside run() and emit into
    // the parent, they must not be auto-recursed.
    CHECK_FALSE(cg::passes::Materialization{}.recurse_into_subgraphs());
    CHECK_FALSE(cg::passes::FreeInsertion{}.recurse_into_subgraphs());
    CHECK_FALSE(cg::passes::GPUPlacement{}.recurse_into_subgraphs()); // shared-budget walk
    CHECK_FALSE(cg::passes::IOPrefetch{}.recurse_into_subgraphs());   // own walk + hoist out of loops
    // Analysis passes aggregate across sub-graphs inside run() rather than
    // being re-run per sub-graph (which would clobber their counters).
    CHECK_FALSE(cg::passes::MemoryPlanning{}.recurse_into_subgraphs());
    CHECK_FALSE(cg::passes::InplaceOptimization{}.recurse_into_subgraphs());
    CHECK_FALSE(cg::passes::GPUDiagnostics{}.recurse_into_subgraphs());
    // CSE stays opt-out: it has a soundness gap on the mutable-tensor reuse a
    // loop body exhibits, it merges nodes writing distinct, later-mutated
    // outputs. Recursing it broke the SCF example. See CSE.hpp.
    CHECK_FALSE(cg::passes::CSE{}.recurse_into_subgraphs());
}

TEST_CASE("DeadNodeElimination transforms a loop body via PassManager recursion", "[ComputeGraph][Recursion]") {
    // A dead body intermediate (written, never read). With DNE opted into
    // sub-graph recursion, running the PassManager eliminates it inside the
    // body: proving the recursion actually transforms the body, not just
    // the top-level graph. (We use DNE rather than CSE/Reorder, which stay
    // opt-out, see the policy test above.)
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph g("dne_in_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &dead = body.create_zero_tensor<double, 2>("dead", 4, 4);
        cg::einsum("ik;kj->ij", 0.0, &dead, 1.0, A, A); // written, never read
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, A);    // real work
    }
    size_t const before = body.num_nodes();

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    bool const modified = pm.run(g);

    CHECK(modified);
    CHECK(body.num_nodes() == before - 1); // dead node removed inside the body
}

TEST_CASE("CSE does NOT recurse into a loop body (deliberate opt-out)", "[ComputeGraph][Recursion][CSE]") {
    // CSE is unsound on mutable-tensor reuse (see CSE.hpp). It must leave
    // loop bodies untouched: two body einsums writing distinct outputs
    // stay as two nodes even though CSE would merge them at top level.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);

    cg::Graph g("cse_no_recurse");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, A, B);
    }
    REQUIRE(body.num_nodes() == 2);

    cg::PassManager pm;
    pm.add<cg::passes::CSE>();
    pm.run(g);

    CHECK(body.num_nodes() == 2); // body left untouched
}
