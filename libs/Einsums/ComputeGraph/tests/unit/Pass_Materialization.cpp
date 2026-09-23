//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_Materialization.cpp
/// @brief Tests for the Materialization pass, focused on body-declared
///        deferred scratch: the hoisted Materialize / Initialize nodes must
///        carry a parent TensorId so the dependency builder orders them
///        before the owning Loop under a concurrent executor, not just by
///        node position.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
namespace cg = einsums::compute_graph;

namespace {

size_t count_nodes(cg::Graph const &g, cg::OpKind kind) {
    size_t n = 0;
    for (auto const &node : g.nodes()) {
        if (node.kind == kind) {
            n++;
        }
    }
    return n;
}

} // namespace

TEST_CASE("Materialization - body-declared deferred scratch hoists with a parent tid", "[ComputeGraph][Materialization][Loop][Dataflow]") {
    // A deferred+zero scratch DECLARED INSIDE a loop body is hoisted to the
    // parent so it is allocated once per outer execution. Pre-fix the hoisted
    // Materialize / Initialize carried EMPTY outputs (owns_tid=false), so they
    // floated as edgeless roots: the DataflowExecutor could run the Loop body
    // before the buffer was materialized. They must now carry a parent
    // TensorId so a RAW edge orders them before the Loop.
    constexpr size_t n   = 8;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("body_scratch_mat");
    auto     &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &W = body.scratch_zero<double, 2>("W", n, n); // deferred + intermediate + zero-init
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);                        // W = A*A, recomputed per iteration
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A);                      // acc += W*A
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    g.apply(pm);

    // Lifecycle nodes hoisted to the parent, none left in the body.
    REQUIRE(count_nodes(g, cg::OpKind::Materialize) == 1);
    REQUIRE(count_nodes(g, cg::OpKind::Initialize) == 1);
    REQUIRE(count_nodes(body, cg::OpKind::Materialize) == 0);
    REQUIRE(count_nodes(body, cg::OpKind::Initialize) == 0);

    // Materialize and Initialize land BEFORE the Loop and, the point of the
    // fix, carry a non-empty outputs list (the parent TensorId).
    size_t mat_pos = SIZE_MAX, init_pos = SIZE_MAX, loop_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        switch (g.nodes()[idx].kind) {
        case cg::OpKind::Materialize:
            mat_pos = idx;
            CHECK_FALSE(g.nodes()[idx].outputs.empty());
            break;
        case cg::OpKind::Initialize:
            init_pos = idx;
            CHECK_FALSE(g.nodes()[idx].outputs.empty());
            break;
        case cg::OpKind::Loop:
            loop_pos = idx;
            break;
        default:
            break;
        }
    }
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(init_pos != SIZE_MAX);
    REQUIRE(loop_pos != SIZE_MAX);
    REQUIRE(mat_pos < loop_pos);
    REQUIRE(init_pos < loop_pos);

    // Hand reference: two iterations of acc += (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = 2.0 * s;
        }
    }

    // Sequential path still correct.
    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }

    // Concurrent path: the hoisted buffer must be materialized before the loop
    // body reads it, every replay.
    for (int rep = 0; rep < 20; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
            }
        }
    }
}

TEST_CASE("Materialization - branch-declared deferred scratch hoists before the Conditional",
          "[ComputeGraph][Materialization][ControlFlow][Dataflow]") {
    // A deferred+zero scratch DECLARED INSIDE a conditional then-branch is
    // hoisted to the parent so it is materialized before the Conditional node
    // regardless of which branch runs. The hoisted Materialize must carry a
    // parent TensorId (non-empty outputs) so a RAW edge orders it before the
    // Conditional under a concurrent executor. Both predicate paths must
    // execute cleanly: the false path skips the then-branch but the buffer is
    // still allocated (unused), which must not break execution.
    constexpr size_t n   = 8;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("branch_scratch_mat");
    bool      take_then = true;

    auto [then_g, else_g] = g.add_conditional("branch", [&]() { return take_then; });
    {
        cg::CaptureGuard const guard(then_g);
        auto                  &W = then_g.scratch_zero<double, 2>("W", n, n); // deferred + intermediate + zero
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);                          // W = A*A
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A);                        // acc += W*A
    }
    // else_g left empty.

    cg::passes::Materialization mat;
    REQUIRE(mat.run(g));

    // Lifecycle hoisted to the parent, none left in the branch.
    REQUIRE(count_nodes(g, cg::OpKind::Materialize) == 1);
    REQUIRE(count_nodes(g, cg::OpKind::Initialize) == 1);
    REQUIRE(count_nodes(then_g, cg::OpKind::Materialize) == 0);
    REQUIRE(count_nodes(then_g, cg::OpKind::Initialize) == 0);

    // Materialize lands BEFORE the Conditional and carries a parent TensorId.
    size_t mat_pos = SIZE_MAX, init_pos = SIZE_MAX, cond_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        switch (g.nodes()[idx].kind) {
        case cg::OpKind::Materialize:
            mat_pos = idx;
            CHECK_FALSE(g.nodes()[idx].outputs.empty());
            break;
        case cg::OpKind::Initialize:
            init_pos = idx;
            CHECK_FALSE(g.nodes()[idx].outputs.empty());
            break;
        case cg::OpKind::Conditional:
            cond_pos = idx;
            break;
        default:
            break;
        }
    }
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(init_pos != SIZE_MAX);
    REQUIRE(cond_pos != SIZE_MAX);
    REQUIRE(mat_pos < cond_pos);
    REQUIRE(init_pos < cond_pos);

    // Hand reference for the true path: acc = (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = s;
        }
    }

    // True path (sequential): then-branch runs, acc = A^3.
    take_then = true;
    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }

    // False path (sequential): then-branch skipped, buffer still materialized
    // (unused), acc stays zero, execution must not throw.
    take_then = false;
    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(0.0, 1e-14));
        }
    }

    // True path under the concurrent executor: the hoisted buffer must be
    // materialized before the branch reads it, every replay.
    take_then = true;
    for (int rep = 0; rep < 10; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
            }
        }
    }
}

TEST_CASE("Materialization - inner-body deferred scratch hoists once and executes", "[ComputeGraph][Materialization][Loop][Dataflow]") {
    // Companion to the structural "nested body-declared" test: scratch declared
    // in the INNER of two nested loop bodies and consumed there. Exactly one
    // lifecycle pair lands in the outermost parent, none in either body, and the
    // graph produces the right numerics both sequentially and concurrently.
    constexpr size_t n   = 6;
    auto             A   = create_random_tensor<double>("A", n, n);
    auto             acc = create_zero_tensor<double>("acc", n, n);

    cg::Graph g("inner_scratch_mat");
    auto     &outer = g.add_loop("outer", 2, [](size_t it) { return it < 2; });
    auto     &inner = outer.add_loop("inner", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(inner);
        auto                  &W = inner.scratch_zero<double, 2>("W", n, n); // deferred + intermediate + zero
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);                         // W = A*A, recomputed per inner pass
        cg::einsum("ik;kj->ij", 1.0, &acc, 1.0, W, A);                       // acc += W*A
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::FreeInsertion>(size_t{0});
    g.apply(pm);

    // Single lifecycle pair in the outermost parent, none in either body.
    REQUIRE(count_nodes(g, cg::OpKind::Materialize) == 1);
    REQUIRE(count_nodes(g, cg::OpKind::Initialize) == 1);
    REQUIRE(count_nodes(outer, cg::OpKind::Materialize) == 0);
    REQUIRE(count_nodes(outer, cg::OpKind::Initialize) == 0);
    REQUIRE(count_nodes(inner, cg::OpKind::Materialize) == 0);
    REQUIRE(count_nodes(inner, cg::OpKind::Initialize) == 0);

    // Materialize / Initialize before the outer Loop (the only parent-level loop).
    size_t mat_pos = SIZE_MAX, init_pos = SIZE_MAX, loop_pos = SIZE_MAX;
    for (size_t idx = 0; idx < g.nodes().size(); idx++) {
        switch (g.nodes()[idx].kind) {
        case cg::OpKind::Materialize:
            mat_pos = idx;
            break;
        case cg::OpKind::Initialize:
            init_pos = idx;
            break;
        case cg::OpKind::Loop:
            loop_pos = idx;
            break;
        default:
            break;
        }
    }
    REQUIRE(mat_pos != SIZE_MAX);
    REQUIRE(init_pos != SIZE_MAX);
    REQUIRE(loop_pos != SIZE_MAX);
    REQUIRE(mat_pos < loop_pos);
    REQUIRE(init_pos < loop_pos);

    // Hand reference: 2 outer x 2 inner = 4 passes of acc += (A*A)*A.
    Tensor<double, 2> ref("ref", n, n);
    ref.zero();
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            double s = 0.0;
            for (size_t kk = 0; kk < n; kk++) {
                for (size_t ll = 0; ll < n; ll++) {
                    s += A(ii, kk) * A(kk, ll) * A(ll, jj);
                }
            }
            ref(ii, jj) = 4.0 * s;
        }
    }

    acc.zero();
    REQUIRE_NOTHROW(g.execute());
    for (size_t ii = 0; ii < n; ii++) {
        for (size_t jj = 0; jj < n; jj++) {
            REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
        }
    }

    for (int rep = 0; rep < 15; rep++) {
        acc.zero();
        cg::DataflowExecutor df;
        g.execute(df);
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t jj = 0; jj < n; jj++) {
                REQUIRE_THAT(acc(ii, jj), Catch::Matchers::WithinAbs(ref(ii, jj), 1e-11));
            }
        }
    }
}

TEST_CASE("Materialization - nested body-declared deferred scratch each hoist once with tids", "[ComputeGraph][Materialization][Loop]") {
    // Two deferred scratch tensors, one in an outer body and one in an inner
    // (nested) body, are each hoisted exactly once to the outermost parent
    // with a distinct parent TensorId. Exercises collect_descendant_deferred
    // and the ptr-keyed dedup.
    constexpr size_t n = 5;
    auto             A = create_random_tensor<double>("A", n, n);

    cg::Graph g("nested_mat");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &W1    = outer.scratch_zero<double, 2>("W1", n, n);
    auto     &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    auto     &W2    = inner.scratch_zero<double, 2>("W2", n, n);
    {
        cg::CaptureGuard const guard(outer);
        cg::einsum("ik;kj->ij", 0.0, &W1, 1.0, A, A);
    }
    {
        cg::CaptureGuard const guard(inner);
        cg::einsum("ik;kj->ij", 0.0, &W2, 1.0, A, A);
    }

    cg::passes::Materialization mat;
    REQUIRE(mat.run(g));

    // One lifecycle pair per tensor, both in the outermost parent.
    CHECK(count_nodes(g, cg::OpKind::Materialize) == 2);
    CHECK(count_nodes(g, cg::OpKind::Initialize) == 2);
    CHECK(count_nodes(outer, cg::OpKind::Materialize) == 0);
    CHECK(count_nodes(outer, cg::OpKind::Initialize) == 0);
    CHECK(count_nodes(inner, cg::OpKind::Materialize) == 0);
    CHECK(count_nodes(inner, cg::OpKind::Initialize) == 0);

    // Distinct, non-empty parent tids on the two Materialize nodes.
    std::vector<cg::TensorId> mat_tids;
    for (auto const &node : g.nodes()) {
        if (node.kind == cg::OpKind::Materialize) {
            REQUIRE(node.outputs.size() == 1);
            mat_tids.push_back(node.outputs.front());
        }
    }
    REQUIRE(mat_tids.size() == 2);
    CHECK(mat_tids[0] != mat_tids[1]);
}

TEST_CASE("Materialization - a graph-owned deferred tensor no node uses is left unallocated", "[ComputeGraph][Materialization]") {
    // What a structural rewrite leaves behind when it dissolves an intermediate: the declaration
    // stays, because a caller may still hold the handle, and the storage must not follow it.
    cg::Graph g("unused");
    auto      A      = create_random_tensor<double>("A", 4, 4);
    auto      R      = create_zero_tensor<double>("R", 4, 4);
    auto     &used   = g.declare_runtime_tensor<double>("used", {4, 4}, /*intermediate=*/true);
    auto     &unused = g.declare_runtime_tensor<double>("unused", {16, 16, 16, 16}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &used, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, used, A);
    }

    cg::passes::Materialization mat;
    REQUIRE(mat.run(g));
    CHECK(mat.num_materialized() == 1);
    CHECK(mat.num_unused() == 1);
    REQUIRE(count_nodes(g, cg::OpKind::Materialize) == 1);
    for (auto const &node : g.nodes()) {
        if (node.kind == cg::OpKind::Materialize) {
            REQUIRE(node.outputs.size() == 1);
            CHECK(node.outputs.front() == g.find_tensor_id_by_ptr(&used));
        }
    }
    CHECK(g.find_tensor(g.find_tensor_id_by_ptr(&unused))->alloc_state == cg::AllocState::Deferred);
    g.execute();

    // A graph whose only deferred tensor is unused gets no node at all, and says so.
    cg::Graph none("nothing");
    none.declare_runtime_tensor<double>("shell", {8, 8}, /*intermediate=*/true);
    cg::passes::Materialization second;
    CHECK_FALSE(second.run(none));
    CHECK(second.num_materialized() == 0);
    CHECK(second.num_unused() == 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// The two storage invariants, and the audit that states them
//
// Neither moves a number, which is why both of the defects behind them shipped:
// a buffer allocated for a tensor a rewrite dissolved costs memory, and a second
// lifecycle for one tensor re-runs an allocation that happens to be idempotent.
// The audit is what the differential fuzzers call, so it is tested here on
// graphs whose answer is known rather than only exercised through them.
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

/// A Materialize node for @p tid, of the shape the pass emits, spliced at @p position.
///
/// Planting one is how a test shows the audit BITES. Every other way of producing the
/// violation has been fixed, which is the point, and an invariant nothing can fail is
/// indistinguishable from an invariant nobody wrote.
void plant_materialize(cg::Graph &graph, std::string const &name, cg::TensorId tid, std::size_t position) {
    cg::Node planted;
    planted.kind    = cg::OpKind::Materialize;
    planted.label   = fmt::format("materialize({})", name);
    planted.outputs = {tid};
    planted.execute = []() {};

    std::vector<std::pair<std::size_t, std::vector<cg::Node>>> group;
    group.emplace_back(position, std::vector<cg::Node>{std::move(planted)});
    graph.insert_node_groups(std::move(group));
}

} // namespace

TEST_CASE("Materialization - the storage invariants hold across control flow and views", "[ComputeGraph][Materialization][audit]") {
    // The shapes the audit has to see through: a scratch used only inside a loop body, whose
    // lifecycle is HOISTED to the parent and whose use is therefore in a different graph from its
    // Materialize; and a scratch written only through a view, whose use names a handle the
    // Materialize does not.
    constexpr size_t n = 6;
    auto             A = create_random_tensor<double>("A", n, n);
    auto             R = create_zero_tensor<double>("R", n, n);

    cg::Graph g("audit_ok");
    auto     &sliced = g.declare_zero_runtime_tensor<double>("sliced", {n, n}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(g);
        auto                  &block = cg::view_runtime(sliced, {cg::ViewAxis::range(0, 2), cg::ViewAxis::full()});
        cg::axpby(1.0, A(Range{0, 2}, Range{0, n}), 0.0, &block);
    }
    auto &body = g.add_loop("iter", 2, [](size_t it) { return it < 2; });
    {
        cg::CaptureGuard const guard(body);
        auto                  &W = body.scratch_zero<double, 2>("W", n, n);
        cg::einsum("ik;kj->ij", 0.0, &W, 1.0, A, A);
        cg::einsum("ik;kj->ij", 1.0, &R, 1.0, W, A);
    }

    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    REQUIRE(pm.run(g));

    // Both invariants, on a graph where every deferred tensor is genuinely used.
    CHECK(cg::passes::duplicate_materializations(g).empty());
    CHECK(cg::passes::stranded_materializations(g).empty());
    g.execute();
}

TEST_CASE("Materialization - the audit names a Materialize nothing uses", "[ComputeGraph][Materialization][audit]") {
    cg::Graph g("audit_stranded");
    auto      A      = create_random_tensor<double>("A", 4, 4);
    auto      R      = create_zero_tensor<double>("R", 4, 4);
    auto     &used   = g.declare_runtime_tensor<double>("used", {4, 4}, /*intermediate=*/true);
    auto     &orphan = g.declare_runtime_tensor<double>("orphan", {4, 4}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &used, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, used, A);
    }

    cg::passes::Materialization mat;
    REQUIRE(mat.run(g));
    REQUIRE(cg::passes::stranded_materializations(g).empty());

    // The state a dissolving rewrite used to leave: the orphan's declaration stays, and a buffer
    // for it does too.
    plant_materialize(g, "orphan", g.find_tensor_id_by_ptr(&orphan), 0);
    auto const stranded = cg::passes::stranded_materializations(g);
    REQUIRE(stranded.size() == 1);
    CHECK(stranded.front() == "orphan");
    // Named because it is a graph-owned INTERMEDIATE. A caller-owned deferred tensor is one the
    // caller may read after the graph runs, so allocating an unused one is deliberate.
    CHECK(cg::passes::duplicate_materializations(g).empty());
}

TEST_CASE("Materialization - the audit names a tensor with two Materialize nodes", "[ComputeGraph][Materialization][audit]") {
    cg::Graph g("audit_duplicate");
    auto      A    = create_random_tensor<double>("A", 4, 4);
    auto      R    = create_zero_tensor<double>("R", 4, 4);
    auto     &used = g.declare_runtime_tensor<double>("used", {4, 4}, /*intermediate=*/true);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &used, 1.0, A, A);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, used, A);
    }

    cg::passes::Materialization mat;
    REQUIRE(mat.run(g));
    REQUIRE(cg::passes::duplicate_materializations(g).empty());

    plant_materialize(g, "used", g.find_tensor_id_by_ptr(&used), 0);
    auto const duplicated = cg::passes::duplicate_materializations(g);
    REQUIRE(duplicated.size() == 1);
    CHECK(duplicated.front() == "used");
}

TEST_CASE("Materialization - a tensor another pass already gave a lifecycle gets no second one",
          "[ComputeGraph][Materialization][ContractionPlanning]") {
    // ContractionPlanning emits a Materialize for the scratch it declares, so that applying it
    // standalone produces an executable graph, and this pass then found the same deferred
    // declaration and emitted another.
    //
    // Tolerated on the grounds that materialize_fn is idempotent, and that is the wrong test:
    // idempotence is a property of today's hook rather than a contract, the second node carries a
    // dependency edge that serializes the chain against itself, and every question this module
    // asks about a lifecycle is name-keyed, so a graph holding two of them has no single answer to
    // "who materializes this". Materialization now leaves a tensor that already has one alone,
    // which is what already_materialized_in was already doing for setup bodies.
    constexpr size_t n = 100;
    auto             A = create_random_tensor<double>("A", n, n);
    auto             B = create_random_tensor<double>("B", n, 1);
    auto             C = create_random_tensor<double>("C", 1, n);
    auto             R = create_zero_tensor<double>("R", n, n);

    // Captured as (B C) A, which builds an n x n intermediate and then does an n^3 contraction;
    // B (C A) is two n^2 ones. So the pass has a reason to re-bracket and a scratch to declare for
    // it. The running product goes in the A slot of the later member, which is the shape the chain
    // finder recognizes.
    cg::Graph g("cp_then_materialization");
    auto     &T1 = g.create_zero_tensor<double, 2>("T1", n, n);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("ik;kj->ij", 0.0, &T1, 1.0, B, C);
        cg::einsum("ik;kj->ij", 0.0, &R, 1.0, T1, A);
    }

    cg::CostModel model;
    model.cpu.peak_gflops_fp64          = 100.0;
    model.cpu.mem_bandwidth_gbps        = 40.0;
    model.cpu.kernel_launch_overhead_us = 0.1;
    model.cpu.name                      = "TestCPU";

    cg::passes::ContractionPlanning planning(model);
    REQUIRE(planning.run(g));
    REQUIRE(planning.chains_restructured() == 1);
    size_t const from_planning = count_nodes(g, cg::OpKind::Materialize);
    REQUIRE(from_planning >= 1);

    cg::passes::Materialization mat;
    mat.run(g);
    CHECK(count_nodes(g, cg::OpKind::Materialize) == from_planning);
    CHECK(cg::passes::duplicate_materializations(g).empty());
    CHECK(cg::passes::stranded_materializations(g).empty());

    // One lifecycle per chain scratch, and the scratch is real: the graph runs.
    size_t chain_scratch = 0;
    for (auto const &[tid, handle] : g.tensors_map()) {
        if (handle.name.starts_with("_cp_")) {
            chain_scratch++;
        }
    }
    CHECK(from_planning == chain_scratch);
    g.execute();
}

TEST_CASE("Materialization - a setup body's output gets ONE lifecycle, not two", "[ComputeGraph][Passes][Materialization]") {
    // Two arms of this pass place a lifecycle inside a setup body: the one that follows a
    // parent-declared tensor to the node that writes it, and the one that covers the body's
    // own workspace. The second used to run first, and its comment says a body's copy of a
    // parent-declared tensor carries no allocating hook and is therefore skipped there. That
    // is false for a deferred RUNTIME tensor, which capture adopts into the body complete
    // with a hook, so every graph with a setup body carried two Materialize nodes per output
    // and allocated each of them twice.
    //
    // Fixed by running the workspace arm second, where its name-keyed guard sees the node the
    // first arm placed. That node is the one that has to survive: it is built from the
    // PARENT's handle, and the parent's readers hold the buffer it allocates.
    auto out  = create_zero_tensor<double>("out", 3, 3);
    auto one  = create_zero_tensor<double>("one", 3, 3);
    one(0, 0) = 1.0;

    cg::Graph g("setup_lifecycle");
    auto     &fitted = g.declare_runtime_tensor<double>("fitted", {3, 3}, /*intermediate=*/true);
    {
        auto                  &body = g.add_setup("fit");
        cg::CaptureGuard const guard(body);
        // A body-declared scratch beside the parent-declared output, so both arms have work.
        auto &scratch = body.declare_runtime_tensor<double>("fit_scratch", {3, 3}, /*intermediate=*/true);
        cg::permute("ij <- ij", 0.0, &scratch, 1.0, one);
        cg::permute("ij <- ij", 0.0, &fitted, 1.0, scratch);
    }
    {
        cg::CaptureGuard const guard(g);
        cg::permute("ij <- ij", 0.0, &out, 1.0, fitted);
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);

    CHECK(cg::passes::duplicate_materializations(g).empty());
    CHECK(cg::passes::stranded_materializations(g).empty());

    g.execute();
    CHECK(out(0, 0) == Catch::Approx(1.0));
}

TEST_CASE("Materialization - a setup nested in a loop body materializes its workspace inside itself",
          "[ComputeGraph][Passes][Materialization]") {
    // A fitting emitted into a loop body, which is what a re-fitted amplitude is, puts a setup
    // node inside the body, and that setup declares its own workspace. The hoist walk used to
    // descend into it and give the workspace a lifecycle in the outermost parent. The body's
    // validation looks for a Materialize in its own nodes and its descendants, never in its
    // ancestors, and accepts a live allocation only if the parent's node has already run; that
    // node had no edge to the loop, so whether it ran first was a matter of schedule order, and
    // the same program passed on three platforms and failed on the fourth. The workspace of a
    // setup body is materialized inside that body at any depth, and this case says where.
    auto out  = create_zero_tensor<double>("out", 3, 3);
    auto one  = create_zero_tensor<double>("one", 3, 3);
    one(0, 0) = 1.0;

    cg::Graph g("nested_setup");
    auto     &first  = g.declare_runtime_tensor<double>("first", {3, 3}, /*intermediate=*/true);
    auto     &second = g.declare_runtime_tensor<double>("second", {3, 3}, /*intermediate=*/true);
    {
        // A parent-level setup with a workspace of the same name, so the two are told apart by
        // identity and not by name.
        auto                  &body = g.add_setup("fit_parent");
        cg::CaptureGuard const guard(body);
        auto                  &scratch = body.declare_runtime_tensor<double>("workspace", {3, 3}, /*intermediate=*/true);
        cg::permute("ij <- ij", 0.0, &scratch, 1.0, one);
        cg::permute("ij <- ij", 0.0, &first, 1.0, scratch);
    }
    cg::Graph *nested = nullptr;
    {
        auto &loop = g.add_loop("iterate", 2, [](std::size_t it) { return it + 1 < 2; });
        auto &fit  = loop.add_setup("fit_nested");
        nested     = &fit;
        cg::CaptureGuard const guard(fit);
        auto                  &scratch = fit.declare_runtime_tensor<double>("workspace", {3, 3}, /*intermediate=*/true);
        cg::permute("ij <- ij", 0.0, &scratch, 1.0, one);
        cg::permute("ij <- ij", 0.0, &second, 1.0, scratch);
    }
    {
        cg::CaptureGuard const guard(g);
        cg::permute("ij <- ij", 0.0, &out, 1.0, first);
        cg::permute("ij <- ij", 1.0, &out, 1.0, second);
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);

    auto const materialize_of = [](cg::Graph const &graph, std::string const &name) {
        return std::ranges::count_if(graph.nodes(), [&](cg::Node const &node) {
            return node.kind == cg::OpKind::Materialize && node.label == fmt::format("materialize({})", name);
        });
    };
    CHECK(materialize_of(*nested, "workspace") == 1);
    CHECK(materialize_of(g, "workspace") == 0);
    CHECK(cg::passes::duplicate_materializations(g).empty());
    CHECK(cg::passes::stranded_materializations(g).empty());

    g.execute();
    CHECK(out(0, 0) == Catch::Approx(2.0));
}

TEST_CASE("Materialization - a factor's lifecycle follows the setup that writes it through a reorder",
          "[ComputeGraph][Passes][Materialization][Reorder]") {
    // Two setups that share no tensor have no dependency between them, so Reorder is free to
    // put either first, and its memory heuristic does exactly that: the one whose body frees
    // the most is scheduled first whichever order the program declared them in. Every rule that
    // asked "which node produces this buffer" by looking at a POSITION got a different answer
    // per platform out of that, and two CI failures this week were the same shape.
    //
    // A setup lists the parent tensors its body writes as its outputs now, so the question has
    // an ordinary answer. This case asserts the answer twice, once per schedule order: the
    // setup that writes a factor carries it in node.outputs, and the factor's lifecycle lands
    // in that setup's body and nowhere else.
    //
    // Both orders are reached by moving the LARGE workspace from the first-declared setup to
    // the second: whichever body frees more is scheduled first, so one arm reorders and the
    // other does not, and between them both final orders are exercised.
    auto const run_arm = [](bool first_setup_is_large) {
        CAPTURE(first_setup_is_large);

        std::size_t const large = 48;
        std::size_t const small = 2;
        std::size_t const dim_a = first_setup_is_large ? large : small;
        std::size_t const dim_b = first_setup_is_large ? small : large;

        auto seed_a  = create_zero_tensor<double>("seed_a", dim_a, dim_a);
        auto seed_b  = create_zero_tensor<double>("seed_b", dim_b, dim_b);
        seed_a(0, 0) = 1.0;
        seed_b(0, 0) = 1.0;
        auto out_a   = create_zero_tensor<double>("out_a", dim_a, dim_a);
        auto out_b   = create_zero_tensor<double>("out_b", dim_b, dim_b);

        cg::Graph g("two_setups");
        auto     &factor_a = g.declare_runtime_tensor<double>("factor_a", {dim_a, dim_a}, /*intermediate=*/true);
        auto     &factor_b = g.declare_runtime_tensor<double>("factor_b", {dim_b, dim_b}, /*intermediate=*/true);

        cg::Graph *body_a = nullptr;
        cg::Graph *body_b = nullptr;
        {
            // Both bodies name their workspace identically, so the two are told apart by
            // identity and never by name.
            auto &body = g.add_setup("fit_a");
            body_a     = &body;
            cg::CaptureGuard const guard(body);
            auto                  &scratch = body.declare_runtime_tensor<double>("workspace", {dim_a, dim_a}, /*intermediate=*/true);
            cg::permute("ij <- ij", 0.0, &scratch, 1.0, seed_a);
            cg::permute("ij <- ij", 0.0, &factor_a, 1.0, scratch);
        }
        {
            auto &body = g.add_setup("fit_b");
            body_b     = &body;
            cg::CaptureGuard const guard(body);
            auto                  &scratch = body.declare_runtime_tensor<double>("workspace", {dim_b, dim_b}, /*intermediate=*/true);
            cg::permute("ij <- ij", 0.0, &scratch, 1.0, seed_b);
            cg::permute("ij <- ij", 0.0, &factor_b, 1.0, scratch);
        }
        {
            cg::CaptureGuard const guard(g);
            cg::permute("ij <- ij", 0.0, &out_a, 1.0, factor_a);
            cg::permute("ij <- ij", 0.0, &out_b, 1.0, factor_b);
        }

        cg::TensorId const id_a = g.find_tensor_id_by_ptr(&factor_a);
        cg::TensorId const id_b = g.find_tensor_id_by_ptr(&factor_b);
        REQUIRE(id_a != 0);
        REQUIRE(id_b != 0);

        auto pm = cg::PassManager::create_default();
        g.apply(pm);

        auto const setup_at = [&](std::string const &label) -> std::size_t {
            auto const &nodes = g.nodes();
            for (std::size_t i = 0; i < nodes.size(); i++) {
                if (nodes[i].kind == cg::OpKind::Setup && nodes[i].label == label) {
                    return i;
                }
            }
            FAIL(fmt::format("no setup node labelled '{}'", label));
            return 0;
        };
        std::size_t const pos_a = setup_at("fit_a");
        std::size_t const pos_b = setup_at("fit_b");

        // The larger body is scheduled first, so the two arms put the setups in opposite
        // orders and neither assertion below is reading a fixed schedule.
        CHECK((first_setup_is_large ? pos_a < pos_b : pos_b < pos_a));

        // The mechanism: each setup names what its body produces, and nothing else's.
        auto const writes = [&](std::size_t pos, cg::TensorId tid) {
            auto const &outs = g.nodes()[pos].outputs;
            return std::ranges::any_of(outs, [&](cg::TensorId out) { return g.resolve_alias(out) == g.resolve_alias(tid); });
        };
        CHECK(writes(pos_a, id_a));
        CHECK_FALSE(writes(pos_a, id_b));
        CHECK(writes(pos_b, id_b));
        CHECK_FALSE(writes(pos_b, id_a));

        // The consequence: the lifecycle of a factor is inside the body that writes it.
        auto const materialize_of = [](cg::Graph const &graph, std::string const &name) {
            return std::ranges::count_if(graph.nodes(), [&](cg::Node const &node) {
                return node.kind == cg::OpKind::Materialize && node.label == fmt::format("materialize({})", name);
            });
        };
        CHECK(materialize_of(*body_a, "factor_a") == 1);
        CHECK(materialize_of(*body_b, "factor_b") == 1);
        CHECK(materialize_of(*body_a, "factor_b") == 0);
        CHECK(materialize_of(*body_b, "factor_a") == 0);
        CHECK(materialize_of(g, "factor_a") == 0);
        CHECK(materialize_of(g, "factor_b") == 0);
        CHECK(materialize_of(*body_a, "workspace") == 1);
        CHECK(materialize_of(*body_b, "workspace") == 1);
        CHECK(materialize_of(g, "workspace") == 0);

        CHECK(cg::passes::duplicate_materializations(g).empty());
        CHECK(cg::passes::stranded_materializations(g).empty());

        g.execute();
        CHECK(out_a(0, 0) == Catch::Approx(1.0));
        CHECK(out_b(0, 0) == Catch::Approx(1.0));
    };

    run_arm(true);
    run_arm(false);
}

TEST_CASE("Materialization - a top-level setup's chain intermediate is materialized inside the body",
          "[ComputeGraph][Passes][Materialization][Setup]") {
    // The shape a basis truncation's projection has: a setup body writes a parent-declared
    // tensor through a CHAIN, and the buffer between the two steps belongs to nobody the
    // parent's algebra names. It is not a setup output, because no parent node reads it, and
    // it is not the parent's own scratch, because the parent never declared it; it reaches a
    // lifecycle only because a setup node lists what its body writes, so the parent's usage
    // analysis puts a request at the setup's position and the search descends from there.
    //
    // Asserted as a PLACE and not as a successful execute, which is the lesson the placement
    // note records: a lifecycle in the parent runs before the setup often enough that execute
    // says nothing about where it landed, and a body's own validation looks in its nodes and
    // its descendants and never in its ancestors.
    auto out  = create_zero_tensor<double>("out", 3, 3);
    auto one  = create_zero_tensor<double>("one", 3, 3);
    one(0, 0) = 2.0;

    cg::Graph  g("chain_setup");
    auto      &projected = g.declare_runtime_tensor<double>("projected", {3, 3}, /*intermediate=*/true);
    cg::Graph *body      = nullptr;
    {
        auto &fit = g.add_setup_at("project", 0);
        body      = &fit;
        // Declared BEFORE the guard, which is where a workspace of a setup body belongs: a
        // declaration is not a capture, and the buffer has to exist before the nodes reading it.
        auto &half = fit.declare_runtime_tensor<double>("chain_half", {3, 3}, /*intermediate=*/true);

        cg::CaptureGuard const guard(fit);
        cg::permute("ij <- ij", 0.0, &half, 0.5, one);
        cg::permute("ij <- ij", 0.0, &projected, 1.0, half);
    }
    {
        cg::CaptureGuard const guard(g);
        cg::permute("ij <- ij", 0.0, &out, 1.0, projected);
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);

    auto const materialize_of = [](cg::Graph const &graph, std::string const &name) {
        return std::ranges::count_if(graph.nodes(), [&](cg::Node const &node) {
            return node.kind == cg::OpKind::Materialize && node.label == fmt::format("materialize({})", name);
        });
    };
    CHECK(materialize_of(*body, "chain_half") == 1);
    CHECK(materialize_of(g, "chain_half") == 0);
    CHECK(materialize_of(*body, "projected") == 1);
    CHECK(materialize_of(g, "projected") == 0);
    CHECK(cg::passes::duplicate_materializations(g).empty());
    CHECK(cg::passes::stranded_materializations(g).empty());

    g.execute();
    CHECK(out(0, 0) == Catch::Approx(1.0));
}

TEST_CASE("Materialization - a body handle whose last use a rewrite removed gets no lifecycle", "[ComputeGraph][Materialization][audit]") {
    // The program shape the region fuzz found: an intermediate written in the parent and read
    // only inside a loop body. Two of the default pipeline's rewrites then take the use away.
    // LoopInvariantHoisting moves the body's one statement up, because nothing in it varies with
    // the iteration, which leaves the BODY holding a handle for the intermediate and no node that
    // names it. ContractionPlanning then re-associates the chain the hoist assembled, reads the
    // two inputs directly and drops the intermediate altogether.
    //
    // Nothing anywhere uses the buffer at that point. The pass's own arm for the parent's tensors
    // has declined to allocate such a tensor since the CCSD tau terms; its arm for a descendant's
    // asked only whether the body HELD a deferred handle, which a body goes on doing after its
    // last use is gone, and so it hoisted a lifecycle for a tensor whose whole point was to stop
    // existing.
    constexpr size_t big = 64, small = 16;
    auto             left  = create_random_tensor<double>("left", big, small);
    auto             right = create_random_tensor<double>("right", small, big);
    auto             out   = create_zero_tensor<double>("out", big, small);

    cg::Graph g("body_handle_without_a_use");
    auto     &half = g.scratch<double, 2>("half", big, big);
    {
        cg::CaptureGuard const guard(g);
        cg::einsum("qp;ps->qs", 0.0, &half, 1.0, left, right);
    }
    auto &body = g.add_loop("iteration", 1, [](size_t it) { return it < 0; });
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("qs;sr->qr", 0.0, &out, 1.0, half, left);
    }

    // The rewrites that reach this state are LoopInvariantHoisting followed by
    // ContractionPlanning, and whether the second one fires depends on the MEASURED cost model
    // of the machine: it re-associated the chain on one platform and left it alone on three
    // others, and a precondition that depends on that is not a precondition. So the state is
    // constructed directly. The body's one statement and the parent's writer are removed, as
    // the two rewrites remove them, and what is left is exactly what the pass then walked: a
    // body holding a handle for a tensor no node anywhere names.
    body.erase_nodes(std::vector<bool>(body.nodes().size(), true));
    {
        std::vector<bool> remove(g.nodes().size(), false);
        for (size_t i = 0; i < g.nodes().size(); ++i) {
            remove[i] = g.nodes()[i].kind == cg::OpKind::Einsum;
        }
        g.erase_nodes(remove);
    }

    auto pm = cg::PassManager::create_default();
    g.apply(pm);

    // The precondition, asserted rather than assumed: nothing names the tensor, so the case is
    // about a handle nothing uses and not about an ordinary live intermediate.
    auto const uses_of = [](cg::Graph const &graph, std::string const &name) {
        size_t seen = 0;
        for (auto const &node : graph.nodes()) {
            if (node.kind == cg::OpKind::Materialize || node.kind == cg::OpKind::Initialize) {
                continue;
            }
            auto const note = [&](cg::TensorId tid) {
                if (cg::TensorHandle const *handle = graph.find_tensor(graph.resolve_alias(tid)); handle != nullptr) {
                    seen += static_cast<size_t>(handle->name == name);
                }
            };
            std::ranges::for_each(node.inputs, note);
            std::ranges::for_each(node.outputs, note);
        }
        return seen;
    };
    REQUIRE(uses_of(g, "half") == 0);
    REQUIRE(uses_of(body, "half") == 0);
    // And the body still HOLDS the handle, which is what the pass was walking.
    REQUIRE(std::ranges::any_of(body.tensors_map(), [](auto const &entry) { return entry.second.name == "half"; }));

    auto const materialize_of = [](cg::Graph const &graph, std::string const &name) {
        return std::ranges::count_if(graph.nodes(), [&](cg::Node const &node) {
            return node.kind == cg::OpKind::Materialize && node.label == fmt::format("materialize({})", name);
        });
    };
    CHECK(materialize_of(g, "half") == 0);
    CHECK(materialize_of(body, "half") == 0);
    CHECK(cg::passes::stranded_materializations(g).empty());
    CHECK(cg::passes::duplicate_materializations(g).empty());

    // Nothing computes `out` any more; what execute proves is that validation reaches no
    // deferred handle, which is the failure a hoisted lifecycle for this tensor used to hide.
    g.execute();
}
