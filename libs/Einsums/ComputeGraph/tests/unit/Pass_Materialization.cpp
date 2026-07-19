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

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
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
