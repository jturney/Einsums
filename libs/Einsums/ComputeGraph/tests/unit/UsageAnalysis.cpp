//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file UsageAnalysis.cpp
/// @brief Tests for the graph-owned reader/writer/liveness index that the
///        optimizer passes share (pr18): canonical alias resolution,
///        effective-IO coverage of control-flow nodes, the per-call-site
///        filter policies, and cache invalidation at the declared mutation
///        points.

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
/// The graph has no public ptr->TensorId lookup; tests scan the handle table.
template <typename T>
cg::TensorId tid_of(cg::Graph const &g, T const &t) {
    for (auto const &[id, h] : g.tensors_map()) {
        if (h.tensor_ptr == static_cast<void const *>(&t)) {
            return id;
        }
    }
    FAIL("tensor not registered with graph");
    return 0;
}
} // namespace

TEST_CASE("UsageAnalysis - positions, directions, and helpers", "[ComputeGraph][UsageAnalysis]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    cg::Graph graph("ua_basic");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);          // node 0: writes C, reads A, B
        cg::gemm<false, false>(1.0, C, B, 0.0, &D); // node 1: reads C, B; writes D
        cg::scale(2.0, &D);                         // node 2: RMW on D
    }
    graph.topological_sort();

    auto const &ua = graph.usage();

    // C: written at 0, read at 1.
    auto const *c_use = ua.find(graph, tid_of(graph, C));
    REQUIRE(c_use != nullptr);
    CHECK(c_use->first_writer() == 0);
    CHECK(c_use->last_writer() == 0);
    CHECK(c_use->last_use() == 1);
    CHECK(c_use->writes() == 1);
    CHECK(c_use->reads() == 1);
    CHECK(c_use->has_writer_before(1));
    CHECK_FALSE(c_use->has_writer_before(0));

    // B: read at 0 and 1, never written.
    auto const *b_use = ua.find(graph, tid_of(graph, B));
    REQUIRE(b_use != nullptr);
    CHECK(b_use->first_writer() == cg::TensorUsage::npos);
    CHECK(b_use->reads() == 2);
    CHECK(b_use->first_use() == 0);
    CHECK(b_use->last_use() == 1);

    // D: written at 1, read+written at 2 (scale is RMW).
    auto const *d_use = ua.find(graph, tid_of(graph, D));
    REQUIRE(d_use != nullptr);
    CHECK(d_use->first_writer() == 1);
    CHECK(d_use->last_writer() == 2);
    CHECK(d_use->last_use() == 2);
}

TEST_CASE("UsageAnalysis - loop body uses surface via subtree flag", "[ComputeGraph][UsageAnalysis]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto acc = create_zero_tensor<double>("acc", 4, 4);

    cg::Graph graph("ua_loop");
    auto     &body = graph.add_loop("iter", 3, [](size_t i) { return i < 3; });
    {
        cg::CaptureGuard const guard(body);
        cg::axpy(1.0, A, &acc); // acc += A, inside the loop body only
    }
    graph.topological_sort();

    auto const &ua    = graph.usage();
    auto const *a_use = ua.find(graph, tid_of(graph, A));

    // A is only used inside the body: visible with subtree expansion (at the
    // Loop node's position), invisible without it - the distinction the raw
    // per-pass scans historically depended on.
    REQUIRE(a_use != nullptr);
    CHECK(a_use->first_use(/*include_subtree=*/true) != cg::TensorUsage::npos);
    CHECK(a_use->first_use(/*include_subtree=*/false) == cg::TensorUsage::npos);
    REQUIRE(!a_use->uses.empty());
    CHECK(a_use->uses.front().via_subtree);
    CHECK(a_use->uses.front().kind == cg::OpKind::Loop);
}

TEST_CASE("UsageAnalysis - lifecycle filters (Alloc creation, Free scheduling edge)", "[ComputeGraph][UsageAnalysis]") {
    constexpr size_t N   = 400; // clears FreeInsertion's min-bytes threshold
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("ua_lifecycle");
    auto     &tmp = graph.scratch<double, 2>("tmp", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &out, tmp, B);
    }
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.topological_sort();

    auto const &ua      = graph.usage();
    auto const *tmp_use = ua.find(graph, tid_of(graph, tmp));
    REQUIRE(tmp_use != nullptr);

    // The pipeline gave tmp a Materialize (writer) and a Free (final "use").
    // With the FreeInsertion policy filters, the last REAL use precedes the
    // Free and the first REAL writer is the producing einsum, not the
    // lifecycle machinery.
    size_t const last_any  = tmp_use->last_use(/*ignore_free=*/false);
    size_t const last_real = tmp_use->last_use(/*ignore_free=*/true);
    CHECK(last_real < last_any); // the Free sits after the last consumer

    size_t const first_writer_any = tmp_use->first_writer(/*ignore_alloc=*/false);
    CHECK(first_writer_any <= tmp_use->first_writer(/*ignore_alloc=*/true));
}

TEST_CASE("UsageAnalysis - cache invalidation at mutation points", "[ComputeGraph][UsageAnalysis]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    cg::Graph graph("ua_invalidate");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, A);
    }
    graph.topological_sort();

    auto const *c_before = graph.usage().find(graph, tid_of(graph, C));
    REQUIRE(c_before != nullptr);
    CHECK(c_before->reads() == 0);

    // Capture another node reading C: add_node must invalidate the cache.
    {
        cg::CaptureGuard const guard(graph);
        cg::gemm<false, false>(1.0, C, A, 0.0, &D);
    }
    graph.topological_sort();

    auto const *c_after = graph.usage().find(graph, tid_of(graph, C));
    REQUIRE(c_after != nullptr);
    CHECK(c_after->reads() == 1);
    CHECK(c_after->last_use() == 1);
}
