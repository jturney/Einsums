//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_DistributionPasses.cpp
/// @brief Unit tests for the distribution-planning passes: DistributionPlanning,
///        InputSlicing, and SUMMAExpansion.
///
/// All three short-circuit on comm::world_size() <= 1, so under the single-rank
/// build (mock, or MPI without mpirun) they are no-ops. These tests pin the
/// single-rank guard - the branch with no prior direct coverage - plus the
/// single-rank bookkeeping DistributionPlanning still performs, and confirm the
/// pipeline stays a numeric identity when nothing distributes. The firing paths
/// live under DistributedIntegration.cpp (run with mpirun -np N).

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

std::vector<cg::OpKind> kinds_of(cg::Graph const &g) {
    std::vector<cg::OpKind> ks;
    ks.reserve(g.nodes().size());
    for (auto const &n : g.nodes())
        ks.push_back(n.kind);
    return ks;
}

size_t deferred_count(cg::Graph const &g) {
    size_t c = 0;
    for (auto const &[tid, handle] : g.tensors_map())
        if (handle.alloc_state == cg::AllocState::Deferred)
            c++;
    return c;
}

template <typename T, size_t R>
void check_equal(Tensor<T, R> const &actual, Tensor<T, R> const &expected, double margin = 1e-10) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); i++)
        CHECK(actual.data()[i] == Catch::Approx(expected.data()[i]).margin(margin));
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// DistributionPlanning - single-rank behavior
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DistributionPlanning - no-op on a fully pre-allocated graph", "[ComputeGraph][Dist][DistributionPlanning]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: planning fires; this pins the single-rank guard");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);
    auto C = create_zero_tensor<double>("C", 8, 10);

    cg::Graph graph("plan-noop");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const before = kinds_of(graph);

    cg::passes::DistributionPlanning pass(/*threshold=*/1, /*enable_summa=*/false);
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_distributed() == 0);
    CHECK(pass.num_replicated() == 0); // no deferred tensors to classify
    CHECK(kinds_of(graph) == before);
}

TEST_CASE("DistributionPlanning - single rank counts deferred tensors as replicated but distributes none",
          "[ComputeGraph][Dist][DistributionPlanning]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: deferred tensors may distribute");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);

    cg::Graph graph("plan-deferred");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 8, 10);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    size_t const n_deferred = deferred_count(graph);
    REQUIRE(n_deferred >= 1); // C is deferred

    cg::passes::DistributionPlanning pass(/*threshold=*/1, /*enable_summa=*/false);
    // Single-rank branch: returns false, tallies deferred as replicated, and
    // does NOT flip is_distributed on any handle.
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_distributed() == 0);
    CHECK(pass.num_replicated() == n_deferred);
    for (auto const &[tid, handle] : graph.tensors_map())
        CHECK_FALSE(handle.is_distributed);
}

TEST_CASE("DistributionPlanning - single-rank no-op is idempotent", "[ComputeGraph][Dist][DistributionPlanning]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);

    cg::Graph graph("plan-idem");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 8, 10);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const                       before = kinds_of(graph);
    cg::passes::DistributionPlanning pass(/*threshold=*/1, /*enable_summa=*/false);
    CHECK_FALSE(pass.run(graph));
    CHECK_FALSE(pass.run(graph));
    CHECK(kinds_of(graph) == before);
}

TEST_CASE("DistributionPlanning - no-op on degenerate dim-1 deferred shapes", "[ComputeGraph][Dist][DistributionPlanning]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank");

    auto A = create_random_tensor<double>("A", 1, 6); // 1xN
    auto B = create_random_tensor<double>("B", 6, 1); // Nx1

    cg::Graph graph("plan-dim1");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 1, 1);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    cg::passes::DistributionPlanning pass(/*threshold=*/1, /*enable_summa=*/false);
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_distributed() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// InputSlicing - single-rank no-op guard
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("InputSlicing - no-op at single rank", "[ComputeGraph][Dist][InputSlicing]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: slicing may insert begin/end views; this pins the single-rank guard");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);

    cg::Graph graph("slice-noop");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 8, 10);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const before = kinds_of(graph);

    cg::passes::InputSlicing pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_sliced() == 0);
    // No begin_slice / end_slice Custom nodes were inserted.
    CHECK(kinds_of(graph) == before);

    CHECK_FALSE(pass.run(graph)); // idempotent no-op
    CHECK(kinds_of(graph) == before);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMAExpansion - single-rank / 1x1-grid no-op guard
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SUMMAExpansion - no-op at single rank (1x1 grid)", "[ComputeGraph][Dist][SUMMAExpansion]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: SUMMA may expand on a square grid; this pins the single-rank guard");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);

    cg::Graph graph("summa-noop");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 8, 10);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const before = kinds_of(graph);

    cg::passes::SUMMAExpansion pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_expanded() == 0);
    CHECK(kinds_of(graph) == before);
}

TEST_CASE("SUMMAExpansion - single-rank pipeline is a numeric identity", "[ComputeGraph][Dist][SUMMAExpansion]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: local partitions differ from the full reference");

    auto A = create_random_tensor<double>("A", 12, 8);
    auto B = create_random_tensor<double>("B", 8, 10);

    // Eager reference.
    auto C_ref = create_zero_tensor<double>("C_ref", 12, 10);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("summa-numeric");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 12, 10);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Full distribution pipeline including SUMMA; all passes no-op at one rank,
    // so the einsum executes unmodified and must match the eager result.
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/true);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::SUMMAExpansion>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);
    graph.execute();

    REQUIRE(C.is_materialized());
    check_equal(C, C_ref);
}
