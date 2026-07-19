//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_CommunicationPasses.cpp
/// @brief Unit tests for the communication passes: CommunicationElimination
///        (rank-independent, redundant-Allreduce removal), CommunicationInsertion,
///        and CommunicationScheduling.
///
/// CommunicationInsertion and CommunicationScheduling short-circuit on
/// comm::world_size() <= 1, so under the single-rank build (mock or MPI without
/// mpirun) they are pure no-ops. Those cases pin the no-op guard - the branch
/// that had no direct coverage before. CommunicationElimination has NO rank
/// guard, so its full contract is exercised here on hand-built graphs.

#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <functional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

namespace {

// Snapshot of node kinds, so a no-op guard can prove nothing was added,
// removed, or retyped.
std::vector<cg::OpKind> kinds_of(cg::Graph const &g) {
    std::vector<cg::OpKind> ks;
    ks.reserve(g.nodes().size());
    for (auto const &n : g.nodes())
        ks.push_back(n.kind);
    return ks;
}

// Build an Allreduce node for a given tensor id. Optional execute lambda lets a
// test observe whether the survivor still runs after elimination.
cg::Node make_allreduce(cg::TensorId tid, std::string const &name, std::function<void()> exec = {}) {
    cg::Node n;
    n.kind    = cg::OpKind::Allreduce;
    n.label   = "allreduce(" + name + ")";
    n.inputs  = {tid};
    n.outputs = {tid}; // in-place, matches CommunicationInsertion
    cg::CommDescriptor d;
    d.tensor_id  = tid;
    d.size_bytes = 64;
    n.op_data    = d;
    n.execute    = std::move(exec);
    return n;
}

// A plain compute node that writes `tid` (any non-comm kind). Such a write
// invalidates the "already reduced" status of tid.
cg::Node make_writer(cg::TensorId tid) {
    cg::Node n;
    n.kind    = cg::OpKind::Scale;
    n.label   = "writer";
    n.outputs = {tid};
    return n;
}

size_t count_kind(cg::Graph const &g, cg::OpKind kind) {
    size_t c = 0;
    for (auto const &n : g.nodes())
        if (n.kind == kind)
            c++;
    return c;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// CommunicationElimination (rank-independent)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("CommunicationElimination - no-op on a graph with no Allreduce", "[ComputeGraph][Comm][CommunicationElimination]") {
    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);
    auto C = create_zero_tensor<double>("C", 8, 10);

    cg::Graph graph("no-allreduce");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const before = kinds_of(graph);

    cg::passes::CommunicationElimination pass;
    bool const                           modified = pass.run(graph);

    CHECK_FALSE(modified);
    CHECK(pass.num_eliminated() == 0);
    CHECK(kinds_of(graph) == before);
}

TEST_CASE("CommunicationElimination - empty graph is a no-op", "[ComputeGraph][Comm][CommunicationElimination]") {
    cg::Graph                            graph("empty");
    cg::passes::CommunicationElimination pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_eliminated() == 0);
    CHECK(graph.num_nodes() == 0);
}

TEST_CASE("CommunicationElimination - removes an exact-duplicate Allreduce", "[ComputeGraph][Comm][CommunicationElimination]") {
    cg::Graph graph("dup-allreduce");
    graph.nodes().push_back(make_allreduce(7, "C"));
    graph.nodes().push_back(make_allreduce(7, "C")); // redundant: same tensor, unmodified since
    graph.mark_sorted();

    REQUIRE(graph.num_nodes() == 2);

    cg::passes::CommunicationElimination pass;
    bool const                           modified = pass.run(graph);

    CHECK(modified);
    CHECK(pass.num_eliminated() == 1);
    // Exactly one Allreduce survives, and it still targets tensor 7 in-place.
    REQUIRE(count_kind(graph, cg::OpKind::Allreduce) == 1);
    auto const &survivor = graph.nodes().front();
    CHECK(survivor.kind == cg::OpKind::Allreduce);
    CHECK(survivor.inputs == std::vector<cg::TensorId>{7});
    CHECK(survivor.outputs == std::vector<cg::TensorId>{7});
    auto const *desc = std::get_if<cg::CommDescriptor>(&survivor.op_data);
    REQUIRE(desc != nullptr);
    CHECK(desc->tensor_id == 7);
}

TEST_CASE("CommunicationElimination - keeps Allreduces for distinct tensors", "[ComputeGraph][Comm][CommunicationElimination]") {
    cg::Graph graph("distinct-tensors");
    graph.nodes().push_back(make_allreduce(1, "C1"));
    graph.nodes().push_back(make_allreduce(2, "C2"));
    graph.mark_sorted();

    cg::passes::CommunicationElimination pass;
    CHECK_FALSE(pass.run(graph)); // nothing redundant
    CHECK(pass.num_eliminated() == 0);
    CHECK(count_kind(graph, cg::OpKind::Allreduce) == 2);
}

TEST_CASE("CommunicationElimination - a write between two Allreduces prevents elimination",
          "[ComputeGraph][Comm][CommunicationElimination]") {
    cg::Graph graph("write-guards");
    graph.nodes().push_back(make_allreduce(5, "C"));
    graph.nodes().push_back(make_writer(5)); // re-dirties tensor 5
    graph.nodes().push_back(make_allreduce(5, "C"));
    graph.mark_sorted();

    cg::passes::CommunicationElimination pass;
    // The intervening write invalidates the reduced state, so the second
    // Allreduce is genuinely needed and must survive.
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_eliminated() == 0);
    CHECK(count_kind(graph, cg::OpKind::Allreduce) == 2);
    CHECK(graph.num_nodes() == 3);
}

TEST_CASE("CommunicationElimination - a Broadcast between Allreduces does NOT protect the duplicate",
          "[ComputeGraph][Comm][CommunicationElimination]") {
    // Surprising truth: the invalidation loop skips Allreduce/Broadcast/Allgather
    // outputs, so a Broadcast of the same tensor between two Allreduces leaves the
    // "already reduced" status intact and the second Allreduce is still removed.
    cg::Graph graph("broadcast-passthrough");
    graph.nodes().push_back(make_allreduce(9, "C"));
    cg::Node bcast;
    bcast.kind    = cg::OpKind::Broadcast;
    bcast.label   = "broadcast(C)";
    bcast.inputs  = {9};
    bcast.outputs = {9};
    graph.nodes().push_back(std::move(bcast));
    graph.nodes().push_back(make_allreduce(9, "C"));
    graph.mark_sorted();

    cg::passes::CommunicationElimination pass;
    CHECK(pass.run(graph));
    CHECK(pass.num_eliminated() == 1);
    CHECK(count_kind(graph, cg::OpKind::Allreduce) == 1);
    CHECK(count_kind(graph, cg::OpKind::Broadcast) == 1);
}

TEST_CASE("CommunicationElimination - is idempotent", "[ComputeGraph][Comm][CommunicationElimination]") {
    cg::Graph graph("idempotent");
    graph.nodes().push_back(make_allreduce(3, "C"));
    graph.nodes().push_back(make_allreduce(3, "C"));
    graph.nodes().push_back(make_allreduce(3, "C")); // two redundant copies
    graph.mark_sorted();

    cg::passes::CommunicationElimination pass;
    CHECK(pass.run(graph));
    CHECK(pass.num_eliminated() == 2);
    size_t const after_first = graph.num_nodes();
    CHECK(after_first == 1);

    // Second run finds nothing left to remove.
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_eliminated() == 0);
    CHECK(graph.num_nodes() == after_first);
}

TEST_CASE("CommunicationElimination - survivor still executes after the duplicate is dropped",
          "[ComputeGraph][Comm][CommunicationElimination]") {
    // A single-rank allreduce is an identity (nothing to sum across one rank), so
    // dropping the redundant copy is value-preserving. Prove the survivor's
    // executor is intact and runs exactly once.
    int       runs = 0;
    cg::Graph graph("survivor-runs");
    graph.nodes().push_back(make_allreduce(4, "C", [&runs]() { runs++; }));
    graph.nodes().push_back(make_allreduce(4, "C", [&runs]() { runs++; }));
    graph.mark_sorted();

    cg::passes::CommunicationElimination pass;
    REQUIRE(pass.run(graph));
    REQUIRE(count_kind(graph, cg::OpKind::Allreduce) == 1);

    for (auto const &n : graph.nodes())
        if (n.execute)
            n.execute();
    CHECK(runs == 1); // exactly the survivor ran
}

// ═══════════════════════════════════════════════════════════════════════════════
// CommunicationInsertion - single-rank no-op guard
// (world_size() <= 1 short-circuits before any insertion)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("CommunicationInsertion - no-op at single rank", "[ComputeGraph][Comm][CommunicationInsertion]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: insertion may fire; this pins the single-rank guard");

    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);
    auto C = create_zero_tensor<double>("C", 8, 10);

    cg::Graph graph("insertion-noop");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const before = kinds_of(graph);

    cg::passes::CommunicationInsertion pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_inserted() == 0);
    CHECK(kinds_of(graph) == before);

    // Running again changes nothing (no-op is idempotent).
    CHECK_FALSE(pass.run(graph));
    CHECK(kinds_of(graph) == before);
}

TEST_CASE("CommunicationInsertion - no-op on degenerate dim-1 shapes at single rank", "[ComputeGraph][Comm][CommunicationInsertion]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank");

    auto A = create_random_tensor<double>("A", 1, 6); // 1xN
    auto B = create_random_tensor<double>("B", 6, 1); // Nx1
    auto C = create_zero_tensor<double>("C", 1, 1);

    cg::Graph graph("insertion-dim1");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const                         before = kinds_of(graph);
    cg::passes::CommunicationInsertion pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(kinds_of(graph) == before);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CommunicationScheduling - single-rank no-op guard
// The rank guard runs before the Allreduce scan, so even a graph that already
// contains an Allreduce is left untouched (no async split) at single rank.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("CommunicationScheduling - no-op at single rank even with an Allreduce present",
          "[ComputeGraph][Comm][CommunicationScheduling]") {
    if (comm::world_size() > 1)
        SKIP("multi-rank: scheduling may split the Allreduce; this pins the single-rank guard");

    cg::Graph graph("scheduling-noop");
    graph.nodes().push_back(make_allreduce(2, "C"));
    graph.mark_sorted();

    auto const before = kinds_of(graph);

    cg::passes::CommunicationScheduling pass;
    CHECK_FALSE(pass.run(graph));
    CHECK(pass.num_scheduled() == 0);
    CHECK(kinds_of(graph) == before);

    // The Allreduce keeps its synchronous form: no async phases were installed.
    auto const &ar = graph.nodes().front();
    CHECK_FALSE(static_cast<bool>(ar.async_start));
    CHECK_FALSE(static_cast<bool>(ar.async_finish));
}
