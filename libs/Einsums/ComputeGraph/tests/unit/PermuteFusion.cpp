//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PermuteFusion.cpp
/// @brief Tests for the PermuteFusion optimization pass (absorb Permute into Einsum subscript).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

// ── Helpers ─────────────────────────────────────────────────────────────────
namespace {

constexpr double kTol = 1e-12;

template <size_t Rank>
void require_close(Tensor<double, Rank> const &got, Tensor<double, Rank> const &ref) {
    REQUIRE(got.size() == ref.size());
    double const *g = got.data();
    double const *r = ref.data();
    for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(std::abs(g[i] - r[i]) < kTol);
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Basic rewrites
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("PermuteFusion: 2D transpose absorbed into einsum A slot", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Before:  A_T = A^T,  C = einsum("ji;jk->ik", A_T, B)
    // After:   C = einsum("ij;jk->ik", A, B), permute removed.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    // Reference: the fused (direct) computation.
    auto C_ref = create_zero_tensor<double>("C_ref", 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", &C_ref, A, B);
        g.execute();
    }

    // Under test: the unfused graph (permute then einsum).
    auto      A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto      C   = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("fusion_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }
    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_candidates() == 1);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    // After fusion the lone node is the einsum and its A subscript is ["i","j"].
    auto const &node = graph.nodes()[0];
    REQUIRE(node.kind == cg::OpKind::Einsum);
    auto *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->indices != nullptr);
    REQUIRE(desc->indices->a_indices == std::vector<std::string>{"i", "j"});
    REQUIRE(desc->indices->b_indices == std::vector<std::string>{"j", "k"});
    REQUIRE(desc->indices->c_indices == std::vector<std::string>{"i", "k"});

    graph.execute();
    require_close(C, C_ref);
}

TEST_CASE("PermuteFusion: 2D transpose absorbed into einsum B slot", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Before:  B_T = B^T,  C = einsum("ij;kj->ik", A, B_T)  (B_T has the contracted index j as its second axis)
    // After:   C = einsum("ij;jk->ik", A, B)
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", &C_ref, A, B);
        g.execute();
    }

    auto      B_T = create_zero_tensor<double>("B_T", 5, 4);
    auto      C   = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("fusion_b_slot");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("kj <- jk", &B_T, B);
        cg::einsum("ij;kj->ik", &C, A, B_T);
    }
    REQUIRE(graph.num_nodes() == 2);

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    auto const &node = graph.nodes()[0];
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc->indices->b_indices == std::vector<std::string>{"j", "k"});

    graph.execute();
    require_close(C, C_ref);
}

TEST_CASE("PermuteFusion: both inputs permuted in same einsum", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Both A and B come through separate permutes, pass should fuse both in one run.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", &C_ref, A, B);
        g.execute();
    }

    auto      A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto      B_T = create_zero_tensor<double>("B_T", 5, 4);
    auto      C   = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("both");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::permute("kj <- jk", &B_T, B);
        cg::einsum("ji;kj->ik", &C, A_T, B_T);
    }
    REQUIRE(graph.num_nodes() == 3);

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 2);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();
    require_close(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Higher rank
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("PermuteFusion: 3D permute on A slot (rank-3 × matrix)", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // T has axes (p, q, r); einsum uses its axes in the order p, q, r.
    // Permute to (r, p, q), then einsum reads "rpq;rs->pqs".
    // Fused: einsum reads T directly with "pqr;rs->pqs".
    auto T = create_random_tensor<double>("T", 2, 3, 4);
    auto M = create_random_tensor<double>("M", 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 2, 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("pqr;rs->pqs", &C_ref, T, M);
        g.execute();
    }

    auto      T_perm = create_zero_tensor<double>("T_perm", 4, 2, 3);
    auto      C      = create_zero_tensor<double>("C", 2, 3, 5);
    cg::Graph graph("rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("rpq <- pqr", &T_perm, T);
        cg::einsum("rpq;rs->pqs", &C, T_perm, M);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    auto const *desc = std::get_if<cg::EinsumDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(desc->indices->a_indices == std::vector<std::string>{"p", "q", "r"});

    graph.execute();
    require_close(C, C_ref);
}

TEST_CASE("PermuteFusion: identity permute is still fused (no-op removal)", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Permute with identical input/output indices is a plain copy, the
    // subscript rewrite is a no-op but the permute node should still be
    // removed, so the einsum can run directly on the original tensor.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", &C_ref, A, B);
        g.execute();
    }

    auto      A_copy = create_zero_tensor<double>("A_copy", 3, 4);
    auto      C      = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("identity");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", &A_copy, A); // NOLINT
        cg::einsum("ij;jk->ik", &C, A_copy, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    auto const *desc = std::get_if<cg::EinsumDescriptor>(&graph.nodes()[0].op_data);
    // Identity permute → subscript unchanged.
    REQUIRE(desc->indices->a_indices == std::vector<std::string>{"i", "j"});

    graph.execute();
    require_close(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Safety: skip when fusion would break correctness
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("PermuteFusion: skip when permute output has multiple consumers", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // A_T is consumed by TWO einsums, removing the permute would leave
    // the second one dangling. Pass should report the candidate but
    // not rewrite.
    auto A  = create_random_tensor<double>("A", 3, 4);
    auto B1 = create_random_tensor<double>("B1", 4, 5);
    auto B2 = create_random_tensor<double>("B2", 4, 2);

    auto      A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto      C1  = create_zero_tensor<double>("C1", 3, 5);
    auto      C2  = create_zero_tensor<double>("C2", 3, 2);
    cg::Graph graph("multi_consumer");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C1, A_T, B1);
        cg::einsum("ji;jm->im", &C2, A_T, B2);
    }
    REQUIRE(graph.num_nodes() == 3);

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE_FALSE(modified);
    // Candidates: two einsums each find the permute as their A producer.
    REQUIRE(pass.num_candidates() == 2);
    REQUIRE(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 3);
}

TEST_CASE("PermuteFusion: skip when permute has non-trivial alpha", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // permute with alpha != 1 scales values, not a pure axis reorder.
    // Must not fuse.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    auto      A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto      C   = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("scaled_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &A_T, 2.5, A); // beta=0, alpha=2.5
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_candidates() == 1);
    REQUIRE(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("PermuteFusion: skip when permute accumulates (non-zero beta)", "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);

    auto      A_T = create_random_tensor<double>("A_T", 4, 3); // pre-filled; beta=0.5 means A_T += 0.5*perm(A)
    auto      C   = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("accum_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.5, &A_T, 1.0, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 2);
}

TEST_CASE("PermuteFusion: no candidates when no permute feeds einsum", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Straight einsum with no preceding permute, nothing to do.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 3, 5);

    cg::Graph graph("no_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij;jk->ik", &C, A, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_candidates() == 0);
    REQUIRE(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Re-execution: mutable-indices invariant
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("PermuteFusion: fused graph is replayable", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // After fusion, re-executing the graph with different input values
    // must produce the correct result, confirms the shared indices
    // the executor reads are consistent across calls.
    auto A   = create_random_tensor<double>("A", 3, 4);
    auto B   = create_random_tensor<double>("B", 4, 5);
    auto A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto C   = create_zero_tensor<double>("C", 3, 5);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }
    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);

    graph.execute();
    auto C_snapshot = Tensor<double, 2>(C);

    // Rerun: C should be recomputed to the same values.
    C.zero();
    graph.execute();
    require_close(C, C_snapshot);
}
