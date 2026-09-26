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
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;
using einsums::testing::reference_permute;

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
    auto      C = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("fusion_test");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
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
    auto *desc = node.op_data.get_if<cg::EinsumDescriptor>();
    REQUIRE(desc != nullptr);
    REQUIRE(desc->indices != nullptr);
    REQUIRE(desc->indices->spec.a_indices == std::vector<std::string>{"i", "j"});
    REQUIRE(desc->indices->spec.b_indices == std::vector<std::string>{"j", "k"});
    REQUIRE(desc->indices->spec.c_indices == std::vector<std::string>{"i", "k"});

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
    auto      C = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("fusion_b_slot");
    auto     &B_T = graph.scratch<double, 2>("B_T", 5, 4);
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
    auto const *desc = node.op_data.get_if<cg::EinsumDescriptor>();
    REQUIRE(desc->indices->spec.b_indices == std::vector<std::string>{"j", "k"});

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
    auto      C = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("both");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
    auto     &B_T = graph.scratch<double, 2>("B_T", 5, 4);
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
    auto      C = create_zero_tensor<double>("C", 2, 3, 5);
    cg::Graph graph("rank3");
    auto     &T_perm = graph.scratch<double, 3>("T_perm", 4, 2, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("rpq <- pqr", &T_perm, T);
        cg::einsum("rpq;rs->pqs", &C, T_perm, M);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    auto const *desc = graph.nodes()[0].op_data.get_if<cg::EinsumDescriptor>();
    REQUIRE(desc->indices->spec.a_indices == std::vector<std::string>{"p", "q", "r"});

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
    auto      C = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("identity");
    auto     &A_copy = graph.scratch<double, 2>("A_copy", 3, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", &A_copy, A); // NOLINT
        cg::einsum("ij;jk->ik", &C, A_copy, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(pass.num_rewrites() == 1);
    REQUIRE(graph.num_nodes() == 1);

    auto const *desc = graph.nodes()[0].op_data.get_if<cg::EinsumDescriptor>();
    // Identity permute → subscript unchanged.
    REQUIRE(desc->indices->spec.a_indices == std::vector<std::string>{"i", "j"});

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
    auto      A  = create_random_tensor<double>("A", 3, 4);
    auto      B1 = create_random_tensor<double>("B1", 4, 5);
    auto      B2 = create_random_tensor<double>("B2", 4, 2);
    auto      C1 = create_zero_tensor<double>("C1", 3, 5);
    auto      C2 = create_zero_tensor<double>("C2", 3, 2);
    cg::Graph graph("multi_consumer");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
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
    auto      A = create_random_tensor<double>("A", 3, 4);
    auto      B = create_random_tensor<double>("B", 4, 5);
    auto      C = create_zero_tensor<double>("C", 3, 5);
    cg::Graph graph("scaled_permute");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
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

TEST_CASE("PermuteFusion: skip when a complex permute has an imaginary alpha", "[ComputeGraph][Optimizer][PermuteFusion][Complex]") {
    // Regression: PermuteDescriptor stored alpha/beta as `double` and capture
    // filled them with alpha.real(), so alpha = 1+3i recorded as 1.0. can_fuse
    // then read a pure axis reorder and absorbed the permute into the einsum's
    // subscript, silently dropping the 3i.
    using Complex = std::complex<double>;
    auto A        = create_random_tensor<Complex>("A", 3, 4);
    auto B        = create_random_tensor<Complex>("B", 4, 5);

    auto      A_T = create_zero_tensor<Complex>("A_T", 4, 3);
    auto      C   = create_zero_tensor<Complex>("C", 3, 5);
    cg::Graph graph("complex_alpha_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", Complex{0.0, 0.0}, &A_T, Complex{1.0, 3.0}, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE_FALSE(modified);
    REQUIRE(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 2);

    graph.execute();

    // Reference: C = ((1+3i) * A^T)^T-contracted with B, i.e. (1+3i)*(A·B).
    auto C_ref = create_zero_tensor<Complex>("C_ref", 3, 5);
    {
        cg::Graph              g("ref_complex");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", 0.0, &C_ref, Complex{1.0, 3.0}, A, B);
        g.execute();
    }
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < kTol);
        }
    }
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
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 3, 5);

    cg::Graph graph("replay");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
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

TEST_CASE("PermuteFusion: fused einsum output feeding a downstream consumer stays correct", "[ComputeGraph][Optimizer][PermuteFusion]") {
    // Consumer-bearing topology (node-position hazard): after the permute is
    // absorbed, the rewritten einsum must remain positioned before the gemm
    // that reads its output.
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 3);
    auto E = create_random_tensor<double>("E", 3, 5);

    // Reference: direct computation, no permute.
    auto C_ref = create_zero_tensor<double>("C_ref", 3, 3);
    auto D_ref = create_zero_tensor<double>("D_ref", 3, 5);
    {
        cg::Graph              g("ref");
        cg::CaptureGuard const guard(g);
        cg::einsum("ij;jk->ik", &C_ref, A, B);
        cg::einsum("ij;jk->ik", &D_ref, C_ref, E);
        g.execute();
    }
    auto      C = create_zero_tensor<double>("C", 3, 3);
    auto      D = create_zero_tensor<double>("D", 3, 5);
    cg::Graph graph("fusion_consumer");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
        cg::einsum("ij;jk->ik", &D, C, E); // consumer of the fused einsum's output
    }
    REQUIRE(graph.num_nodes() == 3);

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 2);

    graph.execute();
    require_close(C, C_ref);
    require_close(D, D_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Cases the fusion must decline
// ═══════════════════════════════════════════════════════════════════════════

// A permute carrying P(ij) is the antisymmetrizer W = A^T - A, not a relabeling.
// Fusing it into its one reader dropped the operator, so the optimized graph
// computed A @ B from the plain transpose instead.
TEST_CASE("PermuteFusion: skip a permute that carries a permutation operator", "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 4);

    cg::Graph graph("antisymmetrizer");
    auto     &W = graph.scratch<double, 2>("W", 3, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("i,j <- P(ij) j,i", &W, A);
        cg::einsum("ik <- ij ; jk", &C, W, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    CHECK_FALSE(modified);
    CHECK(pass.num_rewrites() == 0);
    REQUIRE(graph.num_nodes() == 2);

    // The skipped permute's scratch output is still deferred; allocate it before running.
    graph.apply<cg::passes::Materialization>();
    graph.execute();
    // W = P(ij) A^T = A^T - A. Spelled out because the reference helpers take no operators.
    auto W_ref = create_zero_tensor<double>("W_ref", 3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            W_ref(i, j) = A(j, i) - A(i, j);
        }
    }
    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4);
    einsums::testing::reference_einsum("ik <- ij ; jk", 0.0, &C_ref, 1.0, W_ref, B);
    require_close(C, C_ref);
}

// Fusing removes the only write to the permute's output. A tensor the caller
// created is one the caller can read after execute(), so it has to keep its
// permute; the pass used to fuse it and leave it holding zeros.
TEST_CASE("PermuteFusion: skip when the caller owns the permute output", "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto A   = create_random_tensor<double>("A", 3, 4);
    auto B   = create_random_tensor<double>("B", 4, 5);
    auto A_T = create_zero_tensor<double>("A_T", 4, 3);
    auto C   = create_zero_tensor<double>("C", 3, 5);

    cg::Graph graph("caller_owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    CHECK_FALSE(modified);
    REQUIRE(graph.num_nodes() == 2);

    // The skipped permute's scratch output is still deferred; allocate it before running.
    graph.apply<cg::passes::Materialization>();
    graph.execute();
    auto A_T_ref = create_zero_tensor<double>("A_T_ref", 4, 3);
    einsums::testing::reference_permute("ji <- ij", 0.0, &A_T_ref, 1.0, A);
    require_close(A_T, A_T_ref);
}

// A Loop node does not list its body's reads, so from the parent's node list the
// permute output has one consumer. Fusing it would leave the body reading a
// tensor nothing writes.
TEST_CASE("PermuteFusion: skip when a loop body reads the permute output", "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 3, 5);
    auto D = create_zero_tensor<double>("D", 3, 5);

    cg::Graph graph("loop_reader");
    auto     &A_T = graph.scratch<double, 2>("A_T", 4, 3);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &A_T, A);
        cg::einsum("ji;jk->ik", &C, A_T, B);
    }
    auto &body = graph.add_loop("once", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const body_guard(body);
        cg::einsum("ji;jk->ik", &D, A_T, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    CHECK_FALSE(modified);

    // The skipped permute's scratch output is still deferred; allocate it before running.
    graph.apply<cg::passes::Materialization>();
    graph.execute();
    auto D_ref = create_zero_tensor<double>("D_ref", 3, 5);
    einsums::testing::reference_einsum("ik <- ij ; jk", 0.0, &D_ref, 1.0, A, B);
    require_close(D, D_ref);
}

// ═══════════════════════════════════════════════════════════════════════════
// Soundness of the source and the redirect
// ═══════════════════════════════════════════════════════════════════════════

// Defends: the fused einsum reads the permute's source where the EINSUM stands, not where the
// permute stood. A write to the source between the two used to go unnoticed, so C came out as
// (X Y)^T B instead of A^T B; the fuzzers reached it through Reorder moving an unrelated writer of
// A into the gap, and it survived O1, O2 and the default pipeline.
TEST_CASE("PermuteFusion: skip when the source is written between the permute and its consumer",
          "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto                    A     = create_random_tensor<double>("A", 4, 4);
    auto                    X     = create_random_tensor<double>("X", 4, 3);
    auto                    Y     = create_random_tensor<double>("Y", 3, 4);
    auto                    B     = create_random_tensor<double>("B", 4, 2);
    auto                    C     = create_zero_tensor<double>("C", 4, 2);
    Tensor<double, 2> const A_old = A;

    cg::Graph graph("pf_source_written");
    auto     &S = graph.create_zero_tensor<double, 2>("S", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &S, A);
        cg::einsum("ik;kj->ij", &A, X, Y);
        cg::einsum("ik;kj->ij", &C, S, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    CHECK_FALSE(modified);
    CHECK(pass.num_rewrites() == 0);
    graph.execute();

    auto S_ref = create_zero_tensor<double>("S_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 2);
    reference_permute("ji <- ij", 0.0, &S_ref, 1.0, A_old);
    reference_einsum("ij <- ik ; kj", 0.0, &C_ref, 1.0, S_ref, B);
    require_close(C, C_ref);
}

// Defends: fusing redirects the scratch's slot at the source, which moves EVERY write through that
// slot into the caller's tensor. An earlier writer of the scratch therefore wrote X Y into A.
TEST_CASE("PermuteFusion: skip when something else writes the permuted scratch", "[ComputeGraph][Optimizer][PermuteFusion]") {
    auto                    A     = create_random_tensor<double>("A", 4, 4);
    auto                    X     = create_random_tensor<double>("X", 4, 3);
    auto                    Y     = create_random_tensor<double>("Y", 3, 4);
    auto                    B     = create_random_tensor<double>("B", 4, 2);
    auto                    C     = create_zero_tensor<double>("C", 4, 2);
    Tensor<double, 2> const A_old = A;

    cg::Graph graph("pf_second_writer");
    auto     &S = graph.create_zero_tensor<double, 2>("S", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &S, X, Y);
        cg::permute("ji <- ij", &S, A);
        cg::einsum("ik;kj->ij", &C, S, B);
    }

    auto [modified, pass] = graph.apply<cg::passes::PermuteFusion>();
    CHECK_FALSE(modified);
    graph.execute();

    require_close(A, A_old);
    auto S_ref = create_zero_tensor<double>("S_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 2);
    reference_permute("ji <- ij", 0.0, &S_ref, 1.0, A_old);
    reference_einsum("ij <- ik ; kj", 0.0, &C_ref, 1.0, S_ref, B);
    require_close(C, C_ref);
}

// Defends: the same hole reached through CSE, which is how O1 found it. CSE merges S2 into S1 by
// redirecting S2's slot, so the node reading S2 still names S2 and a reader count by id sees
// nothing else reading S1. By buffer, S1 has two readers and two writers and is left alone.
TEST_CASE("PermuteFusion: a reader merged in by CSE still counts", "[ComputeGraph][Optimizer][PermuteFusion][CSE]") {
    auto                    A     = create_random_tensor<double>("A", 4, 4);
    auto                    X     = create_random_tensor<double>("X", 4, 3);
    auto                    Y     = create_random_tensor<double>("Y", 3, 4);
    auto                    B     = create_random_tensor<double>("B", 4, 2);
    auto                    C     = create_zero_tensor<double>("C", 4, 2);
    auto                    R     = create_random_tensor<double>("R", 4, 4);
    Tensor<double, 2> const A_old = A;
    Tensor<double, 2> const R_old = R;

    cg::Graph graph("pf_cse_reader");
    auto     &S1 = graph.create_zero_tensor<double, 2>("S1", 4, 4);
    auto     &S2 = graph.create_zero_tensor<double, 2>("S2", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &S1, X, Y);
        cg::einsum("ik;kj->ij", &S2, X, Y);
        cg::axpy(1.0, S2, &R);
        cg::permute("ji <- ij", &S1, A);
        cg::einsum("ik;kj->ij", &C, S1, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::CSE>();
    pm.add<cg::passes::PermuteFusion>();
    graph.apply(pm);
    INFO(pm.explain());
    graph.execute();

    require_close(A, A_old);
    auto XY    = create_zero_tensor<double>("XY", 4, 4);
    auto S_ref = create_zero_tensor<double>("S_ref", 4, 4);
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 2);
    reference_einsum("ij <- ik ; kj", 0.0, &XY, 1.0, X, Y);
    reference_permute("ji <- ij", 0.0, &S_ref, 1.0, A_old);
    reference_einsum("ij <- ik ; kj", 0.0, &C_ref, 1.0, S_ref, B);
    auto R_ref = R_old;
    reference_permute("ij <- ij", 1.0, &R_ref, 1.0, XY);
    require_close(C, C_ref);
    require_close(R, R_ref);
}
