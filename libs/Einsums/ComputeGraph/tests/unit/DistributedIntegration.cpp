//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DistributedIntegration.cpp
/// @brief Integration tests for distributed ComputeGraph execution under MPI.
///
/// These tests verify that a captured einsum graph produces correct numerical
/// results when executed on multiple MPI ranks. They work with both the mock
/// backend (1 rank) and real MPI (mpirun -np N).

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/DistributionDescriptor.hpp>
#include <Einsums/Comm/ProcessGrid.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstddef>
#include <utility>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: compare two tensors element-wise
// ═══════════════════════════════════════════════════════════════════════════════

namespace {
template <typename T, size_t R>
void check_tensors_equal(Tensor<T, R> const &actual, Tensor<T, R> const &expected, double margin = 1e-10) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); i++) {
        CHECK(actual.data()[i] == Catch::Approx(expected.data()[i]).margin(margin));
    }
}

/// Compute balanced blocking start for position `pos` of `nprocs` splitting `N` elements.
/// Matches DistributionDescriptor::local_range() formula.
size_t balanced_start(size_t N, int nprocs, int pos) {
    auto const   P         = static_cast<size_t>(nprocs);
    size_t const base      = N / P;
    size_t const remainder = N % P;
    if (std::cmp_less(pos, remainder))
        return static_cast<size_t>(pos) * (base + 1);
    else
        return remainder * (base + 1) + (static_cast<size_t>(pos) - remainder) * base;
}
} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Simple GEMM through ComputeGraph (no deferred allocation)
// Verifies basic graph capture + optimize + execute matches direct computation.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - GEMM via graph matches direct", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 20, 10);
    auto B = create_random_tensor<double>("B", 10, 15);
    auto C = create_zero_tensor<double>("C", 20, 15);

    // Reference: direct einsum
    auto C_ref = create_zero_tensor<double>("C_ref", 20, 15);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    // Graph: capture, optimize, execute
    cg::Graph graph("dist_gemm");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    // Every rank should get the correct result
    check_tensors_equal(C, C_ref);

    // Log rank info for multi-rank runs
    EINSUMS_LOG_INFO("Rank {}/{}: GEMM via graph test passed", comm::world_rank(), comm::world_size());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: GEMM with deferred allocation (Workspace pattern)
// Tests that Materialization + DistributionPlanning correctly handle
// deferred tensors on all ranks.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - deferred GEMM with Workspace", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 8, 6);
    auto B = create_random_tensor<double>("B", 6, 10);

    // Reference
    auto C_ref = create_zero_tensor<double>("C_ref", 8, 10);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    // Graph with deferred C
    cg::Graph graph("dist_deferred");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), 8, 10);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    REQUIRE(C.is_materialized());
    check_tensors_equal(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Chain of two GEMMs
// Tests that multi-step computation produces consistent results across ranks.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - two-GEMM chain", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 12, 8);
    auto B = create_random_tensor<double>("B", 8, 6);
    auto D = create_random_tensor<double>("D", 6, 10);

    // Reference
    auto T_ref = create_zero_tensor<double>("T_ref", 12, 6);
    auto C_ref = create_zero_tensor<double>("C_ref", 12, 10);
    tensor_algebra::einsum(Indices{i, j}, &T_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, T_ref, Indices{k, j}, D);

    // Graph
    auto T = create_zero_tensor<double>("T", 12, 6);
    auto C = create_zero_tensor<double>("C", 12, 10);

    cg::Graph graph("dist_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
        cg::einsum("ik;kj->ij", &C, T, D);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    check_tensors_equal(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Scale + AXPY (non-GEMM operations)
// Verifies that element-wise operations work correctly across ranks.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - scale and axpy", "[ComputeGraph][Distributed]") {
    auto X = create_random_tensor<double>("X", 5, 5);
    auto Y = create_random_tensor<double>("Y", 5, 5);

    // Reference
    auto X_ref = Tensor<double, 2>(X);
    auto Y_ref = Tensor<double, 2>(Y);
    linear_algebra::scale(2.0, &X_ref);
    linear_algebra::axpy(3.0, X_ref, &Y_ref);

    // Graph
    cg::Graph graph("dist_scale_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &X);
        cg::axpy(3.0, X, &Y);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    check_tensors_equal(X, X_ref);
    check_tensors_equal(Y, Y_ref);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: Verify all ranks produce identical results
// This is the key correctness property: with the current implementation
// (no actual distribution), all ranks should compute identical results.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - all ranks agree on result", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 10, 8);
    auto B = create_random_tensor<double>("B", 8, 12);
    auto C = create_zero_tensor<double>("C", 10, 12);

    // Broadcast A and B from rank 0 so all ranks start with identical data
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT

    cg::Graph graph("dist_agree");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    // Allreduce C to check agreement: if all ranks have the same C,
    // then sum(C) / nprocs == C on each rank.
    int const           nprocs = comm::world_size();
    std::vector<double> local_sum(C.size());
    std::vector<double> global_sum(C.size());
    std::memcpy(local_sum.data(), C.data(), C.size() * sizeof(double));

    auto placeholder = comm::allreduce<double>(local_sum, global_sum, comm::ReduceOp::Sum);
    (void)placeholder;

    for (size_t idx = 0; idx < C.size(); idx++) {
        CHECK(global_sum[idx] / nprocs == Catch::Approx(C.data()[idx]).margin(1e-10));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 6: Float precision chain
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - float GEMM chain", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<float>("A", 16, 10);
    auto B = create_random_tensor<float>("B", 10, 12);
    auto D = create_random_tensor<float>("D", 12, 8);

    // Reference
    auto T_ref = create_zero_tensor<float>("T_ref", 16, 12);
    auto C_ref = create_zero_tensor<float>("C_ref", 16, 8);
    tensor_algebra::einsum(0.0f, Indices{i, j}, &T_ref, 1.0f, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(0.0f, Indices{i, j}, &C_ref, 1.0f, Indices{i, k}, T_ref, Indices{k, j}, D);

    // Graph
    auto T = create_zero_tensor<float>("T", 16, 12);
    auto C = create_zero_tensor<float>("C", 16, 8);

    cg::Graph graph("dist_float_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0f, &T, 1.0f, A, B);
        cg::einsum("ik;kj->ij", 0.0f, &C, 1.0f, T, D);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    check_tensors_equal(C, C_ref, 1e-4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 7: Rank-3 contraction
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - rank-3 contraction", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 4, 5, 3);
    auto B = create_random_tensor<double>("B", 3, 6);
    auto C = create_zero_tensor<double>("C", 4, 5, 6);

    // Reference: use p,q,r,s indices for rank-3
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 5, 6);
    tensor_algebra::einsum(Indices{p, q, s}, &C_ref, Indices{p, q, r}, A, Indices{r, s}, B);

    // Graph
    cg::Graph graph("dist_rank3");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pqr;rs->pqs", &C, A, B);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    check_tensors_equal(C, C_ref);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 8: Optimizer pass pipeline stats
// Verify the optimization passes run and report meaningful stats on all ranks.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - optimizer stats consistent across ranks", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 10, 8);
    auto B = create_random_tensor<double>("B", 8, 12);
    auto C = create_zero_tensor<double>("C", 10, 12);

    cg::Graph graph("dist_stats");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    size_t const nodes_before = graph.num_nodes();
    auto         pm           = cg::PassManager::create_default();
    graph.apply(pm);
    size_t const nodes_after = graph.num_nodes();

    // All ranks should see the same node counts (same graph, same passes)
    std::vector<int> local_counts = {static_cast<int>(nodes_before), static_cast<int>(nodes_after)};
    std::vector<int> global_counts(static_cast<long>(2) * comm::world_size());

    (void)comm::allgather<int>(local_counts, global_counts); // NOLINT

    // Check all ranks have the same counts
    for (int r = 0; r < comm::world_size(); r++) {
        CHECK(std::cmp_equal(global_counts[static_cast<long>(r * 2)], nodes_before));
        CHECK(std::cmp_equal(global_counts[r * 2 + 1], nodes_after));
    }

    graph.execute();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 9: Force distribution with low threshold
// This is the key test: with a low threshold, DistributionPlanning marks
// deferred tensors as distributed, Materialization allocates local partitions,
// and CommunicationInsertion adds allreduce where needed.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - embarrassingly parallel GEMM", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    // Use dimensions divisible by common rank counts (1, 2, 4)
    constexpr size_t M = 40, K = 20, N = 32;

    // Create full data on rank 0, broadcast to all
    auto A_full = create_random_tensor<double>("A_full", M, K);
    auto B      = create_random_tensor<double>("B", K, N);
    (void)comm::broadcast<double>(std::span<double>(A_full.data(), A_full.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0);           // NOLINT

    // Reference: direct einsum (full size)
    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A_full, Indices{k, j}, B);

    // Graph: both A and C are deferred so they can both be distributed along dim 0.
    // B is pre-allocated and replicated (every rank has the full B).
    cg::Graph graph("dist_ep");
    auto     &A = graph.declare_tensor<double, 2>(std::string("A"), M, K);
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);

    // Custom init for A: after materialization (which allocates local partition),
    // copy this rank's slice from A_full.
    for (auto &[tid, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == &A) {
            handle.init_kind = cg::InitKind::Zero; // Triggers an Initialize node
            handle.zero_fn   = [&A, &A_full, M, K]() {
                // A is distributed along Row (dim 0), use grid coordinates
                auto        &grid      = comm::ProcessGrid::default_grid();
                size_t const local_m   = A.dim(0);
                size_t const start_row = balanced_start(M, grid.rows(), grid.my_row());
                for (size_t ii = 0; ii < local_m; ii++) {
                    for (size_t jj = 0; jj < K; jj++) {
                        A(ii, jj) = A_full(start_row + ii, jj);
                    }
                }
            };
            break;
        }
    }

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Custom pass pipeline: outer-product only (no SUMMA), low threshold
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());
    REQUIRE(A.is_materialized());

    if (nprocs == 1) {
        CHECK(C.dim(0) == M);
        CHECK(C.dim(1) == N);
        check_tensors_equal(C, C_ref);
    } else {
        // 2D grid: C is [Row,Col], A is [Row,None]
        auto &grid   = comm::ProcessGrid::default_grid();
        int   my_row = grid.my_row();
        int   my_col = grid.my_col();
        int   pr     = grid.rows();
        int   pc     = grid.cols();

        size_t const expected_m = (M + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
        size_t const expected_n = (N + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);

        CHECK(C.dim(0) <= expected_m);
        CHECK(C.dim(0) > 0);
        CHECK(C.dim(1) <= expected_n);
        CHECK(C.dim(1) > 0);
        CHECK(A.dim(0) == C.dim(0)); // A shares Row distribution with C
        CHECK(A.dim(1) == K);        // A's K dimension is not distributed

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) in {}x{} grid: A local = {}x{}, C local = {}x{}", comm::world_rank(), nprocs, my_row, my_col,
                         pr, pc, A.dim(0), A.dim(1), C.dim(0), C.dim(1));

        // Verify local C against reference using grid coordinates
        size_t const row_start = balanced_start(M, pr, my_row);
        size_t const col_start = balanced_start(N, pc, my_col);
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 10: Allreduce case, distribute along link index K
// Each rank computes a partial sum C_partial = A_local * B_local,
// then allreduce sums partial Cs to get the correct full result.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - SUMMA GEMM on 2D grid", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    // SUMMA requires a square grid (Pr == Pc). Skip on non-square.
    if (nprocs > 1) {
        auto &g = comm::ProcessGrid::default_grid();
        if (g.rows() != g.cols()) {
            SKIP("SUMMA requires square grid, got " + std::to_string(g.rows()) + "x" + std::to_string(g.cols()));
        }
    }

    // All three tensors deferred and fully 2D-distributed via SUMMA.
    // A[i,k] → [Row,Col], B[k,j] → [Row,Col], C[i,j] → [Row,Col]
    // SUMMA broadcasts A panels along rows, B panels along cols.
    constexpr size_t M = 20, K = 20, N = 20; // Square for easy division on 2x2 grid

    // Full data on all ranks for reference computation
    auto A_full = create_random_tensor<double>("A_full", M, K);
    auto B_full = create_random_tensor<double>("B_full", K, N);
    (void)comm::broadcast<double>(std::span<double>(A_full.data(), A_full.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B_full.data(), B_full.size()), 0); // NOLINT

    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A_full, Indices{k, j}, B_full);

    // All three tensors are deferred
    cg::Graph graph("dist_summa");
    auto     &A = graph.declare_tensor<double, 2>(std::string("A"), M, K);
    auto     &B = graph.declare_tensor<double, 2>(std::string("B"), K, N);
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);

    // Custom init for A and B: fill each rank's local partition from full data
    // After SUMMA distribution: A is [Row,Col] = (M/Pr, K/Pc), B is [Row,Col] = (K/Pr, N/Pc)
    for (auto &[tid, handle] : graph.tensors_map()) {
        if (handle.tensor_ptr == &A) {
            handle.init_kind = cg::InitKind::Zero;
            handle.zero_fn   = [&A, &A_full, M, K]() {
                auto        &grid = comm::ProcessGrid::default_grid();
                int const    pr = grid.rows(), pc = grid.cols();
                size_t const row_chunk = (M + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
                size_t const col_chunk = (K + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);
                size_t const row_start = balanced_start(M, pr, grid.my_row());
                size_t const col_start = balanced_start(K, pc, grid.my_col());
                for (size_t ii = 0; ii < A.dim(0); ii++)
                    for (size_t jj = 0; jj < A.dim(1); jj++)
                        A(ii, jj) = A_full(row_start + ii, col_start + jj);
            };
        } else if (handle.tensor_ptr == &B) {
            handle.init_kind = cg::InitKind::Zero;
            handle.zero_fn   = [&B, &B_full, K, N]() {
                auto        &grid = comm::ProcessGrid::default_grid();
                int const    pr = grid.rows(), pc = grid.cols();
                size_t const row_chunk = (K + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
                size_t const col_chunk = (N + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);
                size_t const row_start = balanced_start(K, pr, grid.my_row());
                size_t const col_start = balanced_start(N, pc, grid.my_col());
                for (size_t ii = 0; ii < B.dim(0); ii++)
                    for (size_t jj = 0; jj < B.dim(1); jj++)
                        B(ii, jj) = B_full(row_start + ii, col_start + jj);
            };
        }
    }

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Full pass pipeline with SUMMA
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::SUMMAExpansion>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(A.is_materialized());
    REQUIRE(B.is_materialized());
    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        int   pr = grid.rows(), pc = grid.cols();

        // On a square 2x2 grid, SUMMA gives each rank (M/2, N/2) of C
        size_t const row_chunk = (M + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
        size_t const col_chunk = (N + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);

        CHECK(C.dim(0) <= row_chunk);
        CHECK(C.dim(1) <= col_chunk);

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: A={}x{}, B={}x{}, C={}x{}", comm::world_rank(), nprocs, grid.my_row(),
                         grid.my_col(), pr, pc, A.dim(0), A.dim(1), B.dim(0), B.dim(1), C.dim(0), C.dim(1));

        // Verify local C against reference
        size_t const row_start = balanced_start(M, pr, grid.my_row());
        size_t const col_start = balanced_start(N, pc, grid.my_col());
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 11: Automatic input slicing
// Pre-allocated A and B with deferred distributed C. The InputSlicing pass
// should automatically create views of A for each rank's local rows,
// so no manual init function is needed.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - automatic input slicing", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    // M is largest dim of C, so C distributes along dim 0 (rows)
    constexpr size_t M = 40, K = 20, N = 32;

    // Pre-allocated inputs (replicated on all ranks, NOT deferred)
    auto A = create_random_tensor<double>("A", M, K);
    auto B = create_random_tensor<double>("B", K, N);
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT

    // Reference
    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    // Only C is deferred. A and B are pre-allocated
    cg::Graph graph("dist_autoslice");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Custom pipeline: outer-product (no SUMMA), low threshold
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        CHECK(C.dim(0) == M);
        check_tensors_equal(C, C_ref);
    } else {
        // 2D grid: C is [Row,Col], A and B are pre-allocated (auto-sliced)
        auto &grid   = comm::ProcessGrid::default_grid();
        int   my_row = grid.my_row();
        int   my_col = grid.my_col();
        int   pr     = grid.rows();
        int   pc     = grid.cols();

        size_t const expected_m = (M + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
        size_t const expected_n = (N + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);

        CHECK(C.dim(0) <= expected_m);
        CHECK(C.dim(0) > 0);
        CHECK(C.dim(1) <= expected_n);

        // A and B should be restored to full size after the einsum
        CHECK(A.dim(0) == M);
        CHECK(A.dim(1) == K);
        CHECK(B.dim(0) == K);
        CHECK(B.dim(1) == N);

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: C local = {}x{}, A = {}x{} (restored)", comm::world_rank(), nprocs, my_row, my_col,
                         pr, pc, C.dim(0), C.dim(1), A.dim(0), A.dim(1));

        // Verify local C against reference using grid coordinates
        size_t const row_start = balanced_start(M, pr, my_row);
        size_t const col_start = balanced_start(N, pc, my_col);
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 12: Higher-rank tensor contraction on 2D grid
// C[p,q,s] = A[p,q,r] * B[r,s], rank-3 output, rank-3 and rank-2 inputs
// target_a = {p,q} → Row, target_b = {s} → Col, link = {r} → None
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - rank-3 contraction on 2D grid", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    // Dimensions divisible by 2 for the 2x2 grid
    constexpr size_t P = 8, Q = 6, R = 10, S = 12;

    // Full data on all ranks
    auto A_full = create_random_tensor<double>("A_full", P, Q, R);
    auto B_full = create_random_tensor<double>("B_full", R, S);
    (void)comm::broadcast<double>(std::span<double>(A_full.data(), A_full.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B_full.data(), B_full.size()), 0); // NOLINT

    // Reference
    auto C_ref = create_zero_tensor<double>("C_ref", P, Q, S);
    tensor_algebra::einsum(Indices{p, q, s}, &C_ref, Indices{p, q, r}, A_full, Indices{r, s}, B_full);

    // Deferred C, pre-allocated A and B
    cg::Graph graph("dist_rank3_2d");
    auto     &C = graph.declare_zero_tensor<double, 3>(std::string("C"), P, Q, S);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pqr;rs->pqs", &C, A_full, B_full);
    }

    // Outer-product (no SUMMA) with low threshold
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        int   pr   = grid.rows();
        int   pc   = grid.cols();

        // C is [Row,Row,Col]: p->Row, q->Row, s->Col
        size_t const p_chunk = (P + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
        size_t const q_chunk = (Q + static_cast<size_t>(pr) - 1) / static_cast<size_t>(pr);
        size_t const s_chunk = (S + static_cast<size_t>(pc) - 1) / static_cast<size_t>(pc);

        CHECK(C.dim(0) <= p_chunk);
        CHECK(C.dim(1) <= q_chunk);
        CHECK(C.dim(2) <= s_chunk);

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: C local = {}x{}x{}", comm::world_rank(), nprocs, grid.my_row(), grid.my_col(), pr,
                         pc, C.dim(0), C.dim(1), C.dim(2));

        // Verify local C against reference
        size_t const p_start = balanced_start(P, pr, grid.my_row());
        size_t const q_start = balanced_start(Q, pr, grid.my_row());
        size_t const s_start = balanced_start(S, pc, grid.my_col());

        for (size_t pp = 0; pp < C.dim(0); pp++) {
            for (size_t qq = 0; qq < C.dim(1); qq++) {
                for (size_t ss = 0; ss < C.dim(2); ss++) {
                    CHECK(C(pp, qq, ss) == Catch::Approx(C_ref(p_start + pp, q_start + qq, s_start + ss)).margin(1e-10));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 13: Batched GEMM distributes batch dimension
// C[b,i,j] = A[b,i,k] * B[b,k,j], batch index b is "shared" and should
// be distributed along one grid axis.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - batched GEMM distributes batch dim", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    constexpr size_t B_dim = 4, I_dim = 8, K_dim = 6, J_dim = 10;

    auto A_full = create_random_tensor<double>("A_full", B_dim, I_dim, K_dim);
    auto B_full = create_random_tensor<double>("B_full", B_dim, K_dim, J_dim);
    (void)comm::broadcast<double>(std::span<double>(A_full.data(), A_full.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B_full.data(), B_full.size()), 0); // NOLINT

    auto C_ref = create_zero_tensor<double>("C_ref", B_dim, I_dim, J_dim);
    tensor_algebra::einsum(Indices{n, i, j}, &C_ref, Indices{n, i, k}, A_full, Indices{n, k, j}, B_full);

    // Deferred C, pre-allocated A and B
    cg::Graph graph("dist_batched");
    auto     &C = graph.declare_zero_tensor<double, 3>(std::string("C"), B_dim, I_dim, J_dim);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("nik;nkj->nij", &C, A_full, B_full);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        // Batch dim n is "shared" → distributed along one axis
        // With the balancing heuristic, n (processed first) goes to Row
        // i (target_a, processed second) also goes to Row (or Col depending on counts)
        // j (target_b) goes to Col
        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: C local = {}x{}x{}", comm::world_rank(), nprocs, grid.my_row(), grid.my_col(),
                         grid.rows(), grid.cols(), C.dim(0), C.dim(1), C.dim(2));

        // The key check: C should be smaller than global dims (some dim distributed)
        CHECK(C.size() < B_dim * I_dim * J_dim);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 14: Scale on distributed tensor
// Verify that scale(alpha, C) works correctly after a distributed einsum.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - scale on distributed tensor", "[ComputeGraph][Distributed]") {
    int const        nprocs = comm::world_size();
    constexpr size_t M = 20, K = 10, N = 16;

    auto A = create_random_tensor<double>("A", M, K);
    auto B = create_random_tensor<double>("B", K, N);
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT

    // Reference: C = 2.0 * (A * B)
    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    // Graph: einsum then scale, C is deferred+distributed
    cg::Graph graph("dist_scale");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto        &grid      = comm::ProcessGrid::default_grid();
        size_t const row_start = balanced_start(M, grid.rows(), grid.my_row());
        size_t const col_start = balanced_start(N, grid.cols(), grid.my_col());
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 15: Chain of deferred einsums propagates distribution
// T = A*B; C = T*D, both T and C deferred, distributions should be compatible.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - deferred chain propagates distribution", "[ComputeGraph][Distributed]") {
    int              nprocs = comm::world_size();
    constexpr size_t M = 20, K1 = 12, K2 = 16, N = 10;

    auto A = create_random_tensor<double>("A", M, K1);
    auto D = create_random_tensor<double>("D", K2, N);
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(D.data(), D.size()), 0); // NOLINT

    // Reference
    auto B = create_random_tensor<double>("B_full", K1, K2);
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT
    auto T_ref = create_zero_tensor<double>("T_ref", M, K2);
    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    tensor_algebra::einsum(Indices{i, j}, &T_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, T_ref, Indices{k, j}, D);

    // Graph: both T and C deferred
    cg::Graph graph("dist_chain");
    auto     &T = graph.declare_zero_tensor<double, 2>(std::string("T"), M, K2);
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);

    // Custom init for T: fill from B (T is intermediate, but we need its input data)
    // Actually T is computed FROM A and B_full, so we don't init T, it's the output.
    // But we need B as a pre-allocated input.

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &T, A, B);
        cg::einsum("ik;kj->ij", &C, T, D);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(T.is_materialized());
    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        // Both T and C should be distributed along Row (i-index)
        EINSUMS_LOG_INFO("Rank {}/{}: T = {}x{}, C = {}x{}", comm::world_rank(), nprocs, T.dim(0), T.dim(1), C.dim(0), C.dim(1));

        size_t const row_start = balanced_start(M, grid.rows(), grid.my_row());
        size_t const col_start = balanced_start(N, grid.cols(), grid.my_col());
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 16: Capture-aware dot product
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - dot product in graph capture", "[ComputeGraph][Distributed]") {
    auto A = create_random_tensor<double>("A", 10, 10);
    auto B = create_random_tensor<double>("B", 10, 10);
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT

    double const ref_result = linear_algebra::dot(A, B);

    double    result = 0.0;
    cg::Graph graph("dist_dot");
    {
        cg::CaptureGuard const guard(graph);
        cg::dot(&result, A, B);
    }

    graph.execute();

    CHECK(result == Catch::Approx(ref_result).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 17: Permute/transpose on distributed tensor
// C[j,i] = A[i,j] where A is pre-allocated and C is deferred+distributed.
// The permute crosses grid axes (Row↔Col), requiring InputSlicing to slice
// A along transposed dimensions.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - permute with cross-axis redistribution", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    constexpr size_t M = 20, N = 16;

    auto A = create_random_tensor<double>("A", M, N);
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0); // NOLINT

    // Reference: C[j,i] = A[i,j]
    auto C_ref = create_zero_tensor<double>("C_ref", N, M);
    tensor_algebra::permute(0.0, Indices{j, i}, &C_ref, 1.0, Indices{i, j}, A);

    // Graph: C is deferred, distributed. A is pre-allocated.
    cg::Graph graph("dist_permute");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), N, M);

    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &C, 1.0, A);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        int   pr   = grid.rows();
        int   pc   = grid.cols();

        // C[j,i] is distributed: j→Row (target for C), i→Col
        // But wait, permute has no "target_a/target_b" classification.
        // DistributionPlanning sees C as the output of a permute node.
        // Since permute is not OpKind::Einsum, it won't be in tensor_usage.
        // C gets distributed based on its dims only (largest dim → Row).
        // Let's just verify correctness.

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: C local = {}x{}", comm::world_rank(), nprocs, grid.my_row(), grid.my_col(), pr, pc,
                         C.dim(0), C.dim(1));

        // Verify local C against reference
        size_t const row_start = balanced_start(N, pr, grid.my_row());
        size_t const col_start = balanced_start(M, pc, grid.my_col());
        for (size_t ii = 0; ii < C.dim(0); ii++) {
            for (size_t jj = 0; jj < C.dim(1); jj++) {
                CHECK(C(ii, jj) == Catch::Approx(C_ref(row_start + ii, col_start + jj)).margin(1e-10));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 18: Batched transposed GEMM C[n,i,j] = A[n,i,k] * B[n,j,k]
// Shared batch dim n, target_a=i, target_b=j, link=k, B transposed on k.
// Tests: shared index distribution + multiple Row dims + transpose dispatch
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - batched transposed GEMM", "[ComputeGraph][Distributed]") {
    int nprocs = comm::world_size();

    constexpr size_t N_dim = 4, I_dim = 8, J_dim = 6, K_dim = 10;

    auto A_full = create_random_tensor<double>("A_full", N_dim, I_dim, K_dim);
    auto B_full = create_random_tensor<double>("B_full", N_dim, J_dim, K_dim);
    (void)comm::broadcast<double>(std::span<double>(A_full.data(), A_full.size()), 0); // NOLINT
    (void)comm::broadcast<double>(std::span<double>(B_full.data(), B_full.size()), 0); // NOLINT

    // Reference
    auto C_ref = create_zero_tensor<double>("C_ref", N_dim, I_dim, J_dim);
    tensor_algebra::einsum(Indices{n, i, j}, &C_ref, Indices{n, i, k}, A_full, Indices{n, j, k}, B_full);

    // Graph: deferred C, pre-allocated A and B
    cg::Graph graph("dist_batched_trans");
    auto     &C = graph.declare_zero_tensor<double, 3>(std::string("C"), N_dim, I_dim, J_dim);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("nik;njk->nij", &C, A_full, B_full);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(C.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(C, C_ref);
    } else {
        auto &grid = comm::ProcessGrid::default_grid();
        int   pr   = grid.rows();
        int   pc   = grid.cols();

        EINSUMS_LOG_INFO("Rank {}/{} ({},{}) grid {}x{}: C local = {}x{}x{}", comm::world_rank(), nprocs, grid.my_row(), grid.my_col(), pr,
                         pc, C.dim(0), C.dim(1), C.dim(2));

        // n→Row, i→Row, j→Col
        size_t const n_start = balanced_start(N_dim, pr, grid.my_row());
        size_t const i_start = balanced_start(I_dim, pr, grid.my_row());
        size_t const j_start = balanced_start(J_dim, pc, grid.my_col());

        for (size_t nn = 0; nn < C.dim(0); nn++) {
            for (size_t ii = 0; ii < C.dim(1); ii++) {
                for (size_t jj = 0; jj < C.dim(2); jj++) {
                    CHECK(C(nn, ii, jj) == Catch::Approx(C_ref(n_start + nn, i_start + ii, j_start + jj)).margin(1e-10));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 19: declare_tensor_filled with range() and global()
// Simulates ERI-like batch fill with automatic distribution support.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - declare_tensor_filled with range/global", "[ComputeGraph][Distributed]") {
    int              nprocs = comm::world_size();
    constexpr size_t N      = 10;

    // Reference: fill a 2D tensor with f(i,j) = i*100 + j
    auto ref = create_zero_tensor<double>("ref", N, N);
    for (size_t ii = 0; ii < N; ii++)
        for (size_t jj = 0; jj < N; jj++)
            ref(ii, jj) = static_cast<double>(ii * 100 + jj);

    // Graph: declare_tensor_filled, the fill lambda uses range() and global()
    cg::Graph graph("dist_fill");
    auto     &T = graph.declare_tensor_filled<double, 2>(std::string("T"), Dim<2>{N, N}, [](Tensor<double, 2> &t) {
        auto [i0, i1] = t.range(0);
        auto [j0, j1] = t.range(1);
        for (size_t ii = i0; ii < i1; ii++) {
            for (size_t jj = j0; jj < j1; jj++) {
                t.global(ii, jj) = static_cast<double>(ii * 100 + jj);
            }
        }
    });

    // Use T in an einsum so DistributionPlanning can classify its indices
    auto B = create_random_tensor<double>("B", N, N);
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0); // NOLINT
    auto &R = graph.declare_zero_tensor<double, 2>(std::string("R"), N, N);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &R, T, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(T.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(T, ref);
    } else {
        auto &grid    = comm::ProcessGrid::default_grid();
        auto [i0, i1] = T.range(0);
        auto [j0, j1] = T.range(1);

        EINSUMS_LOG_INFO("Rank {}/{}: T local = {}x{}, range = [{},{}) x [{},{})", comm::world_rank(), nprocs, T.dim(0), T.dim(1), i0, i1,
                         j0, j1);

        for (size_t ii = i0; ii < i1; ii++) {
            for (size_t jj = j0; jj < j1; jj++) {
                auto expected = static_cast<double>(ii * 100 + jj);
                CHECK(T.global(ii, jj) == expected);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 20: Simulated batch ERI fill + contraction
// Demonstrates the chemistry blueprint pattern: fill ERIs in shell batches
// using range(), then contract with MO coefficients.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Distributed - batch ERI fill + contraction", "[ComputeGraph][Distributed]") {
    int              nprocs = comm::world_size();
    constexpr size_t nao    = 12; // Small basis for testing
    constexpr size_t nmo    = 8;

    // Simulate shell structure: 4 shells of 3 basis functions each
    constexpr int    nshell     = 4;
    constexpr size_t shell_size = 3; // Uniform for simplicity

    // MO coefficients (pre-allocated, replicated)
    auto C = create_random_tensor<double>("C", nao, nmo);
    (void)comm::broadcast<double>(std::span<double>(C.data(), C.size()), 0); // NOLINT

    // Reference: compute ERIs and half-transform on one rank
    auto eri_ref = create_zero_tensor<double>("eri_ref", nao, nao, nao, nao);
    for (size_t p = 0; p < nao; p++)
        for (size_t q = 0; q < nao; q++)
            for (size_t r = 0; r < nao; r++)
                for (size_t s = 0; s < nao; s++)
                    eri_ref(p, q, r, s) = 1.0 / (r + 1.0 + p + q + s); // NOLINT Fake ERI

    auto half_ref = create_zero_tensor<double>("half_ref", nao, nao, nao, nmo);
    tensor_algebra::einsum(Indices{p, q, r, a}, &half_ref, Indices{p, q, r, s}, eri_ref, Indices{s, a}, C);

    // Graph: fill ERIs using batch pattern, then contract
    cg::Graph graph("dist_eri");
    auto     &eri = graph.declare_tensor_filled<double, 4>(
        std::string("ERI"), Dim<4>{nao, nao, nao, nao}, [nao, nshell, shell_size](Tensor<double, 4> &T) {
            // Get local ranges, only fill shells that overlap
            auto [p0, p1] = T.range(0);
            auto [q0, q1] = T.range(1);
            auto [r0, r1] = T.range(2);
            auto [s0, s1] = T.range(3);

            // Iterate over shell quartets
            for (int P = 0; P < nshell; P++) {
                size_t const pf = P * shell_size;
                size_t const pl = pf + shell_size; // NOLINT(misc-confusable-identifiers)
                if (pl <= p0 || pf >= p1)
                    continue; // Skip shells outside our range

                for (int Q = 0; Q < nshell; Q++) {
                    size_t const qf = Q * shell_size;
                    size_t const ql = qf + shell_size; // NOLINT(misc-confusable-identifiers)
                    if (ql <= q0 || qf >= q1)
                        continue;

                    for (int R = 0; R < nshell; R++) {
                        size_t const rf = R * shell_size;
                        size_t const rl = rf + shell_size; // NOLINT(misc-confusable-identifiers)
                        if (rl <= r0 || rf >= r1)
                            continue;

                        for (int S = 0; S < nshell; S++) {
                            size_t const sf = S * shell_size;
                            size_t const sl = sf + shell_size; // NOLINT(misc-confusable-identifiers)
                            if (sl <= s0 || sf >= s1)
                                continue;

                            // "Compute" this shell quartet, only store local elements
                            for (size_t pp = std::max(pf, p0); pp < std::min(pl, p1); pp++)
                                for (size_t qq = std::max(qf, q0); qq < std::min(ql, q1); qq++)
                                    for (size_t rr = std::max(rf, r0); rr < std::min(rl, r1); rr++)
                                        for (size_t ss = std::max(sf, s0); ss < std::min(sl, s1); ss++)
                                            T.global(pp, qq, rr, ss) =
                                                1.0 / (1.0 + pp + qq + rr + ss); // NOLINT(bugprone-narrowing-conversions)
                        }
                    }
                }
            }
        });

    auto &half = graph.declare_zero_tensor<double, 4>(std::string("half"), nao, nao, nao, nmo);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("pqrs;sa->pqra", &half, eri, C);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    graph.apply(pm);

    graph.execute();

    REQUIRE(eri.is_materialized());
    REQUIRE(half.is_materialized());

    if (nprocs == 1) {
        check_tensors_equal(half, half_ref, 1e-10);
    } else {
        // Verify local portion of half-transformed integrals
        auto [p0, p1] = half.range(0);
        auto [q0, q1] = half.range(1);
        auto [r0, r1] = half.range(2);
        auto [a0, a1] = half.range(3);

        EINSUMS_LOG_INFO("Rank {}/{}: half local = {}x{}x{}x{}", comm::world_rank(), nprocs, half.dim(0), half.dim(1), half.dim(2),
                         half.dim(3));

        for (size_t pp = p0; pp < p1; pp++)
            for (size_t qq = q0; qq < q1; qq++)
                for (size_t rr = r0; rr < r1; rr++)
                    for (size_t aa = a0; aa < a1; aa++)
                        CHECK(half.global(pp, qq, rr, aa) == Catch::Approx(half_ref(pp, qq, rr, aa)).margin(1e-10));
    }
}
