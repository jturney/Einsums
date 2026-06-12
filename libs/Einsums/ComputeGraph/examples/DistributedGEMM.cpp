//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file DistributedGEMM.cpp
 * @brief Example: distributed tensor contraction using ComputeGraph + MPI.
 *
 * Demonstrates the transparent distributed computing system:
 * 1. Declare deferred tensors (globally-sized, allocated lazily)
 * 2. Capture einsum operations into a graph
 * 3. Optimization passes automatically distribute, slice, and communicate
 * 4. Execute: each rank computes its local partition
 *
 * Run with: mpirun -np 4 ./CG_DistributedGEMM
 *
 * With 4 ranks, a 2×2 process grid is created. Each rank holds M/2 × N/2
 * of the output C, with zero communication (outer-product parallelism).
 *
 * Works with mock backend too (single rank, no MPI needed).
 */

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/ProcessGrid.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <chrono>
#include <span>

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

int einsums_main() {
    constexpr size_t M = 200, K = 100, N = 160;

    // ── Print grid info ─────────────────────────────────────────────────
    auto &grid = comm::ProcessGrid::default_grid();
    println("Rank {}/{} at grid ({},{}) in {}x{} grid", comm::world_rank(), comm::world_size(), grid.my_row(), grid.my_col(), grid.rows(),
            grid.cols());

    // ── Create input data (pre-allocated, replicated on all ranks) ──────
    auto A = create_random_tensor<double>("A", M, K);
    auto B = create_random_tensor<double>("B", K, N);

    // Broadcast from rank 0 so all ranks have identical data
    (void)comm::broadcast<double>(std::span<double>(A.data(), A.size()), 0);
    (void)comm::broadcast<double>(std::span<double>(B.data(), B.size()), 0);

    // ── Method 1: Direct einsum (serial reference) ──────────────────────
    auto C_ref = create_zero_tensor<double>("C_ref", M, N);
    auto t0    = std::chrono::high_resolution_clock::now();
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto   t1     = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (comm::is_root()) {
        println("\n--- Method 1: Direct einsum (serial) ---");
        println("  Time: {:.2f} ms", ref_ms);
        println("  C(0,0) = {:.8f}", C_ref(0, 0));
    }

    // ── Method 2: ComputeGraph with automatic distribution ──────────────
    // C is deferred, DistributionPlanning will distribute it on the 2D grid.
    // A and B are pre-allocated, InputSlicing auto-slices them.
    cg::Graph graph("distributed_gemm");
    auto     &C = graph.declare_zero_tensor<double, 2>(std::string("C"), M, N);

    {
        cg::CaptureGuard guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Apply optimization passes with low threshold to force distribution
    // (Default threshold is 64MB; our tensors are smaller, so we lower it)
    cg::PassManager pm;
    pm.add<cg::passes::DistributionPlanning>(/*threshold=*/1, /*enable_summa=*/false);
    pm.add<cg::passes::Materialization>();
    pm.add<cg::passes::InputSlicing>();
    pm.add<cg::passes::CommunicationInsertion>();
    pm.add<cg::passes::CommunicationScheduling>();
    graph.apply(pm);

    t0 = std::chrono::high_resolution_clock::now();
    graph.execute();
    t1             = std::chrono::high_resolution_clock::now();
    double dist_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    println("Rank {}: C local dims = {}x{}, time = {:.2f} ms", comm::world_rank(), C.dim(0), C.dim(1), dist_ms);

    // ── Verify correctness ──────────────────────────────────────────────
    // Each rank checks its local partition against the reference.
    size_t row_start = 0, col_start = 0;
    if (comm::world_size() > 1) {
        int pr = grid.rows(), pc = grid.cols();
        // Balanced blocking start computation
        auto balanced = [](size_t total, int nprocs, int pos) -> size_t {
            size_t base      = total / static_cast<size_t>(nprocs);
            size_t remainder = total % static_cast<size_t>(nprocs);
            if (static_cast<size_t>(pos) < remainder)
                return static_cast<size_t>(pos) * (base + 1);
            else
                return remainder * (base + 1) + (static_cast<size_t>(pos) - remainder) * base;
        };
        row_start = balanced(M, pr, grid.my_row());
        col_start = balanced(N, pc, grid.my_col());
    }

    double max_diff = 0.0;
    for (size_t ii = 0; ii < C.dim(0); ii++) {
        for (size_t jj = 0; jj < C.dim(1); jj++) {
            double diff = std::abs(C(ii, jj) - C_ref(row_start + ii, col_start + jj));
            max_diff    = std::max(max_diff, diff);
        }
    }

    println("Rank {}: max error = {:.2e}", comm::world_rank(), max_diff);

    if (comm::is_root()) {
        println("\n--- Summary ---");
        println("  Grid: {}x{} ({} ranks)", grid.rows(), grid.cols(), comm::world_size());
        println("  Tensor dims: C[{},{}] = A[{},{}] * B[{},{}]", M, N, M, K, K, N);
        println("  Serial: {:.2f} ms", ref_ms);
        println("  Distributed: {:.2f} ms per rank", dist_ms);
        if (max_diff < 1e-10) {
            println("  Correctness: PASS (max error {:.2e})", max_diff);
        } else {
            println("  Correctness: FAIL (max error {:.2e})", max_diff);
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
