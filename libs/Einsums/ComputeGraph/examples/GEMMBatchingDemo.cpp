//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GEMMBatchingDemo.cpp
/// @brief Showcase the GEMMBatching optimization pass.
///
/// Demonstrates:
///   - How capturing many independent GEMM-pattern einsums yields one
///     einsum Node per contraction.
///   - How GEMMBatching collapses groups of compatible GEMMs into a
///     single BatchedGemm node backed by blas::gemm_batch.
///   - The kinds of differences between einsums that prevent batching
///     (dims, alpha/beta prefactors, transpose patterns, types).
///   - A small timing measurement so you can see the dispatch-overhead
///     saving on short contractions.
///
/// This example is contrived, the whole point is that the optimization
/// runs automatically via `PassManager::create_default()` once you
/// capture a workload that happens to have many independent GEMMs
/// (stacked attention heads, Kronecker-factored updates, batched
/// per-sample evaluations, etc.).

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <chrono>
#include <vector>

namespace cg = einsums::compute_graph;

namespace {

// Time the total execute() wall clock across `reps` invocations.
double time_execute_us(cg::Graph &graph, int reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        graph.execute();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

} // namespace

int einsums_main() {
    using namespace einsums;

    einsums::println("=== GEMMBatching demo ===\n");

    // ── 1. A workload of N independent small GEMMs ─────────────────────────
    //
    // Stacked attention heads, per-sample transforms, and many chemistry
    // kernels produce exactly this pattern: N matrices A_i, B_i, C_i
    // with identical shapes and no dependencies between the individual
    // contractions.
    constexpr int BATCH = 32;
    constexpr int M = 16, K = 16, N = 16;

    std::vector<Tensor<double, 2>> As, Bs, Cs_unopt, Cs_opt;
    As.reserve(BATCH);
    Bs.reserve(BATCH);
    Cs_unopt.reserve(BATCH);
    Cs_opt.reserve(BATCH);
    for (int i = 0; i < BATCH; ++i) {
        As.push_back(create_random_tensor<double>(fmt::format("A{}", i), M, K));
        Bs.push_back(create_random_tensor<double>(fmt::format("B{}", i), K, N));
        Cs_unopt.push_back(create_zero_tensor<double>(fmt::format("Cu{}", i), M, N));
        Cs_opt.push_back(create_zero_tensor<double>(fmt::format("Co{}", i), M, N));
    }

    // ── 2. Capture the same workload twice: unoptimized + optimized ─────────
    cg::Graph graph_unopt("unoptimized");
    {
        cg::CaptureGuard guard(graph_unopt);
        for (int i = 0; i < BATCH; ++i)
            cg::einsum("ik;kj->ij", &Cs_unopt[i], As[i], Bs[i]);
    }

    cg::Graph graph_opt("optimized");
    {
        cg::CaptureGuard guard(graph_opt);
        for (int i = 0; i < BATCH; ++i)
            cg::einsum("ik;kj->ij", &Cs_opt[i], As[i], Bs[i]);
    }

    einsums::println("Captured {} independent {}x{}x{} GEMMs per graph", BATCH, M, K, N);
    einsums::println("  unoptimized graph: {} nodes", graph_unopt.num_nodes());

    // ── 3. Run just the GEMMBatching pass on one graph ──────────────────────
    auto [modified, pass] = graph_opt.apply<cg::passes::GEMMBatching>();
    einsums::println("  GEMMBatching: modified={} num_batches={} total_batched={}", modified, pass.num_batches(), pass.total_batched());
    einsums::println("  optimized graph:   {} nodes (one BatchedGemm collapses all {})", graph_opt.num_nodes(), BATCH);

    // ── 4. Warm up + time both graphs ───────────────────────────────────────
    constexpr int WARMUP = 3;
    constexpr int REPS   = 20;
    for (int i = 0; i < WARMUP; ++i) {
        graph_unopt.execute();
        graph_opt.execute();
    }

    double us_unopt = time_execute_us(graph_unopt, REPS);
    double us_opt   = time_execute_us(graph_opt, REPS);

    einsums::println("\nTiming ({} reps, {}x{}x{} × {}-GEMM batch):", REPS, M, K, N, BATCH);
    einsums::println("  {} sequential gemm nodes: {:.1f} µs / execute()", BATCH, us_unopt);
    einsums::println("  1 gemm_batch node:          {:.1f} µs / execute()", us_opt);
    if (us_opt > 0.0)
        einsums::println("  speedup:                    {:.2f}x", us_unopt / us_opt);
    einsums::println("  (win comes from OpenMP-parallel batch execution;");
    einsums::println("   blas::gemm_batch is currently a parallel loop of dgemms,");
    einsums::println("   not a vendor-native batched GEMM)");

    // ── 5. Correctness: batched result matches the separate one ────────────
    double max_err = 0.0;
    for (int i = 0; i < BATCH; ++i) {
        auto const *u = Cs_unopt[i].data();
        auto const *o = Cs_opt[i].data();
        for (size_t j = 0; j < Cs_unopt[i].size(); ++j)
            max_err = std::max(max_err, std::abs(u[j] - o[j]));
    }
    einsums::println("\nMax |batched - separate|: {:.2e}", max_err);

    // ── 6. Show that the pass IS the default ───────────────────────────────
    //
    // The manual apply<GEMMBatching> above was pedagogical. In
    // production you just capture and call graph.apply(pm):
    //
    //     auto pm = cg::PassManager::create_default();
    //     graph.apply(pm);   // GEMMBatching runs automatically
    //
    // along with ConstantFolding, PermuteFusion, CSE, etc.
    einsums::println("\nGEMMBatching is in PassManager::create_default() — no opt-in needed.");

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
