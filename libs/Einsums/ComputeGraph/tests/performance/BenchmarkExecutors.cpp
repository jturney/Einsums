//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkExecutors.cpp
/// @brief Benchmarks comparing Sequential, OpenMP, and Dataflow executors on
///        graphs with varying degrees of parallelism.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cstdio>
#include <print>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;
namespace cg = einsums::compute_graph;

// ═══════════════════════════════════════════════════════════════════════════════
// Linear chain: no parallelism (sequential wins)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench Executor: linear chain 4 nodes N=50", "[ComputeGraph][Executor][benchmark]") {
    LabeledSection0();
    constexpr size_t N = 50;
    auto             A = create_random_tensor<double>("A", N, N);
    auto             B = create_random_tensor<double>("B", N, N);
    auto             C = create_zero_tensor<double>("C", N, N);
    auto             D = create_zero_tensor<double>("D", N, N);

    cg::Graph graph("linear_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
        cg::permute("ji <- ij", 0.0, &D, 1.0, C);
        cg::axpy(1.0, A, &D);
    }

    cg::SequentialExecutor seq;
    cg::OpenMPExecutor     omp;
    cg::DataflowExecutor   df;

    auto t_seq = time_us(
        "sequential",
        [&]() {
            C.zero();
            D.zero();
            graph.execute(seq);
        },
        20);
    auto t_omp = time_us(
        "openmp",
        [&]() {
            C.zero();
            D.zero();
            graph.execute(omp);
        },
        20);
    auto t_df = time_us(
        "dataflow",
        [&]() {
            C.zero();
            D.zero();
            graph.execute(df);
        },
        20);

    std::println("[Executor linear N=50] seq: {:.2f} us  omp: {:.2f} us  df: {:.2f} us", t_seq.avg, t_omp.avg, t_df.avg);

    // Publish structured results to profiler server
    ProfileAnnotate("topology", "linear_chain");
    ProfileAnnotate("nodes", int64_t(4));
    ProfileAnnotate("N", int64_t(N));
    publish_benchmark_result("Executor linear N=50", "t_sequential", static_cast<int>(N), t_seq);
    publish_benchmark_result("Executor linear N=50", "t_openmp", static_cast<int>(N), t_omp);
    publish_benchmark_result("Executor linear N=50", "t_dataflow", static_cast<int>(N), t_df);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wide fan-out: lots of parallelism (OpenMP/Dataflow should win)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench Executor: 8 independent nodes N=30", "[ComputeGraph][Executor][benchmark]") {
    LabeledSection0();
    constexpr size_t N = 30;
    ProfileAnnotate("topology", "fan_out");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("nodes", int64_t(8));
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    Tensor<double, 2> results[8] = {
        Tensor<double, 2>{"R0", N, N}, Tensor<double, 2>{"R1", N, N}, Tensor<double, 2>{"R2", N, N}, Tensor<double, 2>{"R3", N, N},
        Tensor<double, 2>{"R4", N, N}, Tensor<double, 2>{"R5", N, N}, Tensor<double, 2>{"R6", N, N}, Tensor<double, 2>{"R7", N, N},
    };

    cg::Graph graph("fan_out");
    {
        cg::CaptureGuard const guard(graph);
        for (auto &result : results) {
            cg::einsum("ik;kj->ij", &result, A, B);
        }
    }

    cg::SequentialExecutor seq;
    cg::OpenMPExecutor     omp;
    cg::DataflowExecutor   df;

    auto zero_all = [&]() {
        for (auto &r : results) // NOLINT(modernize-avoid-c-arrays)
            r.zero();
    };

    auto t_seq = time_us(
        "sequential",
        [&]() {
            zero_all();
            graph.execute(seq);
        },
        20);
    auto t_omp = time_us(
        "openmp",
        [&]() {
            zero_all();
            graph.execute(omp);
        },
        20);
    auto t_df = time_us(
        "dataflow",
        [&]() {
            zero_all();
            graph.execute(df);
        },
        20);

    std::println("[Executor fanout-8 N=30] seq: {:.2f} us  omp: {:.2f} us  df: {:.2f} us", t_seq.avg, t_omp.avg, t_df.avg);
    std::println("  seq/omp: {:.2f}x  seq/df: {:.2f}x", t_seq.avg / t_omp.avg, t_seq.avg / t_df.avg);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Diamond DAG: 2 independent branches merge
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench Executor: diamond DAG N=50", "[ComputeGraph][Executor][benchmark]") {
    LabeledSection0();
    constexpr size_t N = 50;
    ProfileAnnotate("topology", "diamond");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("nodes", int64_t(4));
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);
    auto C = create_zero_tensor<double>("C", N, N);
    auto D = create_zero_tensor<double>("D", N, N);
    auto E = create_zero_tensor<double>("E", N, N);

    cg::Graph graph("diamond");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);        // Branch 1
        cg::permute("ji <- ij", 0.0, &D, 1.0, A); // Branch 2 (independent)
        cg::axpy(1.0, C, &E);                     // Merge (depends on both)
        cg::axpy(1.0, D, &E);
    }

    cg::SequentialExecutor seq;
    cg::OpenMPExecutor     omp;
    cg::DataflowExecutor   df;

    auto zero = [&]() {
        C.zero();
        D.zero();
        E.zero();
    };

    auto t_seq = time_us(
        "sequential",
        [&]() {
            zero();
            graph.execute(seq);
        },
        20);
    auto t_omp = time_us(
        "openmp",
        [&]() {
            zero();
            graph.execute(omp);
        },
        20);
    auto t_df = time_us(
        "dataflow",
        [&]() {
            zero();
            graph.execute(df);
        },
        20);

    std::println("[Executor diamond N=50] seq: {:.2f} us  omp: {:.2f} us  df: {:.2f} us", t_seq.avg, t_omp.avg, t_df.avg);
    std::println("  seq/omp: {:.2f}x  seq/df: {:.2f}x", t_seq.avg / t_omp.avg, t_seq.avg / t_df.avg);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph replay overhead (same graph executed many times)
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench Executor: replay overhead 100 iterations N=20", "[ComputeGraph][Executor][benchmark]") {
    LabeledSection0();
    constexpr size_t N = 20;
    ProfileAnnotate("topology", "replay");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("nodes", int64_t(1));
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);
    auto C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto t_bare = time_us(
        "bare einsum",
        [&]() {
            C.zero();
            tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
        },
        100);

    auto t_graph = time_us(
        "graph replay",
        [&]() {
            C.zero();
            graph.execute();
        },
        100);

    double const overhead_pct = ((t_graph.avg - t_bare.avg) / t_bare.avg) * 100.0;
    std::println("[Replay N=20] bare: {:.2f} us  graph: {:.2f} us  overhead: {:.1f}%", t_bare.avg, t_graph.avg, overhead_pct);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PassManager overhead
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench PassManager: create_default on 10-node graph", "[ComputeGraph][PassManager][benchmark]") {
    LabeledSection0();
    constexpr size_t N = 10;
    ProfileAnnotate("topology", "passmanager");
    ProfileAnnotate("N", int64_t(N));
    ProfileAnnotate("nodes", int64_t(10));
    auto A = create_random_tensor<double>("A", N, N);
    auto B = create_random_tensor<double>("B", N, N);
    auto C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("passes");
    {
        cg::CaptureGuard const guard(graph);
        for (int n = 0; n < 5; n++) {
            cg::einsum("ik;kj->ij", 1.0, &C, 1.0, A, B);
            cg::scale(0.5, &C);
        }
    }

    auto t = time_us(
        "passmanager",
        [&]() {
            auto pm = cg::PassManager::create_default();
            graph.apply(pm);
        },
        50);

    std::println("[PassManager 10-node] {:.2f} us avg  ({} nodes after)", t.avg, graph.num_nodes());
}
