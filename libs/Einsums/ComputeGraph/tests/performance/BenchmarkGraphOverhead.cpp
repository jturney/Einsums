//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BenchmarkGraphOverhead.cpp
/// @brief Benchmarks for the graph machinery itself: capture cost, optimizer
///        pipeline cost, and replay overhead at realistic node counts.
///
/// BenchmarkExecutors.cpp compares executors on small graphs where the tensor
/// math dominates. These benchmarks do the opposite: tiny tensors (N=20) and
/// many nodes, so the framework overhead IS the signal. They exist to keep
/// the capture layer, pass pipeline, and replay path honest - an SCF/CC loop
/// replays the same graph hundreds of times, so per-node bookkeeping that
/// looks free in a single execute compounds into real time.
///
/// The graphs are chains (node i consumes node i-1's output), so CSE cannot
/// fold them and the dependency scans see real edges.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Performance.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <fmt/format.h>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::performance;
namespace cg = einsums::compute_graph;

namespace {

constexpr size_t kDim      = 20; // tensor extent: small on purpose, overhead is the signal
constexpr size_t kChainLen = 50; // einsum+scale pairs -> 100 nodes
constexpr int    kReps     = 20;

/// Capture a 2*kChainLen-node chain into `graph`:
///   T[i] = A * T[i-1]  (einsum, overwrite)
///   T[i] *= 0.99       (scale)
/// Chained inputs defeat CSE; the scale-after-overwrite pairs are NOT
/// eliminable by ScaleAbsorption (the scale follows its einsum, no later
/// overwrite), so the node count survives the default pipeline.
void capture_chain(cg::Graph &graph, Tensor<double, 2> &A, std::vector<Tensor<double, 2>> &pool) {
    cg::CaptureGuard const guard(graph);
    for (size_t n = 1; n < pool.size(); n++) {
        cg::einsum("ik;kj->ij", 0.0, &pool[n], 1.0, A, pool[n - 1]);
        cg::scale(0.99, &pool[n]);
    }
}

std::vector<Tensor<double, 2>> make_pool() {
    std::vector<Tensor<double, 2>> pool;
    pool.reserve(kChainLen + 1);
    pool.emplace_back(create_random_tensor<double>("T0", kDim, kDim));
    for (size_t n = 1; n <= kChainLen; n++) {
        pool.emplace_back(fmt::format("T{}", n), kDim, kDim);
        pool.back().zero();
    }
    return pool;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Capture cost: what does recording 100 nodes into a graph cost?
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench GraphOverhead: capture 100-node chain", "[ComputeGraph][Overhead][benchmark]") {
    LabeledSection0();
    auto A    = create_random_tensor<double>("A", kDim, kDim);
    auto pool = make_pool();

    auto t_capture = time_us(
        "capture",
        [&]() {
            cg::Graph graph("capture_bench");
            capture_chain(graph, A, pool);
        },
        kReps);

    fmt::println("[GraphOverhead capture 100n] {:.2f} us avg ({:.2f} us/node)", t_capture.avg, t_capture.avg / (2.0 * kChainLen));
    ProfileAnnotate("nodes", int64_t(2 * kChainLen));
    publish_benchmark_result("GraphOverhead capture 100n", "t_capture", static_cast<int>(2 * kChainLen), t_capture);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimizer pipeline cost: create_default() over a fresh 100-node chain.
// Reported alongside capture-only so the apply() share is derivable; the
// pipeline mutates the graph, so each rep needs a fresh capture.
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench GraphOverhead: apply default pipeline on 100-node chain", "[ComputeGraph][Overhead][benchmark]") {
    LabeledSection0();
    auto A    = create_random_tensor<double>("A", kDim, kDim);
    auto pool = make_pool();

    auto t_capture_only = time_us(
        "capture only",
        [&]() {
            cg::Graph graph("apply_bench_base");
            capture_chain(graph, A, pool);
        },
        kReps);

    size_t nodes_after     = 0;
    auto   t_capture_apply = time_us(
        "capture + apply",
        [&]() {
            cg::Graph graph("apply_bench");
            capture_chain(graph, A, pool);
            auto pm = cg::PassManager::create_default();
            graph.apply(pm);
            nodes_after = graph.num_nodes();
        },
        kReps);

    double const t_apply = t_capture_apply.avg - t_capture_only.avg;
    fmt::println("[GraphOverhead apply 100n] capture: {:.2f} us  capture+apply: {:.2f} us  -> apply: {:.2f} us ({} nodes after)",
                 t_capture_only.avg, t_capture_apply.avg, t_apply, nodes_after);
    ProfileAnnotate("nodes", int64_t(2 * kChainLen));
    publish_benchmark_result("GraphOverhead apply 100n", "t_capture_only", static_cast<int>(2 * kChainLen), t_capture_only);
    publish_benchmark_result("GraphOverhead apply 100n", "t_capture_apply", static_cast<int>(2 * kChainLen), t_capture_apply);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Serial replay overhead: execute() the same 100-node graph repeatedly, the
// SCF/CC iteration pattern. Compared against the same math run eagerly, so
// the delta is pure framework bookkeeping (profiler strings, timing report,
// slot validation, dispatch).
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench GraphOverhead: serial replay of 100-node chain", "[ComputeGraph][Overhead][benchmark]") {
    LabeledSection0();
    auto A    = create_random_tensor<double>("A", kDim, kDim);
    auto pool = make_pool();

    cg::Graph graph("replay_bench");
    capture_chain(graph, A, pool);
    graph.execute(); // first execute: sort + validation warm-up

    auto t_replay = time_us(
        "graph replay", [&]() { graph.execute(); }, kReps);

    auto t_eager = time_us(
        "eager equivalent",
        [&]() {
            for (size_t n = 1; n < pool.size(); n++) {
                tensor_algebra::einsum(0.0, Indices{i, j}, &pool[n], 1.0, Indices{i, k}, A, Indices{k, j}, pool[n - 1]);
                linear_algebra::scale(0.99, &pool[n]);
            }
        },
        kReps);

    double const per_node_overhead_us = (t_replay.avg - t_eager.avg) / (2.0 * kChainLen);
    fmt::println("[GraphOverhead serial replay 100n] graph: {:.2f} us  eager: {:.2f} us  overhead: {:.2f} us/node", t_replay.avg,
                 t_eager.avg, per_node_overhead_us);
    ProfileAnnotate("nodes", int64_t(2 * kChainLen));
    publish_benchmark_result("GraphOverhead serial replay 100n", "t_graph", static_cast<int>(2 * kChainLen), t_replay);
    publish_benchmark_result("GraphOverhead serial replay 100n", "t_eager", static_cast<int>(2 * kChainLen), t_eager);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dataflow replay: same graph through the TaskPool executor. A chain has zero
// parallelism, so ALL of the gap vs serial replay is scheduling scaffold
// (task allocation, when_all states, submission locking).
// ═══════════════════════════════════════════════════════════════════════════════

EINSUMS_TEST_CASE("Bench GraphOverhead: dataflow replay of 100-node chain", "[ComputeGraph][Overhead][benchmark]") {
    LabeledSection0();
    auto A    = create_random_tensor<double>("A", kDim, kDim);
    auto pool = make_pool();

    cg::Graph graph("df_replay_bench");
    capture_chain(graph, A, pool);
    graph.execute(); // sort once

    cg::SequentialExecutor seq;
    cg::DataflowExecutor   df;

    auto t_seq = time_us(
        "sequential", [&]() { graph.execute(seq); }, kReps);
    auto t_df = time_us(
        "dataflow", [&]() { graph.execute(df); }, kReps);

    double const scaffold_per_node_us = (t_df.avg - t_seq.avg) / (2.0 * kChainLen);
    fmt::println("[GraphOverhead dataflow replay 100n] seq: {:.2f} us  df: {:.2f} us  scaffold: {:.2f} us/node", t_seq.avg, t_df.avg,
                 scaffold_per_node_us);
    ProfileAnnotate("nodes", int64_t(2 * kChainLen));
    publish_benchmark_result("GraphOverhead dataflow replay 100n", "t_sequential", static_cast<int>(2 * kChainLen), t_seq);
    publish_benchmark_result("GraphOverhead dataflow replay 100n", "t_dataflow", static_cast<int>(2 * kChainLen), t_df);
}
