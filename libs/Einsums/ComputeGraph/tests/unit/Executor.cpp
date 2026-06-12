//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("SequentialExecutor - matches default execute()", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);
    auto D = create_zero_tensor<double>("D", 4, 5);

    // Default execute
    cg::Graph graph1("default");
    {
        cg::CaptureGuard const guard(graph1);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    graph1.execute();

    // Sequential executor
    cg::Graph graph2("sequential");
    {
        cg::CaptureGuard const guard(graph2);
        cg::einsum("ik;kj->ij", &D, A, B);
    }
    cg::SequentialExecutor seq;
    graph2.execute(seq);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - D(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("OpenMPExecutor - produces correct results", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 6, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_zero_tensor<double>("C", 6, 5);
    auto D = create_zero_tensor<double>("D", 6, 5);

    // Reference: eager
    tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C);

    // OpenMP executor
    cg::Graph omp_graph("openmp");
    {
        cg::CaptureGuard const guard(omp_graph);
        cg::einsum("ik;kj->ij", &D, A, B);
        cg::scale(2.0, &D);
    }
    cg::OpenMPExecutor omp;
    omp_graph.execute(omp);

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - D(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("OpenMPExecutor - independent nodes", "[ComputeGraph][Executor]") {
    // Two independent operations, can run in parallel
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);
    auto D = create_zero_tensor<double>("D", 5, 5);

    auto C_ref = create_zero_tensor<double>("Cref", 5, 5);
    auto D_ref = create_zero_tensor<double>("Dref", 5, 5);

    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &D_ref, Indices{i, k}, B, Indices{k, j}, A);

    cg::Graph graph("independent");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, B, A);
    }

    cg::OpenMPExecutor omp;
    graph.execute(omp);

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("OpenMPExecutor - dependent chain", "[ComputeGraph][Executor]") {
    // Chain: C = A*B, then D = C*A, must execute in order
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    auto C_ref = create_zero_tensor<double>("Cref", 4, 4);
    auto D_ref = create_zero_tensor<double>("Dref", 4, 4);

    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, j}, &D_ref, Indices{i, k}, C_ref, Indices{k, j}, A);

    cg::Graph graph("chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::einsum("ik;kj->ij", &D, C, A);
    }

    cg::OpenMPExecutor omp;
    graph.execute(omp);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
            REQUIRE(std::abs(D(ii, jj) - D_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("OpenMPExecutor - pipeline with executor", "[ComputeGraph][Executor]") {
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Pipeline pipeline("omp_pipeline");

    {
        auto                  &setup = pipeline.add_stage("compute");
        cg::CaptureGuard const guard(setup);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    size_t count = 0;
    {
        auto                  &loop = pipeline.add_loop("accum", 3, [&](size_t iter) {
            count = iter + 1;
            return iter < 2;
        });
        cg::CaptureGuard const guard(loop);
        cg::axpy(1.0, C, &acc);
    }

    cg::OpenMPExecutor omp;
    pipeline.execute(omp);

    REQUIRE(count == 3);

    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(acc(ii, jj) - 3.0 * C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("DependencyInfo - correct structure", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    cg::Graph graph("deps_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B); // Node 0: reads A,B writes C
        cg::einsum("ik;kj->ij", &D, C, A); // Node 1: reads C,A writes D
    }

    auto const &deps = graph.dependencies();

    // Node 0 has no predecessors (A, B are inputs, no prior writer)
    REQUIRE(deps.predecessors[0].empty());

    // Node 1 depends on Node 0 (reads C which Node 0 writes)
    REQUIRE(deps.predecessors[1].size() >= 1);

    // Node 0 has Node 1 as successor
    REQUIRE(deps.successors[0].size() >= 1);
}

// ── DataflowExecutor tests ───────────────────────────────────────────────────

TEST_CASE("DataflowExecutor - matches sequential results", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 5, 4);
    auto B = create_random_tensor<double>("B", 4, 6);
    auto C = create_zero_tensor<double>("C", 5, 6);
    auto D = create_zero_tensor<double>("D", 5, 6);

    // Sequential reference
    cg::Graph graph_seq("seq");
    {
        cg::CaptureGuard const guard(graph_seq);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
    }
    graph_seq.execute();
    auto C_ref = Tensor<double, 2>(C);

    // DataflowExecutor
    C.zero();
    cg::Graph graph_df("dataflow");
    {
        cg::CaptureGuard const guard(graph_df);
        cg::einsum("ik;kj->ij", &D, A, B);
        cg::scale(2.0, &D);
    }

    cg::DataflowExecutor df;
    graph_df.execute(df);

    for (size_t ii = 0; ii < 5; ii++)
        for (size_t jj = 0; jj < 6; jj++)
            REQUIRE_THAT(D(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
}

TEST_CASE("DataflowExecutor - diamond DAG", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto D = create_zero_tensor<double>("D", 4, 4);

    // B = 2*A, C = A^T (independent), D = B + C (depends on both)
    cg::Graph graph("diamond");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &B, 2.0, A);
        cg::permute("ji <- ij", 0.0, &C, 1.0, A);
        cg::axpy(1.0, B, &D);
        cg::axpy(1.0, C, &D);
    }

    // Reference
    auto D_ref = create_zero_tensor<double>("D_ref", 4, 4);
    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            D_ref(ii, jj) = 2.0 * A(ii, jj) + A(jj, ii);

    cg::DataflowExecutor df;
    graph.execute(df);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE_THAT(D(ii, jj), Catch::Matchers::WithinRel(D_ref(ii, jj), 1e-12));
}

TEST_CASE("DataflowExecutor - empty graph", "[ComputeGraph][Executor]") {
    cg::Graph            graph("empty");
    cg::DataflowExecutor df;
    graph.execute(df); // Should not crash
    REQUIRE(graph.num_nodes() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Empty and single-node coverage
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SequentialExecutor - empty graph", "[ComputeGraph][Executor]") {
    cg::Graph              graph("seq_empty");
    cg::SequentialExecutor seq;
    REQUIRE_NOTHROW(graph.execute(seq));
}

TEST_CASE("OpenMPExecutor - empty graph", "[ComputeGraph][Executor]") {
    cg::Graph          graph("omp_empty");
    cg::OpenMPExecutor omp;
    REQUIRE_NOTHROW(graph.execute(omp));
}

TEST_CASE("SequentialExecutor - single node", "[ComputeGraph][Executor]") {
    auto A     = create_random_tensor<double>("A", 3, 3);
    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);

    cg::Graph graph("seq_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }
    cg::SequentialExecutor seq;
    graph.execute(seq);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(A(ii, jj) - A_ref(ii, jj)) < 1e-12);
}

TEST_CASE("OpenMPExecutor - single node", "[ComputeGraph][Executor]") {
    auto A     = create_random_tensor<double>("A", 3, 3);
    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);

    cg::Graph graph("omp_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }
    cg::OpenMPExecutor omp;
    graph.execute(omp);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(A(ii, jj) - A_ref(ii, jj)) < 1e-12);
}

TEST_CASE("DataflowExecutor - single node", "[ComputeGraph][Executor]") {
    auto A     = create_random_tensor<double>("A", 3, 3);
    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);

    cg::Graph graph("df_single");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }
    cg::DataflowExecutor df;
    graph.execute(df);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(A(ii, jj) - A_ref(ii, jj)) < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Replay (execute twice on same graph)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SequentialExecutor - replay", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("seq_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &B); // B += A each execute
    }
    cg::SequentialExecutor seq;
    graph.execute(seq);
    graph.execute(seq);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(B(ii, jj) - 2.0 * A(ii, jj)) < 1e-12);
}

TEST_CASE("OpenMPExecutor - replay", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("omp_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &B);
    }
    cg::OpenMPExecutor omp;
    graph.execute(omp);
    graph.execute(omp);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(B(ii, jj) - 2.0 * A(ii, jj)) < 1e-12);
}

TEST_CASE("DataflowExecutor - replay", "[ComputeGraph][Executor]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("df_replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &B);
    }
    cg::DataflowExecutor df;
    graph.execute(df);
    graph.execute(df);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE(std::abs(B(ii, jj) - 2.0 * A(ii, jj)) < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DiskRead/DiskWrite with mock I/O
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SequentialExecutor - DiskRead mock", "[ComputeGraph][Executor][IO]") {
    auto src = create_random_tensor<double>("src", 4, 4);
    auto dst = create_zero_tensor<double>("dst", 4, 4);

    cg::Graph graph("seq_io");
    {
        cg::CaptureGuard const guard(graph);
        cg::read("mock load", "fake.h5", "/data", &dst, [&]() { std::memcpy(dst.data(), src.data(), 16 * sizeof(double)); });
    }

    cg::SequentialExecutor seq;
    graph.execute(seq);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(dst(ii, jj) == src(ii, jj));
}

TEST_CASE("OpenMPExecutor - DiskRead mock", "[ComputeGraph][Executor][IO]") {
    auto src = create_random_tensor<double>("src", 4, 4);
    auto dst = create_zero_tensor<double>("dst", 4, 4);

    cg::Graph graph("omp_io");
    {
        cg::CaptureGuard const guard(graph);
        cg::read("mock load", "fake.h5", "/data", &dst, [&]() { std::memcpy(dst.data(), src.data(), 16 * sizeof(double)); });
    }

    cg::OpenMPExecutor omp;
    graph.execute(omp);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(dst(ii, jj) == src(ii, jj));
}

TEST_CASE("DataflowExecutor - DiskRead mock", "[ComputeGraph][Executor][IO]") {
    auto src = create_random_tensor<double>("src", 4, 4);
    auto dst = create_zero_tensor<double>("dst", 4, 4);

    cg::Graph graph("df_io");
    {
        cg::CaptureGuard const guard(graph);
        cg::read("mock load", "fake.h5", "/data", &dst, [&]() { std::memcpy(dst.data(), src.data(), 16 * sizeof(double)); });
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(dst(ii, jj) == src(ii, jj));
}

TEST_CASE("DiskRead then compute - correct ordering", "[ComputeGraph][Executor][IO]") {
    auto src    = create_random_tensor<double>("src", 4, 4);
    auto data   = create_zero_tensor<double>("data", 4, 4);
    auto result = create_zero_tensor<double>("result", 4, 4);

    cg::Graph graph("io_then_compute");
    {
        cg::CaptureGuard const guard(graph);
        cg::read("load data", "fake.h5", "/data", &data, [&]() { std::memcpy(data.data(), src.data(), 16 * sizeof(double)); });
        cg::scale(2.0, &data);
        cg::axpy(1.0, data, &result);
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(std::abs(result(ii, jj) - 2.0 * src(ii, jj)) < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Async I/O with DataflowExecutor
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DataflowExecutor - async read correctness", "[ComputeGraph][Executor][IO]") {
    auto data   = create_zero_tensor<double>("data", 4, 4);
    auto result = create_zero_tensor<double>("result", 4, 4);

    cg::Graph graph("async_read");
    {
        cg::CaptureGuard const guard(graph);

        cg::read_async(
            "async load", "fake.h5", "/data", &data,
            /*start*/ [&]() { /* Begin "async" read, in mock, just set a flag */ },
            /*finish*/
            [&]() {
                // Fill data with known values
                for (size_t idx = 0; idx < data.size(); idx++)
                    data.data()[idx] = static_cast<double>(idx + 1);
            },
            /*sync*/
            [&]() {
                for (size_t idx = 0; idx < data.size(); idx++)
                    data.data()[idx] = static_cast<double>(idx + 1);
            });

        cg::axpy(1.0, data, &result);
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    // Verify data was loaded correctly
    for (size_t idx = 0; idx < 16; idx++)
        REQUIRE(result.data()[idx] == static_cast<double>(idx + 1));
}

TEST_CASE("SequentialExecutor - async read falls back to sync", "[ComputeGraph][Executor][IO]") {
    auto data = create_zero_tensor<double>("data", 4, 4);

    std::atomic<bool> sync_called{false};
    std::atomic<bool> start_called{false};

    cg::Graph graph("async_fallback");
    {
        cg::CaptureGuard const guard(graph);

        cg::read_async(
            "async load", "fake.h5", "/data", &data,
            /*start*/ [&]() { start_called = true; },
            /*finish*/ [&]() {},
            /*sync*/
            [&]() {
                sync_called = true;
                for (size_t idx = 0; idx < data.size(); idx++)
                    data.data()[idx] = 42.0;
            });
    }

    // SequentialExecutor should call the sync lambda, not async_start/finish
    cg::SequentialExecutor seq;
    graph.execute(seq);

    CHECK(sync_called);
    CHECK_FALSE(start_called); // async_start is NOT called by Sequential
    CHECK(data(0, 0) == 42.0);
}

TEST_CASE("DataflowExecutor - async read overlap with compute", "[ComputeGraph][Executor][IO]") {
    // Verify that async I/O can overlap with independent compute.
    // The read takes 20ms (simulated). An independent scale runs concurrently.
    auto data   = create_zero_tensor<double>("data", 4, 4);
    auto indep  = create_random_tensor<double>("indep", 4, 4);
    auto result = create_zero_tensor<double>("result", 4, 4);

    std::atomic<bool> start_done{false};
    std::atomic<bool> finish_done{false};

    cg::Graph graph("async_overlap");
    {
        cg::CaptureGuard const guard(graph);

        cg::read_async(
            "slow load", "fake.h5", "/data", &data,
            /*start*/
            [&]() {
                // Simulate slow I/O start
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                start_done = true;
            },
            /*finish*/
            [&]() {
                finish_done = true;
                for (size_t idx = 0; idx < data.size(); idx++)
                    data.data()[idx] = 1.0;
            },
            /*sync*/
            [&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                start_done  = true;
                finish_done = true;
                for (size_t idx = 0; idx < data.size(); idx++)
                    data.data()[idx] = 1.0;
            });

        // Independent of data, can overlap with the async read
        cg::scale(3.0, &indep);

        // Depends on data, must wait for finish
        cg::axpy(1.0, data, &result);
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    CHECK(start_done);
    CHECK(finish_done);

    // Verify correctness
    for (size_t idx = 0; idx < 16; idx++)
        CHECK(result.data()[idx] == 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// IOPrefetch pass
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("IOPrefetch - moves DiskRead to beginning", "[ComputeGraph][Passes][IO]") {
    auto A    = create_random_tensor<double>("A", 4, 4);
    auto data = create_zero_tensor<double>("data", 4, 4);
    auto C    = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("prefetch_test");
    {
        cg::CaptureGuard const guard(graph);
        // Compute first, then read, read has no dependency on compute
        cg::scale(2.0, &A);
        cg::scale(3.0, &A);
        cg::read("load data", "fake.h5", "/data", &data, [&]() { std::memcpy(data.data(), A.data(), 16 * sizeof(double)); });
        cg::axpy(1.0, data, &C);
    }

    // Before prefetch: DiskRead is at position 2 (after both scales)
    auto  &nodes           = graph.nodes();
    size_t read_pos_before = 0;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (nodes[idx].kind == cg::OpKind::DiskRead) {
            read_pos_before = idx;
            break;
        }
    }
    CHECK(read_pos_before >= 1); // After at least the first scale

    auto [modified, pass] = graph.apply<cg::passes::IOPrefetch>();

    CHECK(modified);
    CHECK(pass.num_prefetched() >= 1);

    // After prefetch: DiskRead should be at position 0 (no predecessors)
    size_t read_pos_after = 0;
    for (size_t idx = 0; idx < graph.nodes().size(); idx++) {
        if (graph.nodes()[idx].kind == cg::OpKind::DiskRead) {
            read_pos_after = idx;
            break;
        }
    }
    CHECK(read_pos_after < read_pos_before);

    // Graph should still execute correctly
    graph.execute();
}

TEST_CASE("IOPrefetch - respects dependencies", "[ComputeGraph][Passes][IO]") {
    auto A    = create_random_tensor<double>("A", 4, 4);
    auto data = create_zero_tensor<double>("data", 4, 4);

    cg::Graph graph("prefetch_deps");
    {
        cg::CaptureGuard const guard(graph);
        // Scale A first, then read data (no dependency between them)
        cg::scale(2.0, &A);
        cg::read("load", "fake.h5", "/data", &data, [&]() {});
    }

    // DiskRead has no inputs → should move to position 0
    auto [modified, pass] = graph.apply<cg::passes::IOPrefetch>();
    CHECK(modified);
    CHECK(graph.nodes()[0].kind == cg::OpKind::DiskRead);
}

TEST_CASE("IOPrefetch - no DiskRead is no-op", "[ComputeGraph][Passes][IO]") {
    auto A = create_random_tensor<double>("A", 3, 3);

    cg::Graph graph("no_prefetch");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    auto [modified, pass] = graph.apply<cg::passes::IOPrefetch>();
    CHECK_FALSE(modified);
    CHECK(pass.num_prefetched() == 0);
}

TEST_CASE("IOPrefetch - empty graph", "[ComputeGraph][Passes][IO]") {
    cg::Graph graph("empty");

    auto [modified, pass] = graph.apply<cg::passes::IOPrefetch>();
    CHECK_FALSE(modified);
}

TEST_CASE("IOPrefetch - hoists a loop-invariant DiskRead out of the loop body", "[ComputeGraph][Passes][IO][Loop]") {
    // A DiskRead whose destination is read-only in the body produces the
    // same data every iteration. Hoisting it before the loop means the file
    // is read ONCE, not once per iteration. The read executor bumps a
    // counter so we can assert exactly that.
    constexpr size_t N          = 5;
    auto             data       = create_zero_tensor<double>("data", 4, 4); // eager
    auto             acc        = create_zero_tensor<double>("acc", 4, 4);  // eager
    int              read_count = 0;

    cg::Graph g("io_loop");
    auto     &body = g.add_loop("iter", N, [](size_t it) { return it + 1 < N; });
    {
        cg::CaptureGuard const guard(body);
        cg::read("load", "fake.h5", "/d", &data, [&]() {
            read_count++;
            for (size_t k = 0; k < data.size(); ++k)
                data.data()[k] = 1.0;
        });
        cg::axpy(1.0, data, &acc); // acc += data (data is read-only here)
    }

    // Apply IOPrefetch in isolation so the test exercises this pass's loop
    // handling directly, not the full pipeline's interactions (e.g.
    // LoopInvariantHoisting also moves loop-invariant nodes).
    cg::passes::IOPrefetch ioprefetch;
    bool const             modified = ioprefetch.run(g);
    CHECK(modified);
    // The DiskRead must be gone from the body (hoisted to the parent).
    size_t body_reads = 0;
    for (auto const &n : body.nodes()) {
        if (n.kind == cg::OpKind::DiskRead)
            body_reads++;
    }
    CHECK(body_reads == 0);

    g.execute();

    // Read once (hoisted), not N times.
    CHECK(read_count == 1);
    // acc = N * data (= N * ones).
    for (size_t k = 0; k < acc.size(); ++k) {
        CHECK(acc.data()[k] == Catch::Approx(static_cast<double>(N)));
    }
}

TEST_CASE("IOPrefetch - does NOT hoist when the destination is overwritten in the body", "[ComputeGraph][Passes][IO][Loop]") {
    // The destination is re-read every iteration and then OVERWRITTEN by an
    // einsum (a second value-writer the optimizer can't absorb away). The
    // read must stay in the body, hoisting would freeze stale data. The
    // read counter must show one read per iteration.
    constexpr size_t N          = 4;
    auto             B          = create_random_tensor<double>("B", 4, 4);
    auto             data       = create_zero_tensor<double>("data", 4, 4);
    auto             acc        = create_zero_tensor<double>("acc", 4, 4);
    int              read_count = 0;

    cg::Graph g("io_loop_mutated");
    auto     &body = g.add_loop("iter", N, [](size_t it) { return it + 1 < N; });
    {
        cg::CaptureGuard const guard(body);
        cg::read("load", "fake.h5", "/d", &data, [&]() {
            read_count++;
            for (size_t k = 0; k < data.size(); ++k)
                data.data()[k] = 1.0;
        });
        cg::einsum("ik;kj->ij", &data, B, B); // overwrites data, second writer
        cg::axpy(1.0, data, &acc);
    }

    // IOPrefetch in isolation (see note above).
    cg::passes::IOPrefetch ioprefetch;
    ioprefetch.run(g);

    size_t body_reads = 0;
    for (auto const &n : body.nodes()) {
        if (n.kind == cg::OpKind::DiskRead)
            body_reads++;
    }
    CHECK(body_reads == 1); // stayed in the body (data has two writers)

    g.execute();
    CHECK(read_count == static_cast<int>(N)); // read every iteration
}

TEST_CASE("IOPrefetch - hoists a DiskRead out of a nested loop", "[ComputeGraph][Passes][IO][Loop]") {
    // Read in an inner loop body, read-only, must be hoisted all the way
    // out, past the outer loop, and read exactly once.
    constexpr size_t No         = 2;
    constexpr size_t Ni         = 3;
    auto             data       = create_zero_tensor<double>("data", 3, 3);
    auto             acc        = create_zero_tensor<double>("acc", 3, 3);
    int              read_count = 0;

    cg::Graph g("io_nested");
    auto     &outer = g.add_loop("outer", No, [](size_t it) { return it + 1 < No; });
    auto     &inner = outer.add_loop("inner", Ni, [](size_t it) { return it + 1 < Ni; });
    {
        cg::CaptureGuard const guard(inner);
        cg::read("load", "fake.h5", "/d", &data, [&]() {
            read_count++;
            for (size_t k = 0; k < data.size(); ++k)
                data.data()[k] = 1.0;
        });
        cg::axpy(1.0, data, &acc); // acc += data
    }

    // IOPrefetch in isolation (see note above).
    cg::passes::IOPrefetch ioprefetch;
    ioprefetch.run(g);
    g.execute();

    CHECK(read_count == 1);                 // hoisted past both loops
    for (size_t k = 0; k < acc.size(); ++k) // acc = No*Ni * ones
        CHECK(acc.data()[k] == Catch::Approx(static_cast<double>(No * Ni)));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wide fan-out
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DataflowExecutor - wide fan-out", "[ComputeGraph][Executor]") {
    auto                             A = create_random_tensor<double>("A", 4, 4);
    constexpr size_t                 N = 8;
    std::array<Tensor<double, 2>, N> outputs;
    for (size_t idx = 0; idx < N; idx++)
        outputs[idx] = create_zero_tensor<double>(fmt::format("out_{}", idx), 4, 4);

    auto A_scaled = Tensor<double, 2>(A);
    linear_algebra::scale(3.0, &A_scaled);

    cg::Graph graph("fanout");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &A); // 1 producer
        for (size_t idx = 0; idx < N; idx++) {
            cg::axpy(1.0, A, &outputs[idx]); // N consumers
        }
    }

    cg::DataflowExecutor df;
    graph.execute(df);

    for (size_t idx = 0; idx < N; idx++) {
        for (size_t r = 0; r < 4; r++)
            for (size_t c = 0; c < 4; c++)
                REQUIRE(std::abs(outputs[idx](r, c) - A_scaled(r, c)) < 1e-12);
    }
}
