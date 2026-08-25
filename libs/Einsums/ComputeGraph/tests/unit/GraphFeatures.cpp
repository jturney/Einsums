//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Tests for Graph features: to_json, move semantics, empty graph, PassManager default.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <sstream>
#include <string>

#include <Einsums/Testing.hpp>

using TensorId        = einsums::compute_graph::TensorId;
namespace packed_gemm = einsums::packed_gemm;

using namespace einsums;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("Graph - to_json produces valid structure", "[ComputeGraph][JSON]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("json_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    std::string const json = graph.to_json();

    // Basic structure checks
    REQUIRE(json.find("\"name\":\"json_test\"") != std::string::npos);
    REQUIRE(json.find("\"tensors\":[") != std::string::npos);
    REQUIRE(json.find("\"nodes\":[") != std::string::npos);
    REQUIRE(json.find("\"edges\":[") != std::string::npos);

    // Should have tensor names
    REQUIRE(json.find("\"A\"") != std::string::npos);
    REQUIRE(json.find("\"B\"") != std::string::npos);
    REQUIRE(json.find("\"C\"") != std::string::npos);

    // Should have the einsum node
    REQUIRE(json.find("\"Einsum\"") != std::string::npos);
}

TEST_CASE("Graph - to_json with edges before execute", "[ComputeGraph][JSON]") {
    // to_json should produce edges even before execute() is called
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("json_edges");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
    }

    // Don't call execute()
    std::string const json = graph.to_json();

    // Should have edges (scale writes A, permute reads A)
    REQUIRE(json.find("\"edges\":[{") != std::string::npos);
    REQUIRE(json.find("\"from\"") != std::string::npos);
    REQUIRE(json.find("\"to\"") != std::string::npos);
}

TEST_CASE("Graph - to_json with timing after execute", "[ComputeGraph][JSON]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("json_timing");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
    }

    graph.execute();
    std::string const json = graph.to_json();

    REQUIRE(json.find("\"timing_ms\"") != std::string::npos);
}

TEST_CASE("Graph - to_json empty graph", "[ComputeGraph][JSON]") {
    cg::Graph const   graph("empty");
    std::string const json = graph.to_json();

    REQUIRE(json.find("\"name\":\"empty\"") != std::string::npos);
    REQUIRE(json.find("\"tensors\":[]") != std::string::npos);
    REQUIRE(json.find("\"nodes\":[]") != std::string::npos);
    REQUIRE(json.find("\"edges\":[]") != std::string::npos);
}

TEST_CASE("Graph - move constructor", "[ComputeGraph][Move]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    auto C_ref = create_zero_tensor<double>("C_ref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph1("original");
    {
        cg::CaptureGuard const guard(graph1);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Move to graph2
    cg::Graph graph2(std::move(graph1));

    REQUIRE(graph2.num_nodes() == 1);
    REQUIRE(graph2.name() == "original");

    graph2.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Graph - move assignment", "[ComputeGraph][Move]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph1("src");
    {
        cg::CaptureGuard const guard(graph1);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    cg::Graph graph2("dest");
    graph2 = std::move(graph1);

    REQUIRE(graph2.num_nodes() == 1);
    REQUIRE(graph2.name() == "src");

    graph2.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Graph - execute empty graph", "[ComputeGraph]") {
    cg::Graph graph("empty");
    // Should not crash
    graph.execute();
    REQUIRE(graph.num_nodes() == 0);
}

TEST_CASE("Graph - print_dot output", "[ComputeGraph]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("dot_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
    }

    std::ostringstream oss;
    graph.print_dot(oss);
    std::string const dot = oss.str();

    REQUIRE(dot.find("digraph") != std::string::npos);
    REQUIRE(dot.find("shape=box") != std::string::npos);
    REQUIRE(dot.find("shape=ellipse") != std::string::npos);
}

TEST_CASE("Graph - print_summary output", "[ComputeGraph]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);

    cg::Graph graph("summary_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
    }

    std::ostringstream oss;
    graph.print_summary(oss);
    std::string const summary = oss.str();

    REQUIRE(summary.find("summary_test") != std::string::npos);
    REQUIRE(summary.find("1 nodes") != std::string::npos);
    REQUIRE(summary.find("Scale") != std::string::npos);
}

TEST_CASE("Graph - PassManager default end-to-end", "[ComputeGraph][PassManager]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 5, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("passmanager");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    // Apply all default passes - should not crash and result should still be correct
    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Graph - timing report populated after execute", "[ComputeGraph][Timing]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);

    cg::Graph graph("timing");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
        cg::scale(2.0, &B);
    }

    graph.execute();

    auto const &report = graph.timing_report();
    REQUIRE(report.size() == 2);
    REQUIRE(report[0].duration_ms >= 0.0);
    REQUIRE(report[1].duration_ms >= 0.0);
}

// ── Runtime dispatch helper tests ────────────────────────────────────────────

TEST_CASE("Graph - create_tensor_dynamic", "[ComputeGraph]") {
    cg::Graph graph("dynamic_tensor");
    auto      result = graph.create_tensor_dynamic("T", packed_gemm::ScalarType::Float64, {3, 4});

    REQUIRE(result.has_value());
    auto [id, ptr] = result.value();
    REQUIRE(ptr != nullptr);

    auto const &h = graph.tensor(id);
    REQUIRE(h.name == "T");
    REQUIRE(h.rank == 2);
    REQUIRE(h.dims == std::vector<size_t>{3, 4});
    REQUIRE(h.is_intermediate);
}

TEST_CASE("Graph - create_tensor_dynamic error on empty dims", "[ComputeGraph]") {
    cg::Graph graph("dynamic_error");
    auto      result = graph.create_tensor_dynamic("T", packed_gemm::ScalarType::Float64, {});

    CHECK_FALSE(result.has_value());
    CHECK(result.error().kind == cg::GraphError::Kind::Type);
}

TEST_CASE("Graph - make_axpy_executor", "[ComputeGraph]") {
    cg::Graph graph("axpy_test");
    auto     &A = graph.create_tensor<double, 2>("A", 3, 3);
    auto     &B = graph.create_tensor<double, 2>("B", 3, 3);

    // Fill A with ones
    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            A(ii, jj) = 1.0;
    B.zero();

    // Find tensor IDs
    TensorId a_id = 0, b_id = 0;
    for (auto const &[id, h] : graph.tensors_map()) {
        if (h.name == "A")
            a_id = id;
        if (h.name == "B")
            b_id = id;
    }

    auto executor = graph.make_axpy_executor(2.5, a_id, b_id);
    executor();

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            REQUIRE_THAT(B(ii, jj), Catch::Matchers::WithinRel(2.5, 1e-12));
}

TEST_CASE("Graph - make_zero_executor", "[ComputeGraph]") {
    cg::Graph graph("zero_test");
    auto     &A = graph.create_tensor<double, 1>("A", 10);
    for (size_t ii = 0; ii < 10; ii++)
        A(ii) = 99.0;

    TensorId a_id = 0;
    for (auto const &[id, h] : graph.tensors_map()) {
        if (h.name == "A")
            a_id = id;
    }

    auto executor = graph.make_zero_executor(a_id);
    executor();

    for (size_t ii = 0; ii < 10; ii++)
        REQUIRE(A(ii) == 0.0);
}

TEST_CASE("Graph - make_copy_executor", "[ComputeGraph]") {
    cg::Graph graph("copy_test");
    auto     &A = graph.create_tensor<double, 2>("A", 4, 4);
    auto     &B = graph.create_tensor<double, 2>("B", 4, 4);

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            A(ii, jj) = static_cast<double>(ii * 4 + jj);
    B.zero();

    TensorId a_id = 0, b_id = 0;
    for (auto const &[id, h] : graph.tensors_map()) {
        if (h.name == "A")
            a_id = id;
        if (h.name == "B")
            b_id = id;
    }

    auto executor = graph.make_copy_executor(a_id, b_id);
    executor();

    for (size_t ii = 0; ii < 4; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(B(ii, jj) == A(ii, jj));
}

// ─── Shape inference ────────────────────────────────────────────────────────

TEST_CASE("Shape inference - valid graph passes", "[ComputeGraph][ShapeInference]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("valid_shapes");
    REQUIRE_NOTHROW([&]() {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }());
}

TEST_CASE("Shape inference - string einsum valid", "[ComputeGraph][ShapeInference]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("valid_string");
    REQUIRE_NOTHROW([&]() {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }());
}

// ─── Execution caching ──────────────────────────────────────────────────────

TEST_CASE("Execution caching - replay skips validation", "[ComputeGraph][Caching]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cache_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    graph.execute();
    auto C_first = Tensor<double, 2>(C);

    C.zero();
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_first(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("Execution caching - apply resets cache", "[ComputeGraph][Caching]") {
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Graph graph("cache_reset");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, A, B);
    }

    graph.execute();

    graph.apply<cg::passes::ScaleAbsorption>();

    C.zero();
    REQUIRE_NOTHROW(graph.execute());
}

// ─── Execution with annotations ─────────────────────────────────────────────

TEST_CASE("Graph - execute with annotations", "[ComputeGraph][Profiler]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("profiled_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
    }

    REQUIRE_NOTHROW(graph.execute());

    auto C_ref = create_zero_tensor<double>("Cref", 4, 5);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);
    linear_algebra::scale(2.0, &C_ref);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("scratch - deferred, intermediate, and fully pass-managed", "[ComputeGraph][Scratch]") {
    // scratch<T,Rank>() is the one-call managed intermediate: deferred until
    // execution and visible to the memory passes (FreeInsertion frees it,
    // MemoryPlanning's arena may host it). 400x400 clears the FreeInsertion
    // min-bytes threshold so the whole lifecycle engages.
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("scratch_managed");
    auto     &tmp = graph.scratch<double, 2>("tmp", N, N);
    CHECK_FALSE(tmp.is_materialized()); // deferred: no allocation yet

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &out, tmp, B);
    }
    CHECK_FALSE(tmp.is_materialized()); // still nothing allocated after capture

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);

    graph.execute();
    // Freed after its last consumer (released back to deferred state or
    // parked by the arena) - either way not holding its own live buffer.
    // The result must be right regardless:
    auto tmp_ref = create_zero_tensor<double>("tmpref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &tmp_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, tmp_ref, Indices{k, j}, B);
    for (size_t ii = 0; ii < N; ii += 37) {
        for (size_t jj = 0; jj < N; jj += 41) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-8);
        }
    }

    out.zero();
    graph.execute(); // replay through the managed lifecycle
    for (size_t ii = 0; ii < N; ii += 37) {
        for (size_t jj = 0; jj < N; jj += 41) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("scratch_zero - zeroed at materialization", "[ComputeGraph][Scratch]") {
    auto A   = create_random_tensor<double>("A", 6, 6);
    auto out = create_zero_tensor<double>("out", 6, 6);

    cg::Graph graph("scratch_zero");
    auto     &acc = graph.scratch_zero<double, 2>("acc", 6, 6);
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &acc); // accumulate into zero-initialized scratch
        cg::axpy(1.0, acc, &out);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            REQUIRE(std::abs(out(ii, jj) - A(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("optimize + explain - one-call pipeline with a readable report", "[ComputeGraph][Optimize]") {
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             out = create_zero_tensor<double>("out", N, N);

    cg::Graph graph("optimize_api");
    auto     &tmp = graph.scratch<double, 2>("tmp", N, N);
    auto     &dup = graph.scratch<double, 2>("dup", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &dup, A, B); // CSE fodder
        cg::einsum("ik;kj->ij", &out, tmp, dup);
    }

    CHECK(graph.explain().empty()); // nothing yet

    bool const modified = graph.optimize();
    CHECK(modified);

    auto const &report = graph.explain();
    INFO(report);
    CHECK_FALSE(report.empty());
    CHECK(report.find("optimize(O2)") != std::string::npos);
    CHECK(report.find("node(s)") != std::string::npos);
    // The memory passes engaged on the >1MB scratch intermediates.
    CHECK(report.find("FreeInsertion") != std::string::npos);

    graph.execute();

    auto tmp_ref = create_zero_tensor<double>("tmpref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &tmp_ref, Indices{i, k}, A, Indices{k, j}, B);
    auto out_ref = create_zero_tensor<double>("OUTref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, tmp_ref, Indices{k, j}, tmp_ref);
    for (size_t ii = 0; ii < N; ii += 41) {
        for (size_t jj = 0; jj < N; jj += 37) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("optimize levels - O0 is a no-op, O1 cleans up only", "[ComputeGraph][Optimize]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("optimize_levels");
    // Graph-owned duplicates: CSE never elides writes to user-visible tensors.
    auto &tmp = graph.create_tensor<double, 2>("tmp", 4, 4);
    auto &dup = graph.create_tensor<double, 2>("dup", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &dup, A, A); // duplicate for CSE
        cg::einsum("ik;kj->ij", &out, tmp, dup);
    }
    size_t const before = graph.num_nodes();

    CHECK_FALSE(graph.optimize(cg::OptLevel::O0));
    CHECK(graph.num_nodes() == before);

    CHECK(graph.optimize(cg::OptLevel::O1)); // CSE folds the duplicate
    CHECK(graph.num_nodes() < before);
    CHECK(graph.explain().find("optimize(O1)") != std::string::npos);

    graph.execute();
    auto tmp_ref = create_zero_tensor<double>("tmpref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &tmp_ref, Indices{i, k}, A, Indices{k, j}, A);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, tmp_ref, Indices{k, j}, tmp_ref);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("deferred tensor without Materialization - actionable execute error", "[ComputeGraph][Validation]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("deferred_misuse");
    auto     &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &out, tmp, A);
    }

    // Executing with tmp still deferred used to be a segfault (null data
    // pointer inside the GEMM). Now it names the tensor and the fix.
    try {
        graph.execute();
        FAIL("expected execute() to reject the unmaterialized deferred tensor");
    } catch (std::runtime_error const &e) {
        std::string const msg = e.what();
        REQUIRE(msg.find("tmp") != std::string::npos);
        REQUIRE(msg.find("still deferred") != std::string::npos);
        REQUIRE(msg.find("graph.optimize()") != std::string::npos);
    }

    // Every optimize level >= O1 includes Materialization, so following the
    // error's advice makes the graph executable.
    graph.optimize(cg::OptLevel::O1);
    REQUIRE_NOTHROW(graph.execute());

    auto tmp_ref = create_zero_tensor<double>("tmpref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &tmp_ref, Indices{i, k}, A, Indices{k, j}, A);
    auto out_ref = create_zero_tensor<double>("OUTref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &out_ref, Indices{i, k}, tmp_ref, Indices{k, j}, A);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(out(ii, jj) - out_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("deferred tensor materialized by hand - no false positive", "[ComputeGraph][Validation]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto out = create_zero_tensor<double>("out", 4, 4);

    cg::Graph graph("deferred_manual");
    auto     &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &out, tmp, A);
    }

    // The handle's alloc_state snapshot still says Deferred; the live
    // is_materialized query must win so direct materialize() keeps working.
    tmp.materialize();
    tmp.zero();
    REQUIRE_NOTHROW(graph.execute());
}

// ── Graph::make_einsum_node ────────────────────────────────────────────────
// The factory a pass uses to synthesize a contraction. The point is that the
// node it produces behaves like a captured one rather than an opaque closure:
// it carries a real EinsumDescriptor other passes can read, and its executor
// reads the LIVE params/indices so a pass that edits the descriptor is honored
// instead of silently ignored.
namespace {
einsums::compute_graph::ParsedEinsumSpec matmul_spec() {
    einsums::compute_graph::ParsedEinsumSpec s;
    s.c_indices = {"i", "j"};
    s.a_indices = {"i", "k"};
    s.b_indices = {"k", "j"};
    s.raw       = "i,j <- i,k ; k,j";
    return s;
}
} // namespace

TEST_CASE("make_einsum_node - produces a real Einsum node with a descriptor", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto A_t = create_random_tensor<double>("A", 3, 4);
    auto B_t = create_random_tensor<double>("B", 4, 2);

    RuntimeTensor<double> A(A_t), B(B_t);
    RuntimeTensor<double> C("C", std::vector<size_t>{3, 2});
    C.zero();

    cg::Graph graph("mk_einsum");
    // register_tensor via the capture-free path: slots come from the handles.
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    auto node = graph.make_einsum_node(a_id, b_id, c_id, matmul_spec(), /*c_pf=*/0.0, /*ab_pf=*/2.0);

    CHECK(node.kind == cg::OpKind::Einsum);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    CHECK(desc->params != nullptr);
    CHECK(desc->indices != nullptr);
    CHECK(desc->spec.link_indices == std::vector<std::string>{"k"});
    // c_pf == 0 is a pure overwrite, so the output must NOT be listed as an input.
    CHECK(node.inputs.size() == 2);
    CHECK(node.outputs == std::vector<cg::TensorId>{c_id});

    graph.add_node(std::move(node));
    graph.execute();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double ref = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                ref += A_t(i, k) * B_t(k, j);
            }
            std::vector<ptrdiff_t> const idx{static_cast<ptrdiff_t>(i), static_cast<ptrdiff_t>(j)};
            CHECK(C(idx) == Catch::Approx(2.0 * ref));
        }
    }
}

TEST_CASE("make_einsum_node - executor honors a later prefactor edit", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto A_t = create_random_tensor<double>("A", 3, 4);
    auto B_t = create_random_tensor<double>("B", 4, 2);

    RuntimeTensor<double> A(A_t), B(B_t);
    RuntimeTensor<double> C("C", std::vector<size_t>{3, 2});
    C.zero();

    cg::Graph  graph("mk_einsum_live");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    graph.add_node(graph.make_einsum_node(a_id, b_id, c_id, matmul_spec(), 0.0, 1.0));
    graph.execute();
    double const first = C(std::vector<ptrdiff_t>{0, 0});

    // A pass folding a scale writes ab_pf through the shared params handle. A
    // baked-closure node would ignore this; a first-class one must not.
    auto *desc = std::get_if<cg::EinsumDescriptor>(&graph.nodes()[0].op_data);
    REQUIRE(desc != nullptr);
    desc->ab_prefactor  = 3.0;
    desc->params->ab_pf = 3.0;

    C.zero();
    graph.execute();
    CHECK(C(std::vector<ptrdiff_t>{0, 0}) == Catch::Approx(3.0 * first));
}

TEST_CASE("make_einsum_node - accumulating form lists its output as an input", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto                  A_t = create_random_tensor<double>("A", 2, 2);
    RuntimeTensor<double> A(A_t), B(A_t);
    RuntimeTensor<double> C("C", std::vector<size_t>{2, 2});
    C.zero();

    cg::Graph  graph("mk_einsum_rmw");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    auto node = graph.make_einsum_node(a_id, b_id, c_id, matmul_spec(), /*c_pf=*/1.0, /*ab_pf=*/1.0);
    CHECK(node.inputs.size() == 3);
    CHECK(std::ranges::find(node.inputs, c_id) != node.inputs.end());
}

TEST_CASE("make_einsum_node - builds a GemmHint when the shapes qualify", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto A_t = create_random_tensor<double>("A", 3, 4);
    auto B_t = create_random_tensor<double>("B", 4, 2);

    RuntimeTensor<double> A(A_t), B(B_t);
    RuntimeTensor<double> C("C", std::vector<size_t>{3, 2});
    C.zero();

    cg::Graph  graph("mk_einsum_hint");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    auto        node = graph.make_einsum_node(a_id, b_id, c_id, matmul_spec(), 0.0, 1.0);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    REQUIRE(desc->gemm_hint != nullptr); // "i,j <- i,k ; k,j" is a plain GEMM

    auto const &h = *desc->gemm_hint;
    CHECK(h.scalar == cg::BlasScalar::Double);
    // link is "k": A's k is axis 1 (not 0) so no transpose; B's k is axis 0 (not 1) likewise.
    CHECK(h.trans_a == 'N');
    CHECK(h.trans_b == 'N');
    CHECK(h.m == 3);
    CHECK(h.n == 2);
    CHECK(h.k == 4);

    // The operands are recorded as ids, so the batched executor can resolve
    // them through the graph's slots at execute time; the leading dimensions
    // beside them are the planning snapshot the batching pass groups on.
    CHECK(h.a.id == a_id);
    CHECK(h.b.id == b_id);
    CHECK(h.c.id == c_id);
    CHECK(h.a.leading_dim > 0);
    CHECK(h.c.leading_dim > 0);
}

TEST_CASE("make_einsum_node - no GemmHint for a higher-rank contraction", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto A_t = create_random_tensor<double>("A", 2, 3);
    auto B_t = create_random_tensor<double>("B", 2, 4, 3, 3);

    RuntimeTensor<double> A(A_t), B(B_t);
    RuntimeTensor<double> C("C", std::vector<size_t>{4, 3});
    C.zero();

    cg::ParsedEinsumSpec spec;
    spec.c_indices = {"a", "e"};
    spec.a_indices = {"m", "f"};
    spec.b_indices = {"m", "a", "f", "e"};
    spec.raw       = "a,e <- m,f ; m,a,f,e";

    cg::Graph  graph("mk_einsum_nohint");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    auto        node = graph.make_einsum_node(a_id, b_id, c_id, spec, 0.0, 1.0);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    // Rank-4 operand: not a GEMM, so no hint and GEMMBatching leaves it alone.
    CHECK(desc->gemm_hint == nullptr);
}

// Statically typed operands. This was impossible while the executor cast
// tensor_ptr to GeneralRuntimeTensor: a Tensor<T, Rank> would have been type
// confusion. Re-viewing through the handle's rank-erased impl removes the
// restriction, so a pass can synthesize a contraction over typed captures --
// which is what ContractionPlanning needs, since it only restructures typed
// rank-2 chains.
TEST_CASE("make_einsum_node - works on statically typed operands", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 2);
    auto C = create_zero_tensor<double>("C", 3, 2);

    cg::Graph  graph("mk_einsum_typed");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    // Typed captures, not runtime tensors.
    CHECK_FALSE(graph.tensor(a_id).is_runtime);
    REQUIRE(graph.tensor(a_id).impl_fn != nullptr);

    auto        node = graph.make_einsum_node(a_id, b_id, c_id, matmul_spec(), 0.0, 1.0);
    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    CHECK(node.kind == cg::OpKind::Einsum);
    CHECK(desc->gemm_hint != nullptr); // rank-2 typed operands still qualify

    graph.add_node(std::move(node));
    graph.execute();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double ref = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                ref += A(i, k) * B(k, j);
            }
            CHECK(C(i, j) == Catch::Approx(ref));
        }
    }
}

TEST_CASE("make_einsum_node - typed rank-4 operand, no static-rank cast needed", "[ComputeGraph][Graph]") {
    namespace cg = einsums::compute_graph;

    // "a,e <- m,f ; m,a,f,e": the CCSD spin-adaptation shape, on TYPED tensors of
    // two different ranks. One dtype dispatch handles both because the impl is
    // rank-erased.
    auto A = create_random_tensor<double>("A", 2, 3);
    auto B = create_random_tensor<double>("B", 2, 4, 3, 3);
    auto C = create_zero_tensor<double>("C", 4, 3);

    cg::ParsedEinsumSpec spec;
    spec.c_indices = {"a", "e"};
    spec.a_indices = {"m", "f"};
    spec.b_indices = {"m", "a", "f", "e"};
    spec.raw       = "a,e <- m,f ; m,a,f,e";

    cg::Graph  graph("mk_einsum_typed_rank4");
    auto const a_id = graph.register_tensor(cg::make_handle(A, 0));
    auto const b_id = graph.register_tensor(cg::make_handle(B, 0));
    auto const c_id = graph.register_tensor(cg::make_handle(C, 0));

    graph.add_node(graph.make_einsum_node(a_id, b_id, c_id, spec, 0.0, 1.0));
    graph.execute();

    for (size_t a = 0; a < 4; ++a) {
        for (size_t e = 0; e < 3; ++e) {
            double ref = 0.0;
            for (size_t m = 0; m < 2; ++m) {
                for (size_t f = 0; f < 3; ++f) {
                    ref += A(m, f) * B(m, a, f, e);
                }
            }
            CHECK(C(a, e) == Catch::Approx(ref));
        }
    }
}

TEST_CASE("Graph - insert_node_groups keeps two groups at one position in order", "[ComputeGraph]") {
    // A pass replacing a run of ADJACENT nodes shifts every replacement group onto
    // the same post-erase index. Splicing runs in descending position order, so
    // without a tiebreak on equal positions the second group lands ahead of the
    // first. For a producer followed by its consumer that means the consumer runs
    // first and reads unwritten storage.
    cg::Graph graph("insert_ties");

    auto make = [&graph](std::string label) {
        cg::Node nd;
        nd.id      = graph.reserve_node_id();
        nd.kind    = cg::OpKind::Custom;
        nd.label   = std::move(label);
        nd.execute = []() {};
        return nd;
    };

    std::vector<std::pair<std::size_t, std::vector<cg::Node>>> groups;
    groups.emplace_back(0, [&] {
        std::vector<cg::Node> g;
        g.push_back(make("first_a"));
        g.push_back(make("first_b"));
        return g;
    }());
    groups.emplace_back(0, [&] {
        std::vector<cg::Node> g;
        g.push_back(make("second_a"));
        return g;
    }());
    graph.insert_node_groups(std::move(groups));

    REQUIRE(graph.num_nodes() == 3);
    CHECK(graph.nodes()[0].label == "first_a");
    CHECK(graph.nodes()[1].label == "first_b");
    CHECK(graph.nodes()[2].label == "second_a");
}

namespace {

/// A pass defined outside the library, to show that explain() is open to
/// extension. Before OptimizerPass::explain existed, PassManager::explain was a
/// type switch over the built-in passes and a pass like this was invisible to it
/// no matter what it counted.
class CountingUserPass : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CountingUserPass"; }

    bool run(cg::Graph &graph) override {
        _seen = graph.num_nodes();
        return false; // observes only
    }

    void reset_stats() override { _seen = 0; }

    [[nodiscard]] std::vector<std::string> explain() const override {
        if (_seen == 0) {
            return {};
        }
        return {fmt::format("CountingUserPass: saw {} node(s)", _seen)};
    }

  private:
    size_t _seen{0};
};

} // namespace

TEST_CASE("explain - a user-defined pass reports itself", "[ComputeGraph][Optimize]") {
    auto A   = create_random_tensor<double>("A", 8, 6);
    auto B   = create_random_tensor<double>("B", 6, 5);
    auto out = create_zero_tensor<double>("out", 8, 5);

    cg::Graph graph("user_pass_explain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &out, A, B);
    }

    auto            user = std::make_shared<CountingUserPass>();
    cg::PassManager pm;
    pm.add(user);
    pm.run(graph);

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("CountingUserPass: saw") != std::string::npos);
}

TEST_CASE("explain - a pass that did nothing stays silent", "[ComputeGraph][Optimize]") {
    // A quiet report has to mean a quiet pipeline, so a pass with nothing to say
    // contributes no line rather than one reporting zeros.
    auto A   = create_random_tensor<double>("A", 4, 3);
    auto B   = create_random_tensor<double>("B", 3, 2);
    auto out = create_zero_tensor<double>("out", 4, 2);

    cg::Graph graph("quiet_explain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &out, A, B);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>(); // nothing is dead here
    pm.run(graph);

    CHECK(pm.explain() == "  (no optimizations applied)\n");
}
