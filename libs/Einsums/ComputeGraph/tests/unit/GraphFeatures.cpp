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
    auto             OUT = create_zero_tensor<double>("OUT", N, N);

    cg::Graph graph("scratch_managed");
    auto     &tmp = graph.scratch<double, 2>("tmp", N, N);
    CHECK_FALSE(tmp.is_materialized()); // deferred: no allocation yet

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &OUT, tmp, B);
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
    auto OUT_ref = create_zero_tensor<double>("OUTref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT_ref, Indices{i, k}, tmp_ref, Indices{k, j}, B);
    for (size_t ii = 0; ii < N; ii += 37) {
        for (size_t jj = 0; jj < N; jj += 41) {
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-8);
        }
    }

    OUT.zero();
    graph.execute(); // replay through the managed lifecycle
    for (size_t ii = 0; ii < N; ii += 37) {
        for (size_t jj = 0; jj < N; jj += 41) {
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("scratch_zero - zeroed at materialization", "[ComputeGraph][Scratch]") {
    auto A   = create_random_tensor<double>("A", 6, 6);
    auto OUT = create_zero_tensor<double>("OUT", 6, 6);

    cg::Graph graph("scratch_zero");
    auto     &acc = graph.scratch_zero<double, 2>("acc", 6, 6);
    {
        cg::CaptureGuard const guard(graph);
        cg::axpy(1.0, A, &acc); // accumulate into zero-initialized scratch
        cg::axpy(1.0, acc, &OUT);
    }

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    graph.execute();

    for (size_t ii = 0; ii < 6; ii++) {
        for (size_t jj = 0; jj < 6; jj++) {
            REQUIRE(std::abs(OUT(ii, jj) - A(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("optimize + explain - one-call pipeline with a readable report", "[ComputeGraph][Optimize]") {
    constexpr size_t N   = 400;
    auto             A   = create_random_tensor<double>("A", N, N);
    auto             B   = create_random_tensor<double>("B", N, N);
    auto             OUT = create_zero_tensor<double>("OUT", N, N);

    cg::Graph graph("optimize_api");
    auto     &tmp = graph.scratch<double, 2>("tmp", N, N);
    auto     &dup = graph.scratch<double, 2>("dup", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &dup, A, B); // CSE fodder
        cg::einsum("ik;kj->ij", &OUT, tmp, dup);
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
    auto OUT_ref = create_zero_tensor<double>("OUTref", N, N);
    tensor_algebra::einsum(Indices{i, j}, &OUT_ref, Indices{i, k}, tmp_ref, Indices{k, j}, tmp_ref);
    for (size_t ii = 0; ii < N; ii += 41) {
        for (size_t jj = 0; jj < N; jj += 37) {
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-8);
        }
    }
}

TEST_CASE("optimize levels - O0 is a no-op, O1 cleans up only", "[ComputeGraph][Optimize]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto OUT = create_zero_tensor<double>("OUT", 4, 4);

    cg::Graph graph("optimize_levels");
    // Graph-owned duplicates: CSE never elides writes to user-visible tensors.
    auto &tmp = graph.create_tensor<double, 2>("tmp", 4, 4);
    auto &dup = graph.create_tensor<double, 2>("dup", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &dup, A, A); // duplicate for CSE
        cg::einsum("ik;kj->ij", &OUT, tmp, dup);
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
    auto OUT_ref = create_zero_tensor<double>("OUTref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &OUT_ref, Indices{i, k}, tmp_ref, Indices{k, j}, tmp_ref);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("deferred tensor without Materialization - actionable execute error", "[ComputeGraph][Validation]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto OUT = create_zero_tensor<double>("OUT", 4, 4);

    cg::Graph graph("deferred_misuse");
    auto     &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &OUT, tmp, A);
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
    auto OUT_ref = create_zero_tensor<double>("OUTref", 4, 4);
    tensor_algebra::einsum(Indices{i, j}, &OUT_ref, Indices{i, k}, tmp_ref, Indices{k, j}, A);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(OUT(ii, jj) - OUT_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("deferred tensor materialized by hand - no false positive", "[ComputeGraph][Validation]") {
    auto A   = create_random_tensor<double>("A", 4, 4);
    auto OUT = create_zero_tensor<double>("OUT", 4, 4);

    cg::Graph graph("deferred_manual");
    auto     &tmp = graph.scratch<double, 2>("tmp", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &OUT, tmp, A);
    }

    // The handle's alloc_state snapshot still says Deferred; the live
    // is_materialized query must win so direct materialize() keeps working.
    tmp.materialize();
    tmp.zero();
    REQUIRE_NOTHROW(graph.execute());
}
