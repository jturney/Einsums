//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Regression tests for topological sort and dependency ordering.
// These test the bugs found during blueprint development:
// - write-after-read dependencies
// - scale/element_transform as both input AND output
// - diamond DAGs
// - cycle detection

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_permute;

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("Dependency - scale is both input and output", "[ComputeGraph][Dependency]") {
    // scale(A) reads and writes A. If a subsequent permute reads A,
    // the scale must execute first.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);

    // Save original A for reference
    auto A_orig = Tensor<double, 2>(A);

    // Reference: scale A by 2, then copy A -> B via permute
    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(2.0, &A_ref);
    auto B_ref = create_zero_tensor<double>("B_ref", 4, 4);
    reference_permute("ij <- ij", 0.0, &B_ref, 1.0, A_ref);

    // Graph: same operations
    A = Tensor<double, 2>(A_orig); // Reset A
    cg::Graph graph("scale_then_permute");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &A);
        cg::permute("ij <- ij", 0.0, &B, 1.0, A);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(B(ii, jj), Catch::Matchers::WithinRel(B_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Dependency - permute then scale (write-after-read)", "[ComputeGraph][Dependency]") {
    // permute reads A, then scale writes A. The permute must finish
    // before scale modifies A (write-after-read dependency).
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_zero_tensor<double>("B", 4, 4);

    auto A_orig = Tensor<double, 2>(A);

    // Reference: permute A->B, then scale A
    auto B_ref = create_zero_tensor<double>("B_ref", 4, 4);
    reference_permute("ji <- ij", 0.0, &B_ref, 1.0, A);
    linear_algebra::scale(0.5, &A);
    auto A_after_ref = Tensor<double, 2>(A);

    // Reset
    A = Tensor<double, 2>(A_orig);
    B.zero();

    cg::Graph graph("permute_then_scale");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &B, 1.0, A);
        cg::scale(0.5, &A);
    }

    graph.execute();

    // B should have the ORIGINAL A transposed (not scaled)
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(B(ii, jj), Catch::Matchers::WithinRel(B_ref(ii, jj), 1e-12));
        }
    }
    // A should be scaled
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(A(ii, jj), Catch::Matchers::WithinRel(A_after_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Dependency - element_transform is both input and output", "[ComputeGraph][Dependency]") {
    auto A = create_random_tensor<double>("A", 5);
    auto B = create_zero_tensor<double>("B", 5);

    auto A_orig = Tensor<double, 1>(A);

    // Reference: transform A (square each element), then axpy A->B
    auto A_ref = Tensor<double, 1>(A);
    for (size_t i = 0; i < 5; i++) {
        A_ref(i) = A_ref(i) * A_ref(i);
    }
    auto B_ref = Tensor<double, 1>(B);
    linear_algebra::axpy(1.0, A_ref, &B_ref);

    // Reset
    A = Tensor<double, 1>(A_orig);
    B.zero();

    cg::Graph graph("transform_then_axpy");
    {
        cg::CaptureGuard const guard(graph);
        cg::element_transform(&A, [](double v) { return v * v; });
        cg::axpy(1.0, A, &B);
    }

    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        REQUIRE_THAT(B(ii), Catch::Matchers::WithinRel(B_ref(ii), 1e-12));
    }
}

TEST_CASE("Dependency - diamond DAG", "[ComputeGraph][Dependency]") {
    // A -> B and A -> C, then B + C -> D
    // Tests that diamond dependencies are handled correctly.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_zero_tensor<double>("B", 3, 3);
    auto C = create_zero_tensor<double>("C", 3, 3);
    auto D = create_zero_tensor<double>("D", 3, 3);

    // Reference
    auto B_ref = create_zero_tensor<double>("B_ref", 3, 3);
    auto C_ref = create_zero_tensor<double>("C_ref", 3, 3);
    auto D_ref = create_zero_tensor<double>("D_ref", 3, 3);
    reference_permute("ij <- ij", 0.0, &B_ref, 2.0, A);
    reference_permute("ji <- ij", 0.0, &C_ref, 1.0, A);
    // D = B + C
    linear_algebra::axpy(1.0, B_ref, &D_ref);
    linear_algebra::axpy(1.0, C_ref, &D_ref);

    cg::Graph graph("diamond");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ij <- ij", 0.0, &B, 2.0, A); // B = 2*A
        cg::permute("ji <- ij", 0.0, &C, 1.0, A); // C = A^T
        cg::axpy(1.0, B, &D);                     // D += B
        cg::axpy(1.0, C, &D);                     // D += C
    }

    REQUIRE(graph.num_nodes() == 4);
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE_THAT(D(ii, jj), Catch::Matchers::WithinRel(D_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Dependency - multiple writes to same tensor", "[ComputeGraph][Dependency]") {
    // Tests write-after-write: scale A, then overwrite A via permute from B.
    auto A = create_random_tensor<double>("A", 3, 3);
    auto B = create_random_tensor<double>("B", 3, 3);

    auto A_orig = Tensor<double, 2>(A);

    // Reference: scale A, then A = B^T (overwrite)
    auto A_ref = Tensor<double, 2>(A);
    linear_algebra::scale(3.0, &A_ref);
    reference_permute("ij <- ji", 0.0, &A_ref, 1.0, B);

    // Reset
    A = Tensor<double, 2>(A_orig);

    cg::Graph graph("write_after_write");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &A);
        cg::permute("ij <- ji", 0.0, &A, 1.0, B);
    }

    graph.execute();

    // A should be B^T (second write wins)
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE_THAT(A(ii, jj), Catch::Matchers::WithinRel(A_ref(ii, jj), 1e-12));
        }
    }
}

TEST_CASE("Dependency - symmetrize pattern in graph", "[ComputeGraph][Dependency]") {
    // This is the exact pattern that failed before the write-after-read fix:
    // 1. permute(A -> At)  [reads A]
    // 2. scale(0.5, A)     [reads+writes A]
    // 3. axpy(0.5, At, A)  [reads At, writes A]
    // Without write-after-read tracking, step 2 could reorder before step 1.
    auto A      = create_random_tensor<double>("A", 4, 4);
    auto A_orig = Tensor<double, 2>(A);

    // Reference: symmetrize A = 0.5*(A + A^T)
    auto A_ref = Tensor<double, 2>(A);
    auto At    = Tensor<double, 2>("At", 4, 4);
    reference_permute("ji <- ij", 0.0, &At, 1.0, A_ref);
    linear_algebra::scale(0.5, &A_ref);
    linear_algebra::axpy(0.5, At, &A_ref);

    // Reset
    A = Tensor<double, 2>(A_orig);

    cg::Graph graph("symmetrize");
    auto     &At_g = graph.create_tensor<double, 2>("At_g", 4, 4);
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &At_g, 1.0, A);
        cg::scale(0.5, &A);
        cg::axpy(0.5, At_g, &A);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(A(ii, jj), Catch::Matchers::WithinRel(A_ref(ii, jj), 1e-12));
        }
    }

    // Symmetrized matrix should be symmetric
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE_THAT(A(ii, jj), Catch::Matchers::WithinRel(A(jj, ii), 1e-12));
        }
    }
}

TEST_CASE("Dependency - replay produces same result", "[ComputeGraph][Dependency]") {
    auto A = create_random_tensor<double>("A", 5, 5);
    auto B = create_random_tensor<double>("B", 5, 5);
    auto C = create_zero_tensor<double>("C", 5, 5);

    cg::Graph graph("replay");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    graph.execute();
    auto C_first = Tensor<double, 2>(C);

    C.zero();
    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE_THAT(C(ii, jj), Catch::Matchers::WithinRel(C_first(ii, jj), 1e-12));
        }
    }
}

// A Loop node's named reads and writes used to cover only its own condition,
// not its body, so a dataset written inside a loop and read after it carried
// no edge from the loop to the reader, and the scheduling passes moved the
// read ahead of the loop.
TEST_CASE("Dependency - a disk read after a loop waits for the loop's disk write", "[ComputeGraph][Dependency][IO][Loop]") {
    RuntimeTensor<double> A{"A", {3UL, 3UL}};
    RuntimeTensor<double> Y{"Y", {3UL}};
    RuntimeTensor<double> X{"X", {3UL}};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            A(i, j) = static_cast<double>(1 + (3 * i) + j); // 1..9
        }
    }
    std::vector<double> dataset(3, 0.0); // stands in for mock.h5#/row
    size_t              iter = 0;

    cg::Graph graph("dep_loop_disk");
    {
        auto                  &body = graph.add_loop("rows", 3, [&iter](size_t) {
            ++iter;
            return iter < 3;
        });
        cg::CaptureGuard const capture(body);
        cg::write_param("r", std::function<std::int64_t()>([&iter] { return static_cast<std::int64_t>(iter); }));
        auto &slice = cg::view_runtime(A, {cg::ViewAxis::drop("r"), cg::ViewAxis::full()});
        cg::axpby(1.0, slice, 0.0, &Y);
        cg::write("save", "mock.h5", "/row", &Y, [&dataset, &Y]() {
            for (size_t j = 0; j < 3; ++j) {
                dataset[j] = Y(j);
            }
        });
    }
    {
        cg::CaptureGuard const capture(graph);
        cg::read("load", "mock.h5", "/row", &X, [&dataset, &X]() {
            for (size_t j = 0; j < 3; ++j) {
                X(j) = dataset[j];
            }
        });
    }

    // The loop writes the dataset, so its named writes must say so.
    auto const loop_node = std::ranges::find_if(graph.nodes(), [](cg::Node const &node) { return node.kind == cg::OpKind::Loop; });
    REQUIRE(loop_node != graph.nodes().end());
    auto const writes = cg::named_writes(*loop_node);
    CHECK(std::ranges::find(writes, std::string{"disk:mock.h5#/row"}) != writes.end());

    auto pm = cg::PassManager::create_default();
    graph.apply(pm);
    INFO(pm.explain());
    std::string order;
    for (auto const &node : graph.nodes()) {
        order += node.label + "; ";
    }
    INFO("parent order after the pipeline: " << order);

    iter = 0;
    REQUIRE_NOTHROW(graph.execute());
    // The last iteration writes row 2.
    CHECK(X(0) == 7.0);
    CHECK(X(1) == 8.0);
    CHECK(X(2) == 9.0);
}

// Two custom nodes that touch no tensor, each acting only on the world outside
// the graph (here a log), shared no tensor and no named key, so the hazard scan
// drew no edge between them and the parallel executors were free to run them
// in either order or at once.
TEST_CASE("Dependency - effect-only custom nodes keep program order", "[ComputeGraph][Dependency][Custom]") {
    std::vector<int> log;

    cg::Graph graph("dep_custom_effects");
    {
        cg::CaptureGuard const capture(graph);
        cg::custom("first", [&log]() { log.push_back(1); });
        cg::custom("second", [&log]() { log.push_back(2); });
    }

    graph.topological_sort();
    auto const &deps = graph.dependencies();
    REQUIRE(deps.predecessors.size() == 2);
    CHECK(std::ranges::find(deps.predecessors[1], size_t{0}) != deps.predecessors[1].end());
    CHECK(deps.levels.size() == 2);

    graph.execute();
    CHECK(log == std::vector<int>{1, 2});
}

namespace {

/// Whether the hazard scan gives node @p to an edge from node @p from.
bool has_edge(cg::Graph &graph, size_t from, size_t to) {
    graph.topological_sort();
    auto const &preds = graph.dependencies().predecessors[to];
    return std::ranges::find(preds, from) != preds.end();
}

} // namespace

// A closure the graph cannot see into may read the file a DiskWrite stores or
// write the one a DiskRead loads, and shares no tensor with either, so an
// opaque custom node is ordered against every named resource.
TEST_CASE("Dependency - an opaque custom node is ordered against disk accesses", "[ComputeGraph][Dependency][Custom][IO]") {
    RuntimeTensor<double> Y{"Y", {3UL}};
    RuntimeTensor<double> X{"X", {3UL}};
    Y.zero();

    cg::Graph graph("dep_custom_disk");
    {
        cg::CaptureGuard const capture(graph);
        cg::write("save", "mock.h5", "/row", &Y, []() {});
        cg::custom("inspect_file", []() {});
        cg::read("load", "mock.h5", "/other", &X, []() {});
    }

    CHECK(has_edge(graph, 0, 1)); // the closure may read what was stored
    CHECK(has_edge(graph, 1, 2)); // and may write what is loaded
}

// Declared pure, the same closure touches only what it lists, and the disk
// accesses around it stay free of it.
TEST_CASE("Dependency - a custom node declared pure gains no named edges", "[ComputeGraph][Dependency][Custom][IO]") {
    RuntimeTensor<double> Y{"Y", {3UL}};
    RuntimeTensor<double> X{"X", {3UL}};
    Y.zero();

    cg::Graph graph("dep_custom_pure_disk");
    {
        cg::CaptureGuard const capture(graph);
        cg::write("save", "mock.h5", "/row", &Y, []() {});
        cg::custom(cg::pure, "compute", []() {});
        cg::read("load", "mock.h5", "/other", &X, []() {});
    }

    CHECK_FALSE(has_edge(graph, 0, 1));
    CHECK_FALSE(has_edge(graph, 1, 2));
}

// The wildcard an opaque node carries must not leak into unrelated keys: two
// parameter writes of different names share nothing and stay independent.
TEST_CASE("Dependency - writes of different parameters stay independent", "[ComputeGraph][Dependency][WriteParam]") {
    cg::Graph graph("dep_two_params");
    {
        cg::CaptureGuard const capture(graph);
        cg::write_param("a", std::function<std::int64_t()>([] { return std::int64_t{1}; }));
        cg::write_param("b", std::function<std::int64_t()>([] { return std::int64_t{2}; }));
    }

    CHECK_FALSE(has_edge(graph, 0, 1));
    CHECK(graph.dependencies().levels.size() == 1);
}

// The parameter half of a loop's named writes: a parameter written in the body
// is read by a slice after the loop, so the slice must follow the loop. This
// was unreachable while a captured body held a ParamTable of its own.
TEST_CASE("Dependency - a slice after a loop waits for the loop's parameter write", "[ComputeGraph][Dependency][WriteParam][Loop]") {
    RuntimeTensor<double> A{"A", {3UL, 3UL}};
    A.zero();
    size_t iter = 0;

    cg::Graph graph("dep_loop_param");
    auto     &body = graph.add_loop("rows", 3, [&iter](size_t) {
        ++iter;
        return iter < 3;
    });
    {
        cg::CaptureGuard const capture(body);
        cg::write_param("r", std::function<std::int64_t()>([&iter] { return static_cast<std::int64_t>(iter); }));
    }
    {
        cg::CaptureGuard const capture(graph);
        (void)cg::view_runtime(A, {cg::ViewAxis::drop("r"), cg::ViewAxis::full()});
    }

    REQUIRE(graph.nodes().size() == 2);
    CHECK(has_edge(graph, 0, 1));
}
