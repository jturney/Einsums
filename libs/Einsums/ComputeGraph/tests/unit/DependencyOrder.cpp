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
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
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
    tensor_algebra::permute(0.0, Indices{i, j}, &B_ref, 1.0, Indices{i, j}, A_ref);

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
    tensor_algebra::permute(0.0, Indices{j, i}, &B_ref, 1.0, Indices{i, j}, A);
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
    tensor_algebra::element_transform(&A_ref, [](double v) { return v * v; });
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
    tensor_algebra::permute(0.0, Indices{i, j}, &B_ref, 2.0, Indices{i, j}, A);
    tensor_algebra::permute(0.0, Indices{j, i}, &C_ref, 1.0, Indices{i, j}, A);
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
    tensor_algebra::permute(0.0, Indices{i, j}, &A_ref, 1.0, Indices{j, i}, B);

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
    tensor_algebra::permute(0.0, Indices{j, i}, &At, 1.0, Indices{i, j}, A_ref);
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
