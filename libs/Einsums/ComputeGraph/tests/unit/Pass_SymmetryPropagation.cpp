//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Pass_SymmetryPropagation.cpp
/// @brief Unit tests for the SymmetryPropagation pass.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("SymmetryPropagation - A^T A (ki,kj->ij) produces symmetric", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 5, 5);

    cg::Graph graph("ata");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 5, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ki;kj->ij", &C, A, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();

    REQUIRE(pass.num_inferred() == 1);
    REQUIRE(C.has_symmetry());
    REQUIRE(C.symmetry()->size() == 1);
    REQUIRE(C.symmetry()->ops[0].sign == +1);
    REQUIRE(C.symmetry()->ops[0].permutation[0] == 1);
    REQUIRE(C.symmetry()->ops[0].permutation[1] == 0);
}

TEST_CASE("SymmetryPropagation - A A^T (ik,jk->ij) also symmetric", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 7);

    cg::Graph graph("aat");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;jk->ij", &C, A, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();

    REQUIRE(pass.num_inferred() == 1);
    REQUIRE(C.has_symmetry());
}

TEST_CASE("SymmetryPropagation - A B (different tensors) does not infer", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    cg::Graph graph("ab");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();

    REQUIRE(pass.num_inferred() == 0);
    REQUIRE_FALSE(C.has_symmetry());
}

TEST_CASE("SymmetryPropagation - does not mutate user-owned tensors", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    // C is user-owned (not graph-created), so propagation must leave it alone
    // even if the operation writing to it would be provably symmetric.
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("user_owned");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ki;kj->ij", &C, A, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();

    REQUIRE(pass.num_inferred() == 0);
    REQUIRE_FALSE(C.has_symmetry());
}

TEST_CASE("SymmetryPropagation - inferred tag survives re-run", "[ComputeGraph][Passes][Symmetry]") {
    // Running the pass twice should not double-count or error. The second
    // pass finds nothing new to infer.
    auto A = create_random_tensor<double>("A", 5, 5);

    cg::Graph graph("rerun");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 5, 5);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ki;kj->ij", &C, A, A);
    }

    auto [_m1, pass1] = graph.apply<cg::passes::SymmetryPropagation>();
    REQUIRE(pass1.num_inferred() == 1);

    auto [_m2, pass2] = graph.apply<cg::passes::SymmetryPropagation>();
    REQUIRE(pass2.num_inferred() == 0); // already tagged; no new work
    REQUIRE(C.has_symmetry());
}

// Note on Hermitian coverage: the propagate_self_contraction rule
// distinguishes symmetric vs Hermitian output based on EinsumDescriptor's
// conj_a/conj_b flags, but the current cg::einsum capture path does not
// expose a way to set those flags. Once conjugation wiring lands in the
// graph API, a Hermitian integration test can be added without further
// pass changes.

TEST_CASE("SymmetryPropagation - permute of symmetric stays symmetric", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    A.set_symmetry(SymmetryDescriptor::symmetric_pair(0, 1));
    symmetrize(A);

    cg::Graph graph("permute_sym");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &T, 1.0, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();
    REQUIRE(pass.num_inferred() == 1);
    REQUIRE(T.has_symmetry());
    REQUIRE(T.symmetry()->ops[0].sign == +1);
}

TEST_CASE("SymmetryPropagation - permute of general does not infer", "[ComputeGraph][Passes][Symmetry]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    // No descriptor attached → permute has nothing to propagate.

    cg::Graph graph("permute_general");
    auto     &T = graph.create_zero_tensor<double, 2>("T", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", 0.0, &T, 1.0, A);
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();
    REQUIRE(pass.num_inferred() == 0);
    REQUIRE_FALSE(T.has_symmetry());
}

TEST_CASE("SymmetryPropagation - inferred C executes correctly via gemm dispatch", "[ComputeGraph][Passes][Symmetry]") {
    // End-to-end: propagation tags C symmetric; later execute() benefits
    // from the rank-2 BLAS dispatch without the user doing anything.
    auto A = create_random_tensor<double>("A", 6, 6);
    auto B = create_random_tensor<double>("B", 6, 6);
    auto D = create_zero_tensor<double>("D", 6, 6);

    cg::Graph graph("e2e");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 6, 6);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ki;kj->ij", &C, A, A); // C is AᵀA → symmetric
        cg::einsum("ij;jk->ik", &D, C, B); // D = C·B — hits symm dispatch once C is tagged
    }

    auto [_m, pass] = graph.apply<cg::passes::SymmetryPropagation>();
    REQUIRE(pass.num_inferred() == 1);
    REQUIRE(C.has_symmetry());

    graph.execute();

    auto C_ref = create_zero_tensor<double>("Cref", 6, 6);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{k, i}, A, Indices{k, j}, A);
    auto D_ref = create_zero_tensor<double>("Dref", 6, 6);
    tensor_algebra::einsum(Indices{i, k}, &D_ref, Indices{i, j}, C_ref, Indices{j, k}, B);

    for (int ii = 0; ii < 6; ++ii)
        for (int jj = 0; jj < 6; ++jj)
            CHECK(D(ii, jj) == Catch::Approx(D_ref(ii, jj)).margin(1e-10));
}

// ── Loop-aware behavior (single-writer + no-child-write guards) ──────────

TEST_CASE("SymmetryPropagation - infers on a self-contraction inside a loop body", "[ComputeGraph][Symmetry][Loop]") {
    // S = A·Aᵀ inside a loop body is symmetric every iteration. With the
    // pass recursing, the body intermediate S should get tagged.
    auto A = create_random_tensor<double>("A", 4, 3);

    cg::Graph g("sym_loop");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    auto     &S    = body.create_zero_tensor<double, 2>("S", 4, 4);
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;jk->ij", &S, A, A); // S = A·Aᵀ → symmetric
    }

    cg::PassManager pm;
    pm.add<cg::passes::SymmetryPropagation>();
    pm.run(g);

    REQUIRE(S.has_symmetry());
    REQUIRE(S.symmetry()->ops[0].sign == +1);
}

TEST_CASE("SymmetryPropagation - does NOT tag a multi-writer body tensor", "[ComputeGraph][Symmetry][Loop]") {
    // S is made symmetric by a self-contraction, then overwritten in-place
    // by a non-symmetric add. The single-writer guard must refuse to tag S
    // (its final value isn't symmetric).
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    cg::Graph g("sym_multiwriter");
    auto     &body = g.add_loop("iter", 1, [](size_t) { return false; });
    auto     &S    = body.create_zero_tensor<double, 2>("S", 4, 4);
    {
        cg::CaptureGuard const guard(body);
        cg::einsum("ik;jk->ij", &S, A, A); // S = A·Aᵀ (symmetric)
        cg::axpy(1.0, B, &S);              // S += B (destroys symmetry) — 2nd writer
    }

    cg::PassManager pm;
    pm.add<cg::passes::SymmetryPropagation>();
    pm.run(g);

    CHECK_FALSE(S.has_symmetry()); // two writers → not tagged
}

TEST_CASE("SymmetryPropagation - does NOT tag a tensor written by a nested loop", "[ComputeGraph][Symmetry][Loop]") {
    // S is made symmetric in the outer body, but a nested loop also writes
    // it (non-symmetrically). A Loop node doesn't list its body's writes, so
    // without the subtree guard the pass would wrongly tag S. It must not.
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);

    cg::Graph g("sym_nested");
    auto     &outer = g.add_loop("outer", 1, [](size_t) { return false; });
    auto     &S     = outer.create_zero_tensor<double, 2>("S", 4, 4);
    {
        cg::CaptureGuard const guard(outer);
        cg::einsum("ik;jk->ij", &S, A, A); // S = A·Aᵀ (symmetric) in outer body
    }
    auto &inner = outer.add_loop("inner", 1, [](size_t) { return false; });
    {
        cg::CaptureGuard const guard(inner);
        cg::axpy(1.0, B, &S); // nested loop writes S non-symmetrically
    }

    cg::PassManager pm;
    pm.add<cg::passes::SymmetryPropagation>();
    pm.run(g);

    CHECK_FALSE(S.has_symmetry()); // referenced by a child sub-graph → not tagged
}
