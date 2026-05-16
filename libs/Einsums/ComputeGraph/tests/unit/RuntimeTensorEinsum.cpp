//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase B smoke test: cg::einsum on RuntimeTensor operands.
//
// Validates that the runtime-rank dispatch path through string_einsum's
// generic fallback produces results matching the typed-tensor path on
// identical data. This is the first end-to-end exercise of cg::einsum
// against a tensor variant that lacks a static ::Rank.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("cg::einsum — RuntimeTensor GEMM-shaped contraction", "[ComputeGraph][RuntimeTensor]") {
    auto A_typed = create_random_tensor<double>("A", 4, 3);
    auto B_typed = create_random_tensor<double>("B", 3, 5);
    auto C_typed = create_zero_tensor<double>("C", 4, 5);

    // Reference path: typed Tensor<double, 2> through cg::einsum
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C_typed, A_typed, B_typed);

    // Runtime-rank path: same data, same spec, RuntimeTensor operands
    RuntimeTensor<double> const A_rt(A_typed);
    RuntimeTensor<double> const B_rt(B_typed);
    RuntimeTensor<double>       C_rt("C", {4UL, 5UL});
    C_rt.zero();

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C_rt, A_rt, B_rt);

    REQUIRE(C_rt.rank() == 2);
    REQUIRE(C_rt.dim(0) == 4);
    REQUIRE(C_rt.dim(1) == 5);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C_rt(ii, jj) - C_typed(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("cg::einsum — RuntimeTensor rank-3 batched contraction", "[ComputeGraph][RuntimeTensor]") {
    // "ijk <- ik ; ij" — rank-3 from rank-2 operands.
    auto A_typed = create_random_tensor<double>("A", 2, 3);
    auto B_typed = create_random_tensor<double>("B", 2, 4);
    auto C_typed = create_zero_tensor<double>("C", 2, 4, 3);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ijk <- ik ; ij", &C_typed, A_typed, B_typed);

    RuntimeTensor<double> const A_rt(A_typed);
    RuntimeTensor<double> const B_rt(B_typed);
    RuntimeTensor<double>       C_rt("C", {2UL, 4UL, 3UL});
    C_rt.zero();

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ijk <- ik ; ij", &C_rt, A_rt, B_rt);

    REQUIRE(C_rt.rank() == 3);

    for (size_t ii = 0; ii < 2; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t kk = 0; kk < 3; kk++) {
                REQUIRE(std::abs(C_rt(ii, jj, kk) - C_typed(ii, jj, kk)) < 1e-12);
            }
        }
    }
}

TEST_CASE("cg::einsum — RuntimeTensor through graph capture + execute", "[ComputeGraph][RuntimeTensor]") {
    auto A_typed = create_random_tensor<double>("A", 5, 3);
    auto B_typed = create_random_tensor<double>("B", 3, 4);
    auto C_typed = create_zero_tensor<double>("C", 5, 4);

    // Reference: capture+execute on typed tensors.
    cg::Graph graph_typed("rt_einsum_typed");
    {
        cg::CaptureGuard const guard(graph_typed);
        cg::einsum("ij <- ik ; kj", &C_typed, A_typed, B_typed);
    }
    graph_typed.execute();

    // RuntimeTensor: same data, captured into its own graph, executed.
    RuntimeTensor<double> const A_rt(A_typed);
    RuntimeTensor<double> const B_rt(B_typed);
    RuntimeTensor<double>       C_rt("C", {5UL, 4UL});
    C_rt.zero();

    cg::Graph graph_rt("rt_einsum_runtime");
    {
        cg::CaptureGuard const guard(graph_rt);
        cg::einsum("ij <- ik ; kj", &C_rt, A_rt, B_rt);
    }
    REQUIRE(graph_rt.num_nodes() == 1);
    graph_rt.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C_rt(ii, jj) - C_typed(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("cg::einsum — rank-4 outer product (ijab <- ia ; jb)", "[ComputeGraph][RuntimeTensor]") {
    // C[i,j,a,b] = A[i,a] * B[j,b]. NumPy spelling would be "ia,jb->ijab";
    // ComputeGraph requires ';' as the operand separator.
    constexpr size_t I = 2, J = 3, A = 2, B = 3;

    auto A_typed = create_random_tensor<double>("A", I, A);
    auto B_typed = create_random_tensor<double>("B", J, B);
    auto C_typed = create_zero_tensor<double>("C", I, J, A, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ijab <- ia ; jb", &C_typed, A_typed, B_typed);

    // Validate against the closed-form definition: every entry is the
    // product of one A entry and one B entry, no contractions.
    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            for (size_t a = 0; a < A; a++) {
                for (size_t b = 0; b < B; b++) {
                    REQUIRE(std::abs(C_typed(i, j, a, b) - A_typed(i, a) * B_typed(j, b)) < 1e-12);
                }
            }
        }
    }

    // Same operation on RuntimeTensor operands — exercises the runtime-rank
    // generic_string_einsum path with no link indices and a rank-4 output.
    RuntimeTensor<double> const A_rt(A_typed);
    RuntimeTensor<double> const B_rt(B_typed);
    RuntimeTensor<double>       C_rt("C", {I, J, A, B});
    C_rt.zero();

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ijab <- ia ; jb", &C_rt, A_rt, B_rt);

    REQUIRE(C_rt.rank() == 4);
    REQUIRE(C_rt.dim(0) == I);
    REQUIRE(C_rt.dim(1) == J);
    REQUIRE(C_rt.dim(2) == A);
    REQUIRE(C_rt.dim(3) == B);

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            for (size_t a = 0; a < A; a++) {
                for (size_t b = 0; b < B; b++) {
                    REQUIRE(std::abs(C_rt(i, j, a, b) - C_typed(i, j, a, b)) < 1e-12);
                }
            }
        }
    }
}

TEST_CASE("cg::einsum — RuntimeTensor batched-GEMM via PackedGemm (bij;bjk->bik)", "[ComputeGraph][RuntimeTensor][PackedGemm]") {
    // C[b,i,k] = sum_j A[b,i,j] * B[b,j,k]. Has a Hadamard index (b) and
    // a link index (j) — exactly the shape PackedGemm specializes for.
    // Previously this would have fallen through to the generic nested-loop
    // kernel for RuntimeTensor; now it should reach try_packed_gemm via
    // the runtime ContractionSpec entry point.
    constexpr size_t Bd = 3, I = 2, J = 4, K = 3;

    auto A_typed = create_random_tensor<double>("A", Bd, I, J);
    auto B_typed = create_random_tensor<double>("B", Bd, J, K);
    auto C_typed = create_zero_tensor<double>("C", Bd, I, K);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("bik <- bij ; bjk", &C_typed, A_typed, B_typed);

    auto C_ref = create_zero_tensor<double>("Cr", Bd, I, K);
    for (size_t b = 0; b < Bd; ++b)
        for (size_t i = 0; i < I; ++i)
            for (size_t k = 0; k < K; ++k) {
                double sum = 0.0;
                for (size_t j = 0; j < J; ++j)
                    sum += A_typed(b, i, j) * B_typed(b, j, k);
                C_ref(b, i, k) = sum;
            }

    for (size_t b = 0; b < Bd; ++b)
        for (size_t i = 0; i < I; ++i)
            for (size_t k = 0; k < K; ++k)
                REQUIRE(std::abs(C_typed(b, i, k) - C_ref(b, i, k)) < 1e-12);

    RuntimeTensor<double> const A_rt(A_typed);
    RuntimeTensor<double> const B_rt(B_typed);
    RuntimeTensor<double>       C_rt("C", {Bd, I, K});
    C_rt.zero();

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("bik <- bij ; bjk", &C_rt, A_rt, B_rt);

    REQUIRE(C_rt.rank() == 3);
    for (size_t b = 0; b < Bd; ++b)
        for (size_t i = 0; i < I; ++i)
            for (size_t k = 0; k < K; ++k)
                REQUIRE(std::abs(C_rt(b, i, k) - C_ref(b, i, k)) < 1e-12);
}

TEST_CASE("cg::einsum — operand rank vs spec mismatch throws", "[ComputeGraph][RuntimeTensor][SpecCheck]") {
    // Typed tensor: AType::Rank = 3 but spec says A has 2 indices.
    // For typed tensors the consteval-populated spec.counts make this a
    // constant comparison; the throw is a real exception at runtime, but
    // any optimizer that constant-folds the rank comparison will dead-code
    // the throw branch when ranks match.
    auto A3 = create_random_tensor<double>("A3", 2, 3, 4);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 2, 5);
    REQUIRE_THROWS_AS(
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C, A3, B), std::invalid_argument);

    // RuntimeTensor: same check fires at runtime via tensor.rank().
    RuntimeTensor<double> A_rt("A", {2UL, 3UL, 4UL}); // rank 3 not 2
    RuntimeTensor<double> B_rt("B", {3UL, 5UL});
    RuntimeTensor<double> C_rt("C", {2UL, 5UL});
    REQUIRE_THROWS_AS(
        // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
        cg::einsum("ij <- ik ; kj", &C_rt, A_rt, B_rt), std::invalid_argument);
}

TEST_CASE("cg::einsum — RuntimeTensor with float dtype", "[ComputeGraph][RuntimeTensor]") {
    auto A_typed = create_random_tensor<float>("A", 3, 3);
    auto B_typed = create_random_tensor<float>("B", 3, 3);
    auto C_typed = create_zero_tensor<float>("C", 3, 3);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C_typed, A_typed, B_typed);

    RuntimeTensor<float> const A_rt(A_typed);
    RuntimeTensor<float> const B_rt(B_typed);
    RuntimeTensor<float>       C_rt("C", {3UL, 3UL});
    C_rt.zero();

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C_rt, A_rt, B_rt);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C_rt(ii, jj) - C_typed(ii, jj)) < 1e-5f);
        }
    }
}
