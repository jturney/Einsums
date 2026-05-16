//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
namespace cg = einsums::compute_graph;

TEST_CASE("String einsum - arrow notation, direct execute", "[ComputeGraph][StringEinsum]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    // Reference via template-based einsum
    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, k}, A, Indices{k, j}, B);

    // String-based direct execution (no capture)
    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", &C, A, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - numpy notation, direct execute", "[ComputeGraph][StringEinsum]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, k}, A, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ik;kj -> ij", &C, A, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - with prefactors", "[ComputeGraph][StringEinsum]") {
    auto A          = create_random_tensor<double>("A", 3, 4);
    auto B          = create_random_tensor<double>("B", 4, 3);
    auto C          = create_random_tensor<double>("C", 3, 3);
    auto C_expected = Tensor<double, 2>(C);

    tensor_algebra::einsum(2.0, Indices{i, j}, &C_expected, 3.0, Indices{i, k}, A, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ik ; kj", 2.0, &C, 3.0, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - graph capture and execute", "[ComputeGraph][StringEinsum]") {
    auto A          = create_random_tensor<double>("A", 5, 3);
    auto B          = create_random_tensor<double>("B", 3, 4);
    auto C          = create_zero_tensor<double>("C", 5, 4);
    auto C_expected = create_zero_tensor<double>("Ce", 5, 4);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("string_einsum_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t ii = 0; ii < 5; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - chain in graph", "[ComputeGraph][StringEinsum]") {
    auto A      = create_random_tensor<double>("A", 4, 3);
    auto B      = create_random_tensor<double>("B", 3, 5);
    auto D      = create_random_tensor<double>("D", 5, 2);
    auto T1     = create_zero_tensor<double>("T1", 4, 5);
    auto E      = create_zero_tensor<double>("E", 4, 2);
    auto T1_ref = create_zero_tensor<double>("T1r", 4, 5);
    auto E_ref  = create_zero_tensor<double>("Er", 4, 2);

    tensor_algebra::einsum(Indices{i, j}, &T1_ref, Indices{i, k}, A, Indices{k, j}, B);
    tensor_algebra::einsum(Indices{i, l}, &E_ref, Indices{i, j}, T1_ref, Indices{j, l}, D);

    cg::Graph graph("string_chain");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ik ; kj", &T1, A, B);
        cg::einsum("il <- ij ; jl", &E, T1, D);
    }

    REQUIRE(graph.num_nodes() == 2);
    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t ll = 0; ll < 2; ll++) {
            REQUIRE(std::abs(E(ii, ll) - E_ref(ii, ll)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - works with optimization passes", "[ComputeGraph][StringEinsum]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_random_tensor<double>("C", 4, 5);

    auto C_ref = Tensor<double, 2>(C);
    linear_algebra::scale(2.0, &C_ref);
    tensor_algebra::einsum(0.0, Indices{i, j}, &C_ref, 1.0, Indices{i, k}, A, Indices{k, j}, B);

    cg::Graph graph("string_with_passes");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(2.0, &C);
        cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);
    }

    REQUIRE(graph.num_nodes() == 2);

    // ScaleAbsorption should work on string-based einsum nodes
    auto [modified, _pass] = graph.apply<cg::passes::ScaleAbsorption>();
    REQUIRE(modified);
    REQUIRE(graph.num_nodes() == 1);

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - pipeline with loop", "[ComputeGraph][StringEinsum]") {
    auto A   = create_random_tensor<double>("A", 3, 3);
    auto B   = create_random_tensor<double>("B", 3, 3);
    auto acc = create_zero_tensor<double>("acc", 3, 3);

    auto C = create_zero_tensor<double>("C", 3, 3);

    cg::Pipeline pipeline("string_pipeline");

    {
        auto                  &setup = pipeline.add_stage("setup");
        cg::CaptureGuard const guard(setup);
        cg::einsum("ij <- ik ; kj", &C, A, B);
    }

    size_t count = 0;
    {
        auto                  &loop_body = pipeline.add_loop("accumulate", 3, [&](size_t iter) {
            count = iter + 1;
            return iter < 2;
        });
        cg::CaptureGuard const guard(loop_body);
        cg::axpy(1.0, C, &acc);
    }

    pipeline.execute();

    REQUIRE(count == 3);

    // acc = 3 * C = 3 * A * B
    auto C_ref = create_zero_tensor<double>("Cref", 3, 3);
    tensor_algebra::einsum(Indices{i, j}, &C_ref, Indices{i, k}, A, Indices{k, j}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            REQUIRE(std::abs(acc(ii, jj) - 3.0 * C_ref(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - transposed A", "[ComputeGraph][StringEinsum]") {
    auto A          = create_random_tensor<double>("A", 3, 4);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    // C[i,j] = A[k,i] * B[k,j]  (A is transposed)
    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{k, i}, A, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ki ; kj", &C, A, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Non-GEMM dispatch patterns
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("String einsum - GEMV (matrix-vector)", "[ComputeGraph][StringEinsum][Phase2]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto x          = create_random_tensor<double>("x", 3);
    auto y          = create_zero_tensor<double>("y", 4);
    auto y_expected = create_zero_tensor<double>("ye", 4);

    tensor_algebra::einsum(Indices{i}, &y_expected, Indices{i, k}, A, Indices{k}, x);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("i <- ik ; k", &y, A, x);

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(y(ii) - y_expected(ii)) < 1e-12);
    }
}

TEST_CASE("String einsum - GEMV transposed", "[ComputeGraph][StringEinsum][Phase2]") {
    auto A          = create_random_tensor<double>("A", 3, 4);
    auto x          = create_random_tensor<double>("x", 3);
    auto y          = create_zero_tensor<double>("y", 4);
    auto y_expected = create_zero_tensor<double>("ye", 4);

    tensor_algebra::einsum(Indices{i}, &y_expected, Indices{k, i}, A, Indices{k}, x);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("i <- ki ; k", &y, A, x);

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(y(ii) - y_expected(ii)) < 1e-12);
    }
}

TEST_CASE("String einsum - GEMV vector * matrix", "[ComputeGraph][StringEinsum][Phase2]") {
    auto x          = create_random_tensor<double>("x", 3);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto y          = create_zero_tensor<double>("y", 5);
    auto y_expected = create_zero_tensor<double>("ye", 5);

    tensor_algebra::einsum(Indices{j}, &y_expected, Indices{k}, x, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("j <- k ; kj", &y, x, B);

    for (size_t jj = 0; jj < 5; jj++) {
        REQUIRE(std::abs(y(jj) - y_expected(jj)) < 1e-12);
    }
}

TEST_CASE("String einsum - GER (outer product)", "[ComputeGraph][StringEinsum][Phase2]") {
    auto x          = create_random_tensor<double>("x", 4);
    auto y          = create_random_tensor<double>("y", 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i}, x, Indices{j}, y);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture, einsums-no-link-index)
    cg::einsum("ij <- i ; j", &C, x, y);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - DOT product", "[ComputeGraph][StringEinsum][Phase2]") {
    auto x = create_random_tensor<double>("x", 10);
    auto y = create_random_tensor<double>("y", 10);

    double const expected = linear_algebra::dot(x, y);

    auto result = create_zero_tensor<double>("result", 1);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum(" <- i ; i", &result, x, y);

    REQUIRE(std::abs(result(0) - expected) < 1e-10);
}

TEST_CASE("String einsum - direct product (element-wise)", "[ComputeGraph][StringEinsum][Phase2]") {
    auto A          = create_random_tensor<double>("A", 3, 4);
    auto B          = create_random_tensor<double>("B", 3, 4);
    auto C          = create_zero_tensor<double>("C", 3, 4);
    auto C_expected = create_zero_tensor<double>("Ce", 3, 4);

    linear_algebra::direct_product(1.0, A, B, 0.0, &C_expected);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ij ; ij", &C, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - GEMV in graph capture", "[ComputeGraph][StringEinsum][Phase2]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto x          = create_random_tensor<double>("x", 3);
    auto y          = create_zero_tensor<double>("y", 4);
    auto y_expected = create_zero_tensor<double>("ye", 4);

    tensor_algebra::einsum(Indices{i}, &y_expected, Indices{i, k}, A, Indices{k}, x);

    cg::Graph graph("gemv_graph");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("i <- ik ; k", &y, A, x);
    }

    graph.execute();

    for (size_t ii = 0; ii < 4; ii++) {
        REQUIRE(std::abs(y(ii) - y_expected(ii)) < 1e-12);
    }
}

TEST_CASE("String einsum - GER in graph capture", "[ComputeGraph][StringEinsum][Phase2]") {
    auto x          = create_random_tensor<double>("x", 3);
    auto y          = create_random_tensor<double>("y", 4);
    auto C          = create_zero_tensor<double>("C", 3, 4);
    auto C_expected = create_zero_tensor<double>("Ce", 3, 4);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i}, x, Indices{j}, y);

    cg::Graph graph("ger_graph");
    {
        cg::CaptureGuard const guard(graph);
        // NOLINTNEXTLINE(einsums-no-link-index)
        cg::einsum("ij <- i ; j", &C, x, y);
    }

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String einsum - multi-char indices with GEMM", "[ComputeGraph][StringEinsum][Phase2]") {
    auto A          = create_random_tensor<double>("A", 4, 3);
    auto B          = create_random_tensor<double>("B", 3, 5);
    auto C          = create_zero_tensor<double>("C", 4, 5);
    auto C_expected = create_zero_tensor<double>("Ce", 4, 5);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, k}, A, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("mu,nu <- mu,rho ; rho,nu", &C, A, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Higher-rank contractions (rank 3+)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("String einsum - rank-3 contraction to rank-2", "[ComputeGraph][StringEinsum][HighRank]") {
    // C[i,l] = A[i,j,k] * B[j,k,l]  (contract over j,k)
    auto A          = create_random_tensor<double>("A", 3, 4, 5);
    auto B          = create_random_tensor<double>("B", 4, 5, 2);
    auto C          = create_zero_tensor<double>("C", 3, 2);
    auto C_expected = create_zero_tensor<double>("Ce", 3, 2);

    tensor_algebra::einsum(Indices{i, l}, &C_expected, Indices{i, j, k}, A, Indices{j, k, l}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("il <- ijk ; jkl", &C, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t ll = 0; ll < 2; ll++) {
            REQUIRE(std::abs(C(ii, ll) - C_expected(ii, ll)) < 1e-10);
        }
    }
}

TEST_CASE("String einsum - rank-3 contraction to rank-1", "[ComputeGraph][StringEinsum][HighRank]") {
    // C[i] = A[i,j,k] * B[j,i,k]  (contract over j,k)
    auto A          = create_random_tensor<double>("A", 3, 4, 5);
    auto B          = create_random_tensor<double>("B", 4, 3, 5);
    auto C          = create_zero_tensor<double>("C", 3);
    auto C_expected = create_zero_tensor<double>("Ce", 3);

    tensor_algebra::einsum(Indices{i}, &C_expected, Indices{i, j, k}, A, Indices{j, i, k}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("i <- ijk ; jik", &C, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        REQUIRE(std::abs(C(ii) - C_expected(ii)) < 1e-10);
    }
}

TEST_CASE("String einsum - rank-3 × rank-2 contraction", "[ComputeGraph][StringEinsum][HighRank]") {
    // C[i,j] = A[i,j,k] * B[k,j]  (contract over k)
    // Note: j appears in A, B, and C — it's a target index, not a link
    auto A          = create_random_tensor<double>("A", 3, 4, 5);
    auto B          = create_random_tensor<double>("B", 5, 4);
    auto C          = create_zero_tensor<double>("C", 3, 4);
    auto C_expected = create_zero_tensor<double>("Ce", 3, 4);

    tensor_algebra::einsum(Indices{i, j}, &C_expected, Indices{i, j, k}, A, Indices{k, j}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ij <- ijk ; kj", &C, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-10);
        }
    }
}

TEST_CASE("String einsum - rank-3 × rank-3 to rank-2 in graph", "[ComputeGraph][StringEinsum][HighRank]") {
    auto A          = create_random_tensor<double>("A", 3, 4, 5);
    auto B          = create_random_tensor<double>("B", 4, 5, 2);
    auto C          = create_zero_tensor<double>("C", 3, 2);
    auto C_expected = create_zero_tensor<double>("Ce", 3, 2);

    tensor_algebra::einsum(Indices{i, l}, &C_expected, Indices{i, j, k}, A, Indices{j, k, l}, B);

    cg::Graph graph("rank3_graph");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("il <- ijk ; jkl", &C, A, B);
    }

    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t ll = 0; ll < 2; ll++) {
            REQUIRE(std::abs(C(ii, ll) - C_expected(ii, ll)) < 1e-10);
        }
    }
}

TEST_CASE("String einsum - rank-4 contraction", "[ComputeGraph][StringEinsum][HighRank]") {
    // C[i,j,k,l] = A[i,j,p] * B[k,l,p]  (contract over p)
    auto A          = create_random_tensor<double>("A", 2, 3, 4);
    auto B          = create_random_tensor<double>("B", 2, 3, 4);
    auto C          = create_zero_tensor<double>("C", 2, 3, 2, 3);
    auto C_expected = create_zero_tensor<double>("Ce", 2, 3, 2, 3);

    tensor_algebra::einsum(Indices{i, j, k, l}, &C_expected, Indices{i, j, p}, A, Indices{k, l, p}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("ijkl <- ijp ; klp", &C, A, B);

    for (size_t ii = 0; ii < 2; ii++) {
        for (size_t jj = 0; jj < 3; jj++) {
            for (size_t kk = 0; kk < 2; kk++) {
                for (size_t ll = 0; ll < 3; ll++) {
                    REQUIRE(std::abs(C(ii, jj, kk, ll) - C_expected(ii, jj, kk, ll)) < 1e-10);
                }
            }
        }
    }
}

TEST_CASE("String einsum - higher-rank with prefactors", "[ComputeGraph][StringEinsum][HighRank]") {
    auto A          = create_random_tensor<double>("A", 3, 4, 5);
    auto B          = create_random_tensor<double>("B", 4, 5, 2);
    auto C          = create_random_tensor<double>("C", 3, 2);
    auto C_expected = Tensor<double, 2>(C);

    tensor_algebra::einsum(2.0, Indices{i, l}, &C_expected, 3.0, Indices{i, j, k}, A, Indices{j, k, l}, B);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::einsum("il <- ijk ; jkl", 2.0, &C, 3.0, A, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t ll = 0; ll < 2; ll++) {
            REQUIRE(std::abs(C(ii, ll) - C_expected(ii, ll)) < 1e-10);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// String-based permute tests
// ═════════════════════════════════════════════════════════════════════════════

TEST_CASE("String permute - matrix transpose, direct execute", "[ComputeGraph][StringPermute]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto C = create_zero_tensor<double>("C", 3, 4);

    auto C_expected = create_zero_tensor<double>("Ce", 3, 4);
    tensor_algebra::permute(Indices{j, i}, &C_expected, Indices{i, j}, A);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::permute("ji <- ij", &C, A);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String permute - with prefactors", "[ComputeGraph][StringPermute]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto C = create_random_tensor<double>("C", 3, 4);

    auto C_expected = Tensor<double, 2>(C);
    tensor_algebra::permute(2.0, Indices{j, i}, &C_expected, 3.0, Indices{i, j}, A);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::permute("ji <- ij", 2.0, &C, 3.0, A);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String permute - rank-3 transpose", "[ComputeGraph][StringPermute]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto C = create_zero_tensor<double>("C", 5, 3, 4);

    auto C_expected = create_zero_tensor<double>("Ce", 5, 3, 4);
    tensor_algebra::permute(Indices{k, i, j}, &C_expected, Indices{i, j, k}, A);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture)
    cg::permute("kij <- ijk", &C, A);

    for (size_t ii = 0; ii < 5; ii++)
        for (size_t jj = 0; jj < 3; jj++)
            for (size_t kk = 0; kk < 4; kk++)
                REQUIRE(std::abs(C(ii, jj, kk) - C_expected(ii, jj, kk)) < 1e-12);
}

TEST_CASE("String permute - graph capture and execute", "[ComputeGraph][StringPermute]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto C = create_zero_tensor<double>("C", 3, 4);

    auto C_expected = create_zero_tensor<double>("Ce", 3, 4);
    tensor_algebra::permute(Indices{j, i}, &C_expected, Indices{i, j}, A);

    cg::Graph graph("string_permute_test");
    {
        cg::CaptureGuard const guard(graph);
        cg::permute("ji <- ij", &C, A);
    }

    REQUIRE(graph.num_nodes() == 1);
    graph.execute();

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            REQUIRE(std::abs(C(ii, jj) - C_expected(ii, jj)) < 1e-12);
        }
    }
}

TEST_CASE("String permute - identity (no reorder)", "[ComputeGraph][StringPermute]") {
    auto A = create_random_tensor<double>("A", 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 4);

    // NOLINTNEXTLINE(einsums-cg-call-outside-capture, einsums-redundant-permute)
    cg::permute("ij <- ij", &C, A);

    for (size_t ii = 0; ii < 3; ii++)
        for (size_t jj = 0; jj < 4; jj++)
            REQUIRE(std::abs(C(ii, jj) - A(ii, jj)) < 1e-12);
}
