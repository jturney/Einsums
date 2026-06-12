//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Correctness tests for the Pack-A/Pack-B (BLIS-style) contraction path.
//
// Tests:
//   1. Topology recognition: compute_packing_topology() identifies valid/invalid cases.
//   2. Correctness for C[i,l] += A[i,j,k] * B[j,k,l] at N=32 (K = 32² = 1024 > KC = 256,
//      so multiple kc tiles are exercised).
//   3. Correctness with alpha/beta scaling.

#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;
using namespace einsums::packed_gemm;

namespace {

/// Fill tensor with sequential values starting from `start`.
template <typename TensorType>
void fill_seq(TensorType &T, double start = 1.0) {
    for (size_t idx = 0; idx < T.size(); ++idx) {
        T.data()[idx] = static_cast<typename TensorType::ValueType>(start + static_cast<double>(idx));
    }
}

/// Fill tensor with a constant value.
template <typename TensorType>
void fill_const(TensorType &T, double val) {
    for (size_t idx = 0; idx < T.size(); ++idx) {
        T.data()[idx] = static_cast<typename TensorType::ValueType>(val);
    }
}

/// Element-wise comparison with tolerance.
template <typename TA, typename TB>
bool tensors_close(TA const &A, TB const &B, double tol = 1e-8) {
    if (A.size() != B.size()) {
        return false;
    }
    for (size_t k = 0; k < A.size(); ++k) {
        double diff = std::abs(static_cast<double>(A.data()[k]) - static_cast<double>(B.data()[k]));
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

/// Build a ContractionKey for C[i,l] += A[i,j,k] * B[j,k,l] with dim N.
ContractionKey make_rank3_key(int64_t N) {
    ContractionSpec spec;
    spec.c_indices      = {"i", "l"};
    spec.a_indices      = {"i", "j", "k"};
    spec.b_indices      = {"j", "k", "l"};
    spec.target_indices = {"i", "l"};
    spec.link_indices   = {"j", "k"};
    spec.all_indices    = {"i", "l", "j", "k"};
    spec.scalar_type    = ScalarType::Float64;
    spec.conj_a         = false;
    spec.conj_b         = false;
    spec.scalar_output  = false;

    ContractionKey key;
    key.spec        = spec;
    key.a_desc      = {3, ScalarType::Float64};
    key.b_desc      = {3, ScalarType::Float64};
    key.c_desc      = {2, ScalarType::Float64};
    key.target_dims = {N, N};
    key.link_dims   = {N, N};
    return key;
}

/// Build a ContractionKey for C[i,i] += A[i,i] * B[i,i] (Hadamard, invalid for packing).
ContractionKey make_hadamard_key(int64_t N) {
    ContractionSpec spec;
    spec.c_indices      = {"i"};
    spec.a_indices      = {"i", "i"};
    spec.b_indices      = {"i", "i"};
    spec.target_indices = {"i"};
    spec.link_indices   = {};
    spec.all_indices    = {"i"};
    spec.scalar_type    = ScalarType::Float64;
    spec.conj_a         = false;
    spec.conj_b         = false;
    spec.scalar_output  = false;

    ContractionKey key;
    key.spec        = spec;
    key.a_desc      = {2, ScalarType::Float64};
    key.b_desc      = {2, ScalarType::Float64};
    key.c_desc      = {1, ScalarType::Float64};
    key.target_dims = {N};
    key.link_dims   = {};
    return key;
}

/// Build a ContractionKey for scalar output (C += A[i]*B[i]), invalid for packing.
ContractionKey make_scalar_output_key(int64_t N) {
    ContractionSpec spec;
    spec.c_indices      = {};
    spec.a_indices      = {"i"};
    spec.b_indices      = {"i"};
    spec.target_indices = {};
    spec.link_indices   = {"i"};
    spec.all_indices    = {"i"};
    spec.scalar_type    = ScalarType::Float64;
    spec.scalar_output  = true;

    ContractionKey key;
    key.spec      = spec;
    key.a_desc    = {1, ScalarType::Float64};
    key.b_desc    = {1, ScalarType::Float64};
    key.c_desc    = {0, ScalarType::Float64};
    key.link_dims = {N};
    return key;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Topology recognition tests
// ---------------------------------------------------------------------------

TEST_CASE("PackingPlan topology recognition", "[mlir][packing][topology]") {
    SECTION("C[i,l] += A[i,j,k]*B[j,k,l] should be valid") {
        auto plan = compute_packing_topology(make_rank3_key(8));
        REQUIRE(plan.valid);
        REQUIRE(plan.m_dims.size() == 1);
        REQUIRE(plan.n_dims.size() == 1);
        REQUIRE(plan.k_dims_in_a.size() == 2);
        REQUIRE(plan.k_dims_in_b.size() == 2);
        REQUIRE(plan.M_total == 8);
        REQUIRE(plan.N_total == 8);
        REQUIRE(plan.K_total == 64); // 8 * 8
    }

    SECTION("Hadamard (repeated A indices) should be invalid") {
        auto plan = compute_packing_topology(make_hadamard_key(4));
        REQUIRE_FALSE(plan.valid);
    }

    SECTION("Scalar output should be invalid") {
        auto plan = compute_packing_topology(make_scalar_output_key(4));
        REQUIRE_FALSE(plan.valid);
    }
}

// ---------------------------------------------------------------------------
// Correctness tests via einsum dispatch
// ---------------------------------------------------------------------------

TEST_CASE("MLIR packing path correctness N=32", "[mlir][packing][correctness]") {
    // With N=32: M=32, N=32, K=32*32=1024.
    // KC=256 → 4 kc tiles, so the K-tiling path is exercised.
    SECTION("C[i,l] += A[i,j,k] * B[j,k,l]  N=32, beta=0") {
        constexpr size_t  N = 32;
        Tensor<double, 2> C_mlir{"C_mlir", N, N};
        Tensor<double, 3> A{"A", N, N, N};
        Tensor<double, 3> B{"B", N, N, N};
        Tensor<double, 2> C_ref{"C_ref", N, N};

        fill_seq(A, 0.0);
        fill_seq(B, 1.0);
        C_mlir.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{i, j, k}, A, Indices{j, k, l}, B));

        // Reference: plain triple-loop
        for (size_t ii = 0; ii < N; ++ii) {
            for (size_t ll = 0; ll < N; ++ll) {
                for (size_t jj = 0; jj < N; ++jj) {
                    for (size_t kk = 0; kk < N; ++kk) {
                        C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);
                    }
                }
            }
        }

        REQUIRE(tensors_close(C_mlir, C_ref));
    }

    SECTION("C[i,l] += A[i,j,k] * B[j,k,l]  N=32, non-zero C, beta=1") {
        // Test that existing C values are accumulated (beta=1, alpha=1).
        constexpr size_t  N = 32;
        Tensor<double, 2> C_mlir{"C_mlir", N, N};
        Tensor<double, 3> A{"A", N, N, N};
        Tensor<double, 3> B{"B", N, N, N};
        Tensor<double, 2> C_ref{"C_ref", N, N};

        fill_seq(A, 0.0);
        fill_seq(B, 1.0);
        fill_const(C_mlir, 1.0); // pre-fill C with ones
        fill_const(C_ref, 1.0);

        REQUIRE_NOTHROW(einsum(1.0, Indices{i, l}, &C_mlir, 1.0, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii) {
            for (size_t ll = 0; ll < N; ++ll) {
                // C_ref already has 1.0; add the contraction
                for (size_t jj = 0; jj < N; ++jj) {
                    for (size_t kk = 0; kk < N; ++kk) {
                        C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);
                    }
                }
            }
        }

        REQUIRE(tensors_close(C_mlir, C_ref));
    }

    SECTION("C[i,l] += A[i,j,k] * B[j,k,l]  N=32, alpha=2, beta=3") {
        constexpr size_t  N = 32;
        Tensor<double, 2> C_mlir{"C_mlir", N, N};
        Tensor<double, 3> A{"A", N, N, N};
        Tensor<double, 3> B{"B", N, N, N};
        Tensor<double, 2> C_ref{"C_ref", N, N};

        fill_seq(A, 0.0);
        fill_seq(B, 1.0);
        fill_const(C_mlir, 1.0);
        fill_const(C_ref, 1.0);

        REQUIRE_NOTHROW(einsum(3.0, Indices{i, l}, &C_mlir, 2.0, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii) {
            for (size_t ll = 0; ll < N; ++ll) {
                double sum = 0.0;
                for (size_t jj = 0; jj < N; ++jj) {
                    for (size_t kk = 0; kk < N; ++kk) {
                        sum += A(ii, jj, kk) * B(jj, kk, ll);
                    }
                }
                C_ref(ii, ll) = 3.0 * C_ref(ii, ll) + 2.0 * sum;
            }
        }

        REQUIRE(tensors_close(C_mlir, C_ref));
    }
}

TEST_CASE("MLIR packing path correctness N=8 (no KC tiling)", "[mlir][packing][correctness]") {
    // K=64 < KC=256 → single kc tile (tests the non-tiling sub-path).
    SECTION("C[i,l] += A[i,j,k] * B[j,k,l]  N=8") {
        constexpr size_t  N = 8;
        Tensor<double, 2> C_mlir{"C_mlir", N, N};
        Tensor<double, 3> A{"A", N, N, N};
        Tensor<double, 3> B{"B", N, N, N};
        Tensor<double, 2> C_ref{"C_ref", N, N};

        fill_seq(A, 0.0);
        fill_seq(B, 1.0);
        C_mlir.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii) {
            for (size_t ll = 0; ll < N; ++ll) {
                for (size_t jj = 0; jj < N; ++jj) {
                    for (size_t kk = 0; kk < N; ++kk) {
                        C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);
                    }
                }
            }
        }

        REQUIRE(tensors_close(C_mlir, C_ref));
    }
}

// ---------------------------------------------------------------------------
// Zero-copy layout tests
//
// The flatten+GEMM path can skip copying A and/or B when their memory layout
// already matches the GEMM-ready format (col-major M×K for A, row-major K×N
// for B).  These tests exercise all four combinations.
//
// For column-major tensors with all dims = N:
//   A zero-copy requires M dim at position 0 (stride 1) in A
//   B zero-copy requires N dim at position 0 (stride 1) in B
// ---------------------------------------------------------------------------

TEST_CASE("Zero-copy layout: both A and B zero-copy", "[mlir][packing][zerocopy]") {
    // C[i,l] = A[i,j,k] * B[l,j,k]
    // A: M=i at pos 0 (stride 1) → zero-copy
    // B: N=l at pos 0 (stride 1) → zero-copy
    constexpr size_t  N = 32;
    Tensor<double, 2> C_mlir{"C_mlir", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 2> C_ref{"C_ref", N, N};

    fill_seq(A, 0.0);
    fill_seq(B, 1.0);
    C_mlir.zero();
    C_ref.zero();

    REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{i, j, k}, A, Indices{l, j, k}, B));

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    C_ref(ii, ll) += A(ii, jj, kk) * B(ll, jj, kk);

    REQUIRE(tensors_close(C_mlir, C_ref));
}

TEST_CASE("Zero-copy layout: only A zero-copy", "[mlir][packing][zerocopy]") {
    // C[i,l] = A[i,j,k] * B[j,k,l]  (existing benchmark pattern)
    // A: M=i at pos 0 (stride 1) → zero-copy
    // B: N=l at pos 2 (stride N²) → needs copy
    constexpr size_t  N = 32;
    Tensor<double, 2> C_mlir{"C_mlir", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 2> C_ref{"C_ref", N, N};

    fill_seq(A, 0.0);
    fill_seq(B, 1.0);
    C_mlir.zero();
    C_ref.zero();

    REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{i, j, k}, A, Indices{j, k, l}, B));

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);

    REQUIRE(tensors_close(C_mlir, C_ref));
}

TEST_CASE("Zero-copy layout: only B zero-copy", "[mlir][packing][zerocopy]") {
    // C[i,l] = A[j,i,k] * B[l,j,k]
    // A: M=i at pos 1 (stride N) → needs copy
    // B: N=l at pos 0 (stride 1) → zero-copy
    constexpr size_t  N = 32;
    Tensor<double, 2> C_mlir{"C_mlir", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 2> C_ref{"C_ref", N, N};

    fill_seq(A, 0.0);
    fill_seq(B, 1.0);
    C_mlir.zero();
    C_ref.zero();

    REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{j, i, k}, A, Indices{l, j, k}, B));

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    C_ref(ii, ll) += A(jj, ii, kk) * B(ll, jj, kk);

    REQUIRE(tensors_close(C_mlir, C_ref));
}

TEST_CASE("Zero-copy layout: neither zero-copy", "[mlir][packing][zerocopy]") {
    // C[i,l] = A[j,i,k] * B[j,k,l]
    // A: M=i at pos 1 (stride N) → needs copy
    // B: N=l at pos 2 (stride N²) → needs copy
    constexpr size_t  N = 32;
    Tensor<double, 2> C_mlir{"C_mlir", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 2> C_ref{"C_ref", N, N};

    fill_seq(A, 0.0);
    fill_seq(B, 1.0);
    C_mlir.zero();
    C_ref.zero();

    REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_mlir, Indices{j, i, k}, A, Indices{j, k, l}, B));

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t ll = 0; ll < N; ++ll)
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    C_ref(ii, ll) += A(jj, ii, kk) * B(jj, kk, ll);

    REQUIRE(tensors_close(C_mlir, C_ref));
}

TEST_CASE("Zero-copy layout: both zero-copy with alpha/beta", "[mlir][packing][zerocopy]") {
    // C[i,l] = 3*C[i,l] + 2 * A[i,j,k] * B[l,j,k]  (both zero-copy + scaling)
    constexpr size_t  N = 32;
    Tensor<double, 2> C_mlir{"C_mlir", N, N};
    Tensor<double, 3> A{"A", N, N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 2> C_ref{"C_ref", N, N};

    fill_seq(A, 0.0);
    fill_seq(B, 1.0);
    fill_const(C_mlir, 1.0);
    fill_const(C_ref, 1.0);

    REQUIRE_NOTHROW(einsum(3.0, Indices{i, l}, &C_mlir, 2.0, Indices{i, j, k}, A, Indices{l, j, k}, B));

    for (size_t ii = 0; ii < N; ++ii) {
        for (size_t ll = 0; ll < N; ++ll) {
            double sum = 0.0;
            for (size_t jj = 0; jj < N; ++jj)
                for (size_t kk = 0; kk < N; ++kk)
                    sum += A(ii, jj, kk) * B(ll, jj, kk);
            C_ref(ii, ll) = 3.0 * C_ref(ii, ll) + 2.0 * sum;
        }
    }

    REQUIRE(tensors_close(C_mlir, C_ref));
}

// ---------------------------------------------------------------------------
// TensorView tests
//
// Verify that PackedGemm handles TensorView inputs correctly.
// Views have strides from the parent tensor, which may differ from a
// dense tensor of the view's shape.
// ---------------------------------------------------------------------------

TEST_CASE("PackedGemm with TensorView inputs", "[mlir][packing][view]") {
    SECTION("C[i,l] += A_view[i,j,k] * B_view[j,k,l] — sub-block views") {
        constexpr size_t N  = 16;
        constexpr size_t NP = 32; // parent tensor is larger than the view

        Tensor<double, 3> A_parent{"A_parent", NP, NP, NP};
        Tensor<double, 3> B_parent{"B_parent", NP, NP, NP};
        fill_seq(A_parent, 0.0);
        fill_seq(B_parent, 1.0);

        // Take views of the first N elements along each dimension.
        auto A_view = A_parent(Range{0, N}, Range{0, N}, Range{0, N});
        auto B_view = B_parent(Range{0, N}, Range{0, N}, Range{0, N});

        Tensor<double, 2> C_packed{"C_packed", N, N};
        Tensor<double, 2> C_ref{"C_ref", N, N};
        C_packed.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_packed, Indices{i, j, k}, A_view, Indices{j, k, l}, B_view));

        // Reference: manual loop over the view.
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        C_ref(ii, ll) += A_parent(ii, jj, kk) * B_parent(jj, kk, ll);

        REQUIRE(tensors_close(C_packed, C_ref));
    }

    SECTION("C_view[i,l] += A[i,j,k] * B[j,k,l] — output is a TensorView") {
        constexpr size_t N  = 8;
        constexpr size_t NP = 16;

        Tensor<double, 3> A{"A", N, N, N};
        Tensor<double, 3> B{"B", N, N, N};
        Tensor<double, 2> C_parent{"C_parent", NP, NP};
        fill_seq(A, 0.0);
        fill_seq(B, 1.0);
        C_parent.zero();

        auto C_view = C_parent(Range{0, N}, Range{0, N});

        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_view, Indices{i, j, k}, A, Indices{j, k, l}, B));

        Tensor<double, 2> C_ref{"C_ref", N, N};
        C_ref.zero();
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);

        // Compare element-by-element (can't use tensors_close with data() because
        // TensorView memory is not contiguous).
        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                REQUIRE(std::abs(C_view(ii, ll) - C_ref(ii, ll)) < 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Complex type tests
//
// Verify that PackedGemm handles std::complex<double> correctly,
// both with and without conjugation.
// ---------------------------------------------------------------------------

/// Element-wise comparison for complex tensors using relative tolerance.
template <typename TA, typename TB>
bool complex_tensors_close(TA const &cA, TB const &cB, double rel_tol = 1e-10) {
    if (cA.size() != cB.size()) {
        return false;
    }
    for (size_t idx = 0; idx < cA.size(); ++idx) {
        auto   diff = std::abs(cA.data()[idx] - cB.data()[idx]);
        auto   mag  = std::max(std::abs(cA.data()[idx]), std::abs(cB.data()[idx]));
        double tol  = std::max(1e-10, rel_tol * mag);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

TEST_CASE("PackedGemm with complex<double>", "[mlir][packing][complex]") {
    using cd = std::complex<double>;

    SECTION("C[i,l] += A[i,j,k] * B[j,k,l] — no conjugation") {
        constexpr size_t N = 16;
        Tensor<cd, 2>    C_packed{"C_packed", N, N};
        Tensor<cd, 3>    A{"A", N, N, N};
        Tensor<cd, 3>    B{"B", N, N, N};
        Tensor<cd, 2>    C_ref{"C_ref", N, N};

        for (size_t idx = 0; idx < A.size(); ++idx) {
            A.data()[idx] = cd(static_cast<double>(idx), static_cast<double>(idx) * 0.5);
        }
        for (size_t idx = 0; idx < B.size(); ++idx) {
            B.data()[idx] = cd(static_cast<double>(idx) * 0.3, static_cast<double>(idx) * -0.2);
        }
        C_packed.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, l}, &C_packed, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        C_ref(ii, ll) += A(ii, jj, kk) * B(jj, kk, ll);

        REQUIRE(complex_tensors_close(C_packed, C_ref));
    }

    SECTION("C[i,l] += conj(A[i,j,k]) * B[j,k,l] — ConjA") {
        constexpr size_t N = 8;
        Tensor<cd, 2>    C_packed{"C_packed", N, N};
        Tensor<cd, 3>    A{"A", N, N, N};
        Tensor<cd, 3>    B{"B", N, N, N};
        Tensor<cd, 2>    C_ref{"C_ref", N, N};

        for (size_t idx = 0; idx < A.size(); ++idx)
            A.data()[idx] = cd(static_cast<double>(idx), static_cast<double>(idx) * 0.5);
        for (size_t idx = 0; idx < B.size(); ++idx)
            B.data()[idx] = cd(static_cast<double>(idx) * 0.3, static_cast<double>(idx) * -0.2);
        C_packed.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum<true, false>(Indices{i, l}, &C_packed, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        C_ref(ii, ll) += std::conj(A(ii, jj, kk)) * B(jj, kk, ll);

        REQUIRE(complex_tensors_close(C_packed, C_ref));
    }

    SECTION("C[i,l] += A[i,j,k] * conj(B[j,k,l]) — ConjB") {
        constexpr size_t N = 8;
        Tensor<cd, 2>    C_packed{"C_packed", N, N};
        Tensor<cd, 3>    A{"A", N, N, N};
        Tensor<cd, 3>    B{"B", N, N, N};
        Tensor<cd, 2>    C_ref{"C_ref", N, N};

        for (size_t idx = 0; idx < A.size(); ++idx)
            A.data()[idx] = cd(static_cast<double>(idx), static_cast<double>(idx) * 0.5);
        for (size_t idx = 0; idx < B.size(); ++idx)
            B.data()[idx] = cd(static_cast<double>(idx) * 0.3, static_cast<double>(idx) * -0.2);
        C_packed.zero();
        C_ref.zero();

        REQUIRE_NOTHROW(einsum<false, true>(Indices{i, l}, &C_packed, Indices{i, j, k}, A, Indices{j, k, l}, B));

        for (size_t ii = 0; ii < N; ++ii)
            for (size_t ll = 0; ll < N; ++ll)
                for (size_t jj = 0; jj < N; ++jj)
                    for (size_t kk = 0; kk < N; ++kk)
                        C_ref(ii, ll) += A(ii, jj, kk) * std::conj(B(jj, kk, ll));

        REQUIRE(complex_tensors_close(C_packed, C_ref));
    }
}
