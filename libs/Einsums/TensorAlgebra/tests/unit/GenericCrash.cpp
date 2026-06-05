//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

TEST_CASE("generic_asymmetric_einsum", "[generic]") {
    // Test: C[i,j,k] += A[i,l] * B[l,j,k] via full einsum (uses GEMM, not generic)
    constexpr size_t  N = 4;
    Tensor<double, 3> C{"C", N, N, N};
    Tensor<double, 2> A{"A", N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 3> ref{"ref", N, N, N};

    for (size_t x = 0; x < N; ++x)
        for (size_t y = 0; y < N; ++y) {
            A(x, y) = (double)(x * N + y + 1);
            for (size_t z = 0; z < N; ++z)
                B(x, y, z) = (double)(x * N * N + y * N + z + 1);
        }

    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    ref(ii, jj, kk) += A(ii, ll) * B(ll, jj, kk);

    // Full einsum (should use GEMM)
    C.zero();
    tensor_algebra::detail::AlgorithmChoice alg;
    einsum(Indices{i, j, k}, &C, Indices{i, l}, A, Indices{l, j, k}, B, &alg);
    REQUIRE(alg == tensor_algebra::detail::GEMM);

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                REQUIRE_THAT(C(ii, jj, kk), Catch::Matchers::WithinRel(ref(ii, jj, kk), 0.0001));
}

TEST_CASE("generic_asymmetric_forced", "[generic]") {
    // Force generic algorithm: C[i,j,k] += A[i,l] * B[l,j,k]
    constexpr size_t  N = 4;
    Tensor<double, 3> C{"C", N, N, N};
    Tensor<double, 2> A{"A", N, N};
    Tensor<double, 3> B{"B", N, N, N};
    Tensor<double, 3> ref{"ref", N, N, N};

    for (size_t x = 0; x < N; ++x)
        for (size_t y = 0; y < N; ++y) {
            A(x, y) = (double)(x * N + y + 1);
            for (size_t z = 0; z < N; ++z)
                B(x, y, z) = (double)(x * N * N + y * N + z + 1);
        }

    ref.zero();
    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                for (size_t ll = 0; ll < N; ++ll)
                    ref(ii, jj, kk) += A(ii, ll) * B(ll, jj, kk);

    // OnlyUseGenericAlgorithm=true, DryRun=false, ConjA=false, ConjB=false
    C.zero();
    auto c_idx = Indices{i, j, k};
    auto a_idx = Indices{i, l};
    auto b_idx = Indices{l, j, k};
    tensor_algebra::detail::einsum<true, false, false, false>(0.0, c_idx, &C, 1.0, a_idx, A, b_idx, B);

    for (size_t ii = 0; ii < N; ++ii)
        for (size_t jj = 0; jj < N; ++jj)
            for (size_t kk = 0; kk < N; ++kk)
                REQUIRE_THAT(C(ii, jj, kk), Catch::Matchers::WithinRel(ref(ii, jj, kk), 0.0001));
}
