//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("getrs", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Use column-major tensors for direct BLAS calls.
    // For non-square B (Nx1), row-major ldb=1 is too small for BLAS (needs ldb >= N).
    auto A  = create_tensor<T>(false, "A", N, N);
    A(0, 0) = 1.80;
    A(0, 1) = 5.25;
    A(0, 2) = 1.58;
    A(0, 3) = -1.11;
    A(1, 0) = 2.88;
    A(1, 1) = -2.95;
    A(1, 2) = -2.69;
    A(1, 3) = -0.66;
    A(2, 0) = 2.05;
    A(2, 1) = -0.95;
    A(2, 2) = -2.90;
    A(2, 3) = -0.59;
    A(3, 0) = -0.89;
    A(3, 1) = -3.80;
    A(3, 2) = -1.04;
    A(3, 3) = 0.80;

    // Save original A
    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    A_orig      = A;

    // RHS b (stored as Nx1 matrix for getrs)
    auto b  = create_tensor<T>(false, "b", N, 1);
    b(0, 0) = 9.52;
    b(1, 0) = 24.35;
    b(2, 0) = 0.77;
    b(3, 0) = -6.22;

    auto b_orig = create_tensor<T>(false, "b_orig", N, 1);
    b_orig      = b;

    std::vector<blas::int_t> ipiv(N);

    blas::int_t lda = N; // column-major: lda = N
    blas::int_t ldb = N; // column-major: ldb = N

    // Factor A
    auto info = blas::getrf<T>(N, N, A.data(), lda, ipiv.data());
    REQUIRE(info == 0);

    // Solve A*x = b (column-major, no row-major fixups needed)
    info = blas::getrs<T>('N', N, 1, A.data(), lda, ipiv.data(), b.data(), ldb);
    REQUIRE(info == 0);

    // Verify A_orig * x = b_orig using element-wise loops
    for (int i = 0; i < N; i++) {
        T sum = T{0.0};
        for (int k = 0; k < N; k++) {
            sum += A_orig(i, k) * b(k, 0);
        }
        REQUIRE(sum == Catch::Approx(b_orig(i, 0)).margin(1e-5));
    }
}
