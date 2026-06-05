//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("trsm", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N    = 4;
    constexpr int NRHS = 3;

    // Use column-major tensors for direct BLAS calls.
    // For B (NxNRHS), row-major ldb=NRHS < N, which violates BLAS requirements.

    // Upper triangular matrix A (4x4)
    auto A = create_tensor<T>(false, "A", N, N);
    A.zero();
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(0, 2) = -1.0;
    A(0, 3) = 3.0;
    A(1, 1) = 3.0;
    A(1, 2) = 2.0;
    A(1, 3) = 1.0;
    A(2, 2) = 4.0;
    A(2, 3) = -2.0;
    A(3, 3) = 5.0;

    // RHS matrix B (4x3)
    auto B  = create_tensor<T>(false, "B", N, NRHS);
    B(0, 0) = 10.0;
    B(0, 1) = 7.0;
    B(0, 2) = -3.0;
    B(1, 0) = 12.0;
    B(1, 1) = -1.0;
    B(1, 2) = 5.0;
    B(2, 0) = 8.0;
    B(2, 1) = 4.0;
    B(2, 2) = 6.0;
    B(3, 0) = 15.0;
    B(3, 1) = 3.0;
    B(3, 2) = -2.0;

    // Save original B for verification
    auto B_orig = create_tensor<T>(false, "B_orig", N, NRHS);
    B_orig      = B;

    blas::int_t lda = N;
    blas::int_t ldb = N;

    // Solve A * X = B => X = A^{-1} * B, result overwrites B
    // Column-major: side='L', uplo='U', transa='N', diag='N'
    blas::trsm<T>('L', 'U', 'N', 'N', N, NRHS, T{1.0}, A.data(), lda, B.data(), ldb);

    // Verify: A * X should equal B_orig using element-wise loops
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NRHS; j++) {
            T sum = T{0.0};
            for (int k = 0; k < N; k++) {
                sum += A(i, k) * B(k, j);
            }
            REQUIRE(sum == Catch::Approx(B_orig(i, j)).margin(1e-6));
        }
    }
}
