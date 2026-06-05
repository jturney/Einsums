//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("potrf", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Create a symmetric positive definite matrix: A = I + v*v^T
    // Use column-major for direct BLAS calls.
    auto A = create_tensor<T>(false, "A", N, N);
    A.zero();

    // Start with identity
    for (int i = 0; i < N; i++) {
        A(i, i) = T{1.0};
    }

    // Add v*v^T where v = {1, 2, 3, 4}
    T v[] = {1.0, 2.0, 3.0, 4.0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) += v[i] * v[j];
        }
    }

    // Save original A
    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    A_orig      = A;

    blas::int_t lda = N; // column-major: lda = N

    // Cholesky factorization: A = L*L^T, uplo='L'
    auto info = blas::potrf<T>('L', N, A.data(), lda);
    REQUIRE(info == 0);

    // Extract L (lower triangle of A)
    auto L = create_tensor<T>(false, "L", N, N);
    L.zero();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            L(i, j) = A(i, j);
        }
    }

    // Verify L*L^T = A_orig using element-wise loops
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T sum = T{0.0};
            for (int k = 0; k < N; k++) {
                sum += L(i, k) * L(j, k); // L * L^T
            }
            REQUIRE(sum == Catch::Approx(A_orig(i, j)).margin(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("potrs", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Use column-major for direct BLAS calls.
    auto A = create_tensor<T>(false, "A", N, N);
    A.zero();
    for (int i = 0; i < N; i++) {
        A(i, i) = T{1.0};
    }
    T v[] = {1.0, 2.0, 3.0, 4.0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) += v[i] * v[j];
        }
    }

    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    A_orig      = A;

    // RHS vector b (stored as Nx1 matrix for potrs)
    auto b  = create_tensor<T>(false, "b", N, 1);
    b(0, 0) = 5.0;
    b(1, 0) = 11.0;
    b(2, 0) = 17.0;
    b(3, 0) = 23.0;

    auto b_orig = create_tensor<T>(false, "b_orig", N, 1);
    b_orig      = b;

    blas::int_t lda = N;
    blas::int_t ldb = N;

    // Factor A
    auto info = blas::potrf<T>('L', N, A.data(), lda);
    REQUIRE(info == 0);

    // Solve A*x = b
    info = blas::potrs<T>('L', N, 1, A.data(), lda, b.data(), ldb);
    REQUIRE(info == 0);

    // Verify A_orig * x = b_orig using element-wise loops
    for (int i = 0; i < N; i++) {
        T sum = T{0.0};
        for (int k = 0; k < N; k++) {
            sum += A_orig(i, k) * b(k, 0);
        }
        REQUIRE(sum == Catch::Approx(b_orig(i, 0)).margin(1e-6));
    }
}

TEMPLATE_TEST_CASE("potri", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Use column-major for direct BLAS calls.
    auto A = create_tensor<T>(false, "A", N, N);
    A.zero();
    for (int i = 0; i < N; i++) {
        A(i, i) = T{1.0};
    }
    T v[] = {1.0, 2.0, 3.0, 4.0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) += v[i] * v[j];
        }
    }

    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    A_orig      = A;

    blas::int_t lda = N;

    // Factor A
    auto info = blas::potrf<T>('L', N, A.data(), lda);
    REQUIRE(info == 0);

    // Invert
    info = blas::potri<T>('L', N, A.data(), lda);
    REQUIRE(info == 0);

    // potri only fills the lower triangle; symmetrize A_inv
    auto A_inv = create_tensor<T>(false, "A_inv", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            A_inv(i, j) = A(i, j);
            A_inv(j, i) = A(i, j);
        }
    }

    // Verify A_orig * A_inv = I using element-wise loops
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T sum = T{0.0};
            for (int k = 0; k < N; k++) {
                sum += A_orig(i, k) * A_inv(k, j);
            }
            T expected = (i == j) ? T{1.0} : T{0.0};
            REQUIRE(sum == Catch::Approx(expected).margin(1e-5));
        }
    }
}
