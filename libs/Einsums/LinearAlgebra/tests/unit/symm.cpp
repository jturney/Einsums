//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("symm", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int M = 4;
    constexpr int N = 3;

    // Column-major tensors
    auto A  = create_tensor<T>(false, "A", M, M);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 3.0;
    A(0, 3) = 0.5;
    A(1, 0) = 1.0;
    A(1, 1) = 4.0;
    A(1, 2) = 2.0;
    A(1, 3) = 1.0;
    A(2, 0) = 3.0;
    A(2, 1) = 2.0;
    A(2, 2) = 5.0;
    A(2, 3) = 3.0;
    A(3, 0) = 0.5;
    A(3, 1) = 1.0;
    A(3, 2) = 3.0;
    A(3, 3) = 6.0;

    auto B  = create_tensor<T>(false, "B", M, N);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(0, 2) = 3.0;
    B(1, 0) = 4.0;
    B(1, 1) = 5.0;
    B(1, 2) = 6.0;
    B(2, 0) = 7.0;
    B(2, 1) = 8.0;
    B(2, 2) = 9.0;
    B(3, 0) = 10.0;
    B(3, 1) = 11.0;
    B(3, 2) = 12.0;

    auto C_symm = create_tensor<T>(false, "C_symm", M, N);
    C_symm.zero();

    blas::int_t lda = A.impl().get_lda();
    blas::int_t ldb = B.impl().get_lda();
    blas::int_t ldc = C_symm.impl().get_lda();

    // C = A*B where A is symmetric (use upper triangle)
    blas::symm<T>('L', 'U', M, N, T{1.0}, A.data(), lda, B.data(), ldb, T{0.0}, C_symm.data(), ldc);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T expected = T{0.0};
            for (int k = 0; k < M; k++) {
                expected += A(i, k) * B(k, j);
            }
            REQUIRE(C_symm(i, j) == Catch::Approx(expected).margin(1e-6));
        }
    }
}
