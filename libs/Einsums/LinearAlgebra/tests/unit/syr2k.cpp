//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("syr2k", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;
    constexpr int K = 3;

    // Column-major tensors
    auto A  = create_tensor<T>(false, "A", N, K);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    A(2, 0) = 7.0;
    A(2, 1) = 8.0;
    A(2, 2) = 9.0;
    A(3, 0) = 10.0;
    A(3, 1) = 11.0;
    A(3, 2) = 12.0;

    auto B  = create_tensor<T>(false, "B", N, K);
    B(0, 0) = 2.0;
    B(0, 1) = 1.0;
    B(0, 2) = 0.0;
    B(1, 0) = 3.0;
    B(1, 1) = 2.0;
    B(1, 2) = 1.0;
    B(2, 0) = 4.0;
    B(2, 1) = 3.0;
    B(2, 2) = 2.0;
    B(3, 0) = 5.0;
    B(3, 1) = 4.0;
    B(3, 2) = 3.0;

    auto C_syr2k = create_tensor<T>(false, "C_syr2k", N, N);
    C_syr2k.zero();

    blas::int_t lda = A.impl().get_lda();
    blas::int_t ldb = B.impl().get_lda();
    blas::int_t ldc = C_syr2k.impl().get_lda();

    // C = A*B^T + B*A^T (upper triangle)
    blas::syr2k<T>('U', 'N', N, K, T{1.0}, A.data(), lda, B.data(), ldb, T{0.0}, C_syr2k.data(), ldc);

    // Verify upper triangle
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            T ab_t = T{0.0};
            T ba_t = T{0.0};
            for (int k = 0; k < K; k++) {
                ab_t += A(i, k) * B(j, k);
                ba_t += B(i, k) * A(j, k);
            }
            T expected = ab_t + ba_t;
            REQUIRE(C_syr2k(i, j) == Catch::Approx(expected).margin(1e-6));
        }
    }
}
