//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("syrk", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;
    constexpr int K = 3;

    // Column-major tensors for direct BLAS compatibility
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

    auto C_syrk = create_tensor<T>(false, "C_syrk", N, N);
    C_syrk.zero();

    blas::int_t lda = A.impl().get_lda();
    blas::int_t ldc = C_syrk.impl().get_lda();

    // C = A * A^T (upper triangle)
    blas::syrk<T>('U', 'N', N, K, T{1.0}, A.data(), lda, T{0.0}, C_syrk.data(), ldc);

    // Verify upper triangle: C(i,j) = sum_k A(i,k)*A(j,k)
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            T expected = T{0.0};
            for (int k = 0; k < K; k++) {
                expected += A(i, k) * A(j, k);
            }
            REQUIRE(C_syrk(i, j) == Catch::Approx(expected).margin(1e-6));
        }
    }
}
