//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("trtri", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Column-major upper triangular A
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

    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    A_orig      = A;

    blas::int_t lda = A.impl().get_lda();

    auto info = blas::trtri<T>('U', 'N', N, A.data(), lda);
    REQUIRE(info == 0);

    // Verify A_orig * A_inv = I
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T sum = T{0.0};
            for (int k = 0; k < N; k++) {
                sum += A_orig(i, k) * A(k, j);
            }
            T expected = (i == j) ? T{1.0} : T{0.0};
            REQUIRE(sum == Catch::Approx(expected).margin(1e-6));
        }
    }
}
