//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("trsv", "[linear-algebra]", float, double) {
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

    auto b = create_tensor<T>("b", N);
    b(0)   = 10.0;
    b(1)   = 12.0;
    b(2)   = 8.0;
    b(3)   = 15.0;

    auto b_orig = create_tensor<T>("b_orig", N);
    b_orig      = b;

    blas::int_t lda = A.impl().get_lda();

    // Solve A*x = b in-place
    blas::trsv<T>('U', 'N', 'N', N, A.data(), lda, b.data(), 1);

    // Verify: A * x ≈ b_orig
    for (int i = 0; i < N; i++) {
        T sum = T{0.0};
        for (int k = 0; k < N; k++) {
            sum += A(i, k) * b(k);
        }
        REQUIRE(sum == Catch::Approx(b_orig(i)).margin(1e-6));
    }
}
