//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("sygv", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Column-major tensors
    auto A  = create_tensor<T>(false, "A", N, N);
    A(0, 0) = 2.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(0, 3) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = 3.0;
    A(1, 2) = 1.0;
    A(1, 3) = 0.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = 4.0;
    A(2, 3) = 1.0;
    A(3, 0) = 0.0;
    A(3, 1) = 0.0;
    A(3, 2) = 1.0;
    A(3, 3) = 5.0;

    // SPD matrix B = I + v*v^T
    auto B = create_tensor<T>(false, "B", N, N);
    B.zero();
    for (int i = 0; i < N; i++)
        B(i, i) = T{1.0};
    T v[] = {0.5, 0.3, 0.2, 0.1};
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B(i, j) += v[i] * v[j];

    auto A_orig = create_tensor<T>(false, "A_orig", N, N);
    auto B_orig = create_tensor<T>(false, "B_orig", N, N);
    A_orig      = A;
    B_orig      = B;

    auto w = create_tensor<T>("w", N);

    blas::int_t lda = A.impl().get_lda();
    blas::int_t ldb = B.impl().get_lda();

    auto info = blas::sygv<T>(1, 'V', 'U', N, A.data(), lda, B.data(), ldb, w.data());
    REQUIRE(info == 0);

    // Eigenvectors stored as columns of A (column-major). Column j = eigenvector j.
    for (int eigvec = 0; eigvec < N; eigvec++) {
        for (int i = 0; i < N; i++) {
            T Ax = T{0.0};
            T Bx = T{0.0};
            for (int k = 0; k < N; k++) {
                Ax += A_orig(i, k) * A(k, eigvec);
                Bx += B_orig(i, k) * A(k, eigvec);
            }
            REQUIRE(Ax == Catch::Approx(w(eigvec) * Bx).margin(1e-5));
        }
    }
}
