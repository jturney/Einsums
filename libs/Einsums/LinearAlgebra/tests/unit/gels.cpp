//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("gels", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int M    = 6;
    constexpr int N    = 4;
    constexpr int NRHS = 1;

    // Use column-major tensors for direct BLAS calls with non-square matrices.
    // BLAS expects column-major data and the leading dimensions must satisfy lda >= M.
    auto A  = create_tensor<T>(false, "A", M, N);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 0.0;
    A(0, 3) = 1.0;
    A(1, 0) = 0.0;
    A(1, 1) = 1.0;
    A(1, 2) = 3.0;
    A(1, 3) = 0.0;
    A(2, 0) = 2.0;
    A(2, 1) = 0.0;
    A(2, 2) = 1.0;
    A(2, 3) = 1.0;
    A(3, 0) = 1.0;
    A(3, 1) = 1.0;
    A(3, 2) = 1.0;
    A(3, 3) = 1.0;
    A(4, 0) = 0.0;
    A(4, 1) = 3.0;
    A(4, 2) = 2.0;
    A(4, 3) = 0.0;
    A(5, 0) = 1.0;
    A(5, 1) = 0.0;
    A(5, 2) = 0.0;
    A(5, 3) = 2.0;

    // Known solution x_true
    T x_true[] = {1.0, 2.0, 3.0, 4.0};

    // b = A * x_true (consistent system) computed element-wise
    auto b = create_tensor<T>(false, "b", M, NRHS);
    b.zero();
    for (int i = 0; i < M; i++) {
        T sum = T{0.0};
        for (int k = 0; k < N; k++) {
            sum += A(i, k) * x_true[k];
        }
        b(i, 0) = sum;
    }

    blas::int_t lda = M; // column-major: lda = number of rows
    blas::int_t ldb = M; // column-major: ldb = number of rows

    // Call gels to solve the least-squares problem (column-major, no row-major fixups needed)
    auto info = blas::gels<T>('N', M, N, NRHS, A.data(), lda, b.data(), ldb);
    REQUIRE(info == 0);

    // First N elements of b should be the solution = x_true
    for (int i = 0; i < N; i++) {
        REQUIRE(b(i, 0) == Catch::Approx(x_true[i]).margin(1e-5));
    }
}
