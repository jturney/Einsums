//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>

#include <cmath>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("syevd", "[linear-algebra]", float, double) {
    using T = TestType;

    constexpr int N = 4;

    // Symmetric matrix
    auto A  = create_tensor<T>("A", N, N);
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

    // Copy for syev comparison
    auto A_copy = create_tensor<T>("A_copy", N, N);
    A_copy      = A;

    auto w_syevd = create_tensor<T>("w_syevd", N);
    auto w_syev  = create_tensor<T>("w_syev", N);

    blas::int_t lda      = A.impl().get_lda();
    blas::int_t lda_copy = A_copy.impl().get_lda();

    // Row-major fix: flip uplo 'U' -> 'L'
    auto info = blas::syevd<T>('V', 'L', N, A.data(), lda, w_syevd.data());
    REQUIRE(info == 0);

    // Call syev on the copy for comparison
    // Row-major fix: flip uplo 'U' -> 'L'
    T work_query;
    blas::syev<T>('V', 'L', N, A_copy.data(), lda_copy, w_syev.data(), &work_query, -1);
    blas::int_t    lwork = static_cast<blas::int_t>(work_query);
    std::vector<T> work(lwork);
    info = blas::syev<T>('V', 'L', N, A_copy.data(), lda_copy, w_syev.data(), work.data(), lwork);
    REQUIRE(info == 0);

    // Compare eigenvalues from syevd and syev
    for (int i = 0; i < N; i++) {
        REQUIRE(w_syevd(i) == Catch::Approx(w_syev(i)).margin(1e-6));
    }
}
