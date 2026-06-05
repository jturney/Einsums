//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>

#include <tuple>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("gesv", "[linear-algebra]", float, double) {
    /*
       LAPACKE_dgesv Example.
       ======================

       The program computes the solution to the system of linear
       equations with a square matrix A and multiple
       right-hand sides B, where A is the coefficient matrix:

         6.80  -6.05  -0.45   8.32  -9.67
        -2.11  -3.30   2.58   2.71  -5.14
         5.66   5.36  -2.70   4.35  -7.26
         5.97  -4.44   0.27  -7.17   6.08
         8.23   1.08   9.04   2.14  -6.87

       and B is the right-hand side matrix:

         4.02  -1.56   9.81
         6.19   4.00  -4.09
        -8.22  -8.67  -4.57
        -7.57   1.75  -8.61
        -3.03   2.86   8.99

       Description.
       ============

       The routine solves for X the system of linear equations A*X = B,
       where A is an n-by-n matrix, the columns of matrix B are individual
       right-hand sides, and the columns of X are the corresponding
       solutions.

       The LU decomposition with partial pivoting and row interchanges is
       used to factor A as A = P*L*U, where P is a permutation matrix, L
       is unit lower triangular, and U is upper triangular. The factored
       form of A is then used to solve the system of equations A*X = B.

       Example Program Results.
       ========================

     LAPACKE_dgesv (column-major, high-level) Example Program Results

     Solution
      -0.80  -0.39   0.96
      -0.70  -0.55   0.22
       0.59   0.84   1.90
       1.32  -0.10   5.36
       0.57   0.11   4.04

     Details of LU factorization
       8.23   1.08   9.04   2.14  -6.87
       0.83  -6.94  -7.92   6.55  -3.99
       0.69  -0.67 -14.18   7.24  -5.19
       0.73   0.75   0.02 -13.82  14.19
      -0.26   0.44  -0.59  -0.34  -3.43

     Pivot indices
          5      5      3      4      5
        */

    constexpr int N{5};
    constexpr int NRHS{3};
    constexpr int LDA{N};
    constexpr int LDB{N};

    using namespace einsums;

    using T = TestType;

    auto a = create_tensor<T>(false, "a", N, LDA);
    auto b = create_tensor<T>(false, "b", N, NRHS);

    a.vector_data() = {6.80, -2.11, 5.66, 5.97, 8.23, -6.05, -3.30, 5.36,  -4.44, 1.08,  -0.45, 2.58, -2.70,
                       0.27, 9.04,  8.32, 2.71, 4.35, -7.17, 2.14,  -9.67, -5.14, -7.26, 6.08,  -6.87};
    b.vector_data() = {4.02, 6.19, -8.22, -7.57, -3.03, -1.56, 4.00, -8.67, 1.75, 2.86, 9.81, -4.09, -4.57, -8.61, 8.99};

    auto a_swap = create_tensor<T>(true, "a_swap", N, LDA);
    auto b_swap = create_tensor<T>(true, "b_swap", N, NRHS);

    a_swap = a;
    b_swap = b;

    std::ignore = linear_algebra::gesv(&a, &b);

    CHECK_THAT(a.vector_data(), Catch::Matchers::Approx(std::vector<T>{8.23000000,  0.82624544,  0.68772783,   0.72539490,   -0.25637910,
                                                                       1.08000000,  -6.94234508, -0.66508563,  0.75240087,   0.43545957,
                                                                       9.04000000,  -7.91925881, -14.18404477, 0.02320302,   -0.58842060,
                                                                       2.14000000,  6.55183475,  7.23579360,   -13.81984350, -0.33743379,
                                                                       -6.87000000, -3.99369380, -5.19145820,  14.18877913,  -3.42921969})
                                    .margin(0.00001));
    CHECK_THAT(b.vector_data(), Catch::Matchers::Approx(std::vector<T>{-0.80071403, -0.69524338, 0.59391499, 1.32172561, 0.56575620,
                                                                       -0.38962139, -0.55442713, 0.84222739, -0.10380185, 0.10571095,
                                                                       0.95546491, 0.22065963, 1.90063673, 5.35766149, 4.04060266})
                                    .margin(0.00001));

    std::ignore = linear_algebra::gesv(&a_swap, &b_swap);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            CHECK_THAT(a_swap(i, j), Catch::Matchers::WithinRel(a(i, j)));
        }
        for (int j = 0; j < NRHS; j++) {
            CHECK_THAT(b_swap(i, j), Catch::Matchers::WithinRel(b(i, j)));
        }
    }
}

// TEMPLATE_TEST_CASE("disk gesv", "[linear-algebra]", float, double) {
//     using namespace einsums;

//     constexpr int size = 20;

//     auto A = create_random_tensor<TestType>("coefficients", size, size);
//     auto B = create_random_tensor<TestType>("results", size);

//     DiskTensor<TestType, 2> A_disk(fmt::format("/test/gesv/{}/A", type_name<TestType>()), size, size);
//     auto                    B_copy = B;

//     A_disk.write(A);

//     SECTION("Solve") {
//         linear_algebra::gesv(&A_disk, &B_copy);

//         linear_algebra::gesv(&A, &B);

//         for (int i = 0; i < size; i++) {
//             REQUIRE_THAT(B_copy(i), Catch::Matchers::WithinRel(B(i)));
//         }
//     }
// }
