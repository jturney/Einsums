//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/modules/tensor_utilities.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums;

template <typename T>
auto reconstruct(const tensor<T, 2> &u, const tensor<T, 1> &s, const tensor<T, 2> &vt, size_t N, size_t LDA) -> tensor<T, 2> {

    auto half = create_tensor<T>("sigma", N, LDA);
    zero(half);
    auto new_a = create_tensor<T>("new a", N, LDA);
    zero(new_a);
    auto sigma = diagonal_like(s, new_a);

    linear_algebra::gemm<false, false>(1.0, u, sigma, 0.0, &half);
    linear_algebra::gemm<false, false>(1.0, half, vt, 0.0, &new_a);

    return new_a;
}

template <typename T>
void gesdd_test() {
    /*
   DGESDD Example.
   ==============

   Program computes the singular value decomposition of a general
   rectangular matrix A using a divide and conquer method, where A is:

     7.52  -1.10  -7.95   1.08
    -0.76   0.62   9.34  -7.10
     5.13   6.62  -5.66   0.87
    -4.75   8.52   5.75   5.30
     1.33   4.91  -5.49  -3.52
    -2.40  -6.77   2.34   3.95

   Description.
   ============

   The routine computes the singular value decomposition (SVD) of a real
   m-by-n matrix A, optionally computing the left and/or right singular
   vectors. If singular vectors are desired, it uses a divide and conquer
   algorithm. The SVD is written as

   A = U*SIGMA*VT

   where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
   diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
   is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
   are the singular values of A; they are real and non-negative, and are
   returned in descending order. The first min(m, n) columns of U and V are
   the left and right singular vectors of A.

   Note that the routine returns VT, not V.

           Example Program Results.
           ========================

           DGESDD Example Program Results

                   Singular values
               18.37  13.63  10.85   4.49

               Left singular vectors (stored columnwise)
               -0.57   0.18   0.01   0.53
               0.46  -0.11  -0.72   0.42
               -0.45  -0.41   0.00   0.36
               0.33  -0.69   0.49   0.19
               -0.32  -0.31  -0.28  -0.61
               0.21   0.46   0.39   0.09

               Right singular vectors (stored rowwise)
               -0.52  -0.12   0.85  -0.03
               0.08  -0.99  -0.09  -0.01
               -0.28  -0.02  -0.14   0.95
                                   0.81   0.01   0.50   0.31
                                   */
    constexpr int M{6};
    constexpr int N{4};
    constexpr int LDA{M};
    constexpr int LDU{M};
    constexpr int LDVT{N};

    using namespace einsums;

    auto a = create_tensor<T>("a", N, LDA);

    a.vector_data() = vector_type<T>{7.52,  -0.76, 5.13,  -4.75, 1.33,  -2.40, -1.10, 0.62,  6.62, 8.52, 4.91,  -6.77,
                                     -7.95, 9.34,  -5.66, 5.75,  -5.49, 2.34,  1.08,  -7.10, 0.87, 5.30, -3.52, 3.95};

    auto [u, s, vt] = linear_algebra::svd_dd(a);

    // Using u, s, and vt reconstruct a and test a against the reconstructed a
    auto new_a = reconstruct(u, s, vt, N, LDA);

    CHECK_THAT(new_a.vector_data(), Catch::Matchers::Approx(a.vector_data()).margin(0.0001));
}

TEST_CASE("gesdd") {
    SECTION("double") {
        gesdd_test<double>();
    }
    SECTION("float") {
        gesdd_test<float>();
    }
}
