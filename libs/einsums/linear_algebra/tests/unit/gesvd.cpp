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
void gesvd_test() {
    /*
       DGESVD Example.
       ==============

       Program computes the singular value decomposition of a general
       rectangular matrix A:

         8.79   9.93   9.83   5.45   3.16
         6.11   6.91   5.04  -0.27   7.98
        -9.15  -7.93   4.86   4.85   3.01
         9.57   1.64   8.83   0.74   5.80
        -3.49   4.02   9.80  10.00   4.27
         9.84   0.15  -8.99  -6.02  -5.31

       Description.
       ============

       The routine computes the singular value decomposition (SVD) of a real
       m-by-n matrix A, optionally computing the left and/or right singular
       vectors. The SVD is written as

       A = U*SIGMA*VT

       where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
       diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
       is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
       are the singular values of A; they are real and non-negative, and ar
       returned in descending order. The first min(m, n) columns of U and V are
       the left and right singular vectors of A.

       Note that the routine returns VT, not V.

               Example Program Results.
               ========================

               DGESVD Example Program Results

                       Singular values
                   27.47  22.64   8.56   5.99   2.01

                   Left singular vectors (stored columnwise)
                   -0.59   0.26   0.36   0.31   0.23
                   -0.40   0.24  -0.22  -0.75  -0.36
                   -0.03  -0.60  -0.45   0.23  -0.31
                   -0.43   0.24  -0.69   0.33   0.16
                   -0.47  -0.35   0.39   0.16  -0.52
                   0.29   0.58  -0.02   0.38  -0.65

                   Right singular vectors (stored rowwise)
                   -0.25  -0.40  -0.69  -0.37  -0.41
                   0.81   0.36  -0.25  -0.37  -0.10
                   -0.26   0.70  -0.22   0.39  -0.49
                   0.40  -0.45   0.25   0.43  -0.62
                   -0.22   0.14   0.59  -0.63  -0.44
                                                 */
    constexpr int M{6};
    constexpr int N{5};
    constexpr int LDA{M};
    constexpr int LDU{M};
    constexpr int LDVT{N};

    using namespace einsums;

    auto a = create_tensor<T>("a", N, LDA);

    a.vector_data() = vector_type<T>{8.79, 6.11, -9.15, 9.57, -3.49, 9.84, 9.93, 6.91,  -7.93, 1.64, 4.02, 0.15, 9.83, 5.04, 4.86,
                                     8.83, 9.80, -8.99, 5.45, -0.27, 4.85, 0.74, 10.00, -6.02, 3.16, 7.98, 3.01, 5.80, 4.27, -5.31};

    auto [u, s, vt] = linear_algebra::svd(a);

    // Using u, s, and vt reconstruct a and test a against the reconstructed a
    auto new_a = reconstruct(u, s, vt, N, LDA);

    CHECK_THAT(new_a.vector_data(), Catch::Matchers::Approx(a.vector_data()).margin(0.0001));
}

TEST_CASE("gesvd") {
    SECTION("float") {
        gesvd_test<float>();
    }
    SECTION("double") {
        gesvd_test<double>();
    }
}
