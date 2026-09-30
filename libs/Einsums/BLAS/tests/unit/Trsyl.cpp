//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <Einsums/Testing.hpp>

// The complex trsyl wrappers used to branch on EINSUMS_HAVE_MKL_LAPACKE_H and
// EINSUMS_HAVE_LAPACKE, which no build defines, so every complex call threw "not
// implemented" although the vendor layer implements both.
TEMPLATE_TEST_CASE("trsyl solves a triangular Sylvester equation", "[blas][trsyl]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T    = TestType;
    using Real = einsums::RemoveComplexT<T>;

    constexpr std::size_t m = 4;
    constexpr std::size_t n = 3;

    // Column-major upper-triangular A (m x m) and B (n x n): the Schur form trsyl takes. Positive
    // diagonals keep every eigenvalue sum of A and B away from zero, so A X + X B = C is well posed.
    auto entry = [](std::size_t i, std::size_t j, Real shift) {
        Real const row = Real{0.25} * static_cast<Real>(i + 1);
        Real const col = Real{0.125} * static_cast<Real>(j + 1);
        if constexpr (einsums::IsComplexV<T>) {
            return T{row + shift, col - shift};
        } else {
            return T{row + col + shift};
        }
    };
    std::vector<T> A(m * m, T{0}), B(n * n, T{0}), C(m * n);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i <= j; ++i) {
            A[i + j * m] = i == j ? T{static_cast<Real>(2 + i)} : entry(i, j, Real{0});
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i <= j; ++i) {
            B[i + j * n] = i == j ? T{static_cast<Real>(1 + j)} : entry(i, j, Real{0.5});
        }
    }
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            C[i + j * m] = entry(i, j, Real{-0.25});
        }
    }

    std::vector<T> X     = C;
    Real           scale = 0;
    auto const     info  = einsums::blas::trsyl<T>('N', 'N', 1, m, n, A.data(), m, B.data(), n, X.data(), m, &scale);
    REQUIRE(info == 0);
    REQUIRE(scale > Real{0});

    // Residual of A X + X B = scale C, by plain loops.
    Real const tol = std::is_same_v<Real, float> ? Real{1e-4} : Real{1e-12};
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            T lhs{0};
            for (std::size_t k = 0; k < m; ++k) {
                lhs += A[i + k * m] * X[k + j * m];
            }
            for (std::size_t k = 0; k < n; ++k) {
                lhs += X[i + k * m] * B[k + j * n];
            }
            CAPTURE(i, j);
            REQUIRE(std::abs(lhs - scale * C[i + j * m]) <= tol * (Real{1} + std::abs(scale * C[i + j * m])));
        }
    }
}
