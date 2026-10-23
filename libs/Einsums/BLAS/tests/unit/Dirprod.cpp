//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

template <typename T>
std::vector<T> random_values(std::size_t n, unsigned seed) {
    std::mt19937                     gen(seed);
    std::normal_distribution<double> dist;
    std::vector<T>                   x(n);
    for (auto &value : x) {
        if constexpr (einsums::IsComplexV<T>) {
            value = T(static_cast<einsums::RemoveComplexT<T>>(dist(gen)), static_cast<einsums::RemoveComplexT<T>>(dist(gen)));
        } else {
            value = static_cast<T>(dist(gen));
        }
    }
    return x;
}

template <typename T>
T nan_value() {
    auto const nan = std::numeric_limits<einsums::RemoveComplexT<T>>::quiet_NaN();
    if constexpr (einsums::IsComplexV<T>) {
        return T(nan, nan);
    } else {
        return nan;
    }
}

/// alpha x y + beta z, evaluated in double precision on the elements dirprod reads, with
/// increments walked as BLAS walks them: a negative one starts from the far end.
template <typename T>
std::vector<T> expected(int n, T alpha, std::vector<T> const &x, int incx, std::vector<T> const &y, int incy, T beta,
                        std::vector<T> const &z, int incz) {
    using Wide         = std::conditional_t<einsums::IsComplexV<T>, std::complex<double>, double>;
    auto const     at  = [n](int i, int inc) { return static_cast<std::size_t>(inc < 0 ? (n - 1 - i) * -inc : i * inc); };
    std::vector<T> out = z;
    for (int i = 0; i < n; i++) {
        Wide const product = Wide(alpha) * Wide(x[at(i, incx)]) * Wide(y[at(i, incy)]);
        out[at(i, incz)]   = static_cast<T>(beta == T{0} ? product : Wide(beta) * Wide(z[at(i, incz)]) + product);
    }
    return out;
}

template <typename T>
void check_close(std::vector<T> const &got, std::vector<T> const &want) {
    double const eps = std::numeric_limits<einsums::RemoveComplexT<T>>::epsilon();
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < got.size(); i++) {
        INFO("element " << i);
        CHECK(std::abs(got[i] - want[i]) <= 16 * eps * (1.0 + std::abs(want[i])));
    }
}

} // namespace

// dirprod computes z = alpha x y + beta z in one pass. Scaling z first and accumulating the product
// second read and wrote all of z twice, and the AVX2 kernels meant to speed it up were gated on
// __FMA3__, a macro no compiler defines, so they never ran.
TEMPLATE_TEST_CASE("dirprod scales z in the pass that adds the product", "[BLAS][dirprod]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T = TestType;

    auto const scalar = [](double re, double im) {
        if constexpr (einsums::IsComplexV<T>) {
            return T(static_cast<einsums::RemoveComplexT<T>>(re), static_cast<einsums::RemoveComplexT<T>>(im));
        } else {
            return static_cast<T>(re);
        }
    };
    T const alpha = scalar(1.5, -0.5);
    T const beta  = scalar(0.5, 0.25);

    // 200 is three 64-element blocks and a remainder; 40000 is above the parallel threshold.
    for (int const n : {1, 63, 200, 40000}) {
        INFO("n = " << n);
        SECTION("unit stride, n = " + std::to_string(n)) {
            auto const     x = random_values<T>(n, 1), y = random_values<T>(n, 2), z0 = random_values<T>(n, 3);
            std::vector<T> z = z0;
            einsums::blas::dirprod<T>(n, alpha, x.data(), 1, y.data(), 1, beta, z.data(), 1);
            check_close(z, expected<T>(n, alpha, x, 1, y, 1, beta, z0, 1));
        }
        SECTION("strided, n = " + std::to_string(n)) {
            auto const     x = random_values<T>(2 * n, 1), y = random_values<T>(3 * n, 2), z0 = random_values<T>(2 * n, 3);
            std::vector<T> z = z0;
            einsums::blas::dirprod<T>(n, alpha, x.data(), 2, y.data(), 3, beta, z.data(), 2);
            check_close(z, expected<T>(n, alpha, x, 2, y, 3, beta, z0, 2));
        }
    }
}

// The strided path indexed z[i * incz] from the base pointer, so a negative increment read and wrote
// before the start of the vector, where BLAS walks it backwards from the far end.
TEMPLATE_TEST_CASE("dirprod walks a negative increment backwards from the far end", "[BLAS][dirprod]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T     = TestType;
    int const n = 37;

    auto const     x = random_values<T>(n, 4), y = random_values<T>(2 * n, 5), z0 = random_values<T>(n, 6);
    std::vector<T> z = z0;
    einsums::blas::dirprod<T>(n, T(2), x.data(), -1, y.data(), -2, T(1), z.data(), 1);
    check_close(z, expected<T>(n, T(2), x, -1, y, -2, T(1), z0, 1));
}

// A zero beta is BLAS's "do not read the output": an output carved from reused memory may hold NaN,
// and a scale-then-add multiplied it by zero and kept it.
TEMPLATE_TEST_CASE("dirprod with a zero beta never reads z", "[BLAS][dirprod]", float, double, std::complex<float>, std::complex<double>) {
    using T = TestType;

    for (int const n : {200, 40000}) {
        for (int const inc : {1, 2}) {
            INFO("n = " << n << ", increment " << inc);
            auto const     x = random_values<T>(inc * n, 7), y = random_values<T>(inc * n, 8);
            std::vector<T> z(inc * n, nan_value<T>());
            einsums::blas::dirprod<T>(n, T(1), x.data(), inc, y.data(), inc, T(0), z.data(), inc);
            for (int i = 0; i < n; i++) {
                INFO("element " << i);
                CHECK(std::abs(z[i * inc] - x[i * inc] * y[i * inc]) <=
                      16 * std::numeric_limits<einsums::RemoveComplexT<T>>::epsilon() * (1.0 + std::abs(x[i * inc] * y[i * inc])));
            }
        }
    }
}
