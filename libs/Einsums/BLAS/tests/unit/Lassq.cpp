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
#include <vector>

#include <Einsums/Testing.hpp>

namespace {

template <typename T>
std::vector<T> random_values(std::size_t n) {
    std::mt19937                     gen(29);
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

/// The sum of |x|^2 in long double, which no vendor routine is involved in.
template <typename T>
long double exact_sum_of_squares(std::vector<T> const &x) {
    long double sum = 0;
    for (auto const &value : x) {
        long double const magnitude = std::abs(value);
        sum += magnitude * magnitude;
    }
    return sum;
}

} // namespace

// Apple Accelerate's zlassq returned scale^2 * sumsq about 2.5e-9 relative off for complex<double>,
// some ten million times its epsilon; its dznrm2 was exact. zlange('F') is built on zlassq, so the
// Frobenius norm and sum_square inherited it. The wrappers now combine an nrm2 of the input with
// what the caller passed in, so the result no longer depends on the vendor's lassq.
TEMPLATE_TEST_CASE("lassq accumulates the sum of squares to the type's precision", "[blas][lassq]", float, double, std::complex<float>,
                   std::complex<double>) {
    using T    = TestType;
    using Real = einsums::RemoveComplexT<T>;

    constexpr std::size_t n     = 35;
    auto const            x     = random_values<T>(n);
    long double const     exact = exact_sum_of_squares(x);
    double const          tol   = 16.0 * std::numeric_limits<Real>::epsilon();

    auto const total = [](Real scale, Real sumsq) { return static_cast<long double>(scale) * scale * sumsq; };

    SECTION("starting from LAPACK's scale = 0, sumsq = 1") {
        Real scale = 0, sumsq = 1;
        einsums::blas::lassq<T>(n, x.data(), 1, &scale, &sumsq);
        CHECK(std::abs(total(scale, sumsq) - exact) <= tol * exact);
    }

    SECTION("starting from scale = 1, sumsq = 0") {
        Real scale = 1, sumsq = 0;
        einsums::blas::lassq<T>(n, x.data(), 1, &scale, &sumsq);
        CHECK(std::abs(total(scale, sumsq) - exact) <= tol * exact);
    }

    SECTION("accumulating onto a prior total, with a stride") {
        Real scale = 2, sumsq = 3; // a prior total of 12
        einsums::blas::lassq<T>(n / 2, x.data(), 2, &scale, &sumsq);
        long double strided = 12;
        for (std::size_t i = 0; i < n / 2; ++i) {
            long double const magnitude = std::abs(x[2 * i]);
            strided += magnitude * magnitude;
        }
        CHECK(std::abs(total(scale, sumsq) - strided) <= tol * strided);
    }

    SECTION("an empty input leaves the total unchanged") {
        Real scale = 2, sumsq = 3;
        einsums::blas::lassq<T>(0, x.data(), 1, &scale, &sumsq);
        CHECK(total(scale, sumsq) == 12);
    }
}
