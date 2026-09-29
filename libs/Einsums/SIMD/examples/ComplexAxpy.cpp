//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ComplexAxpy.cpp
/// @brief y = a * conj(x) + y for complex arrays, with CVec.
///
/// A CVec<T> holds interleaved (re, im) pairs in one Vec<T>, the same layout std::complex has in
/// memory, so complex_loadu and complex_storeu are plain loads and stores. complex_mul and
/// conjugate do the shuffling a complex product needs. A CVec holds Vec<T>::lanes / 2 complex
/// numbers, so the loop steps by complex_lanes; the scalar loop finishes whatever is left.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

void conj_axpy(std::size_t n, std::complex<double> a, std::complex<double> const *x, std::complex<double> *y) {
    constexpr std::size_t CL = CVec<double>::complex_lanes;
    CVec<double> const    va = complex_broadcast(a);

    std::size_t i = 0;
    for (; i + CL <= n; i += CL) {
        CVec<double> const prod = complex_mul(va, conjugate(complex_loadu(x + i)));
        complex_storeu(y + i, complex_add(prod, complex_loadu(y + i)));
    }
    for (; i < n; ++i) {
        y[i] += a * std::conj(x[i]);
    }
}

} // namespace

int einsums_main() {
    std::size_t const                 n = 37;
    std::complex<double> const        a{0.5, -2.0};
    std::vector<std::complex<double>> x(n), y(n), expect(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i]      = {std::cos(0.1 * static_cast<double>(i)), std::sin(0.3 * static_cast<double>(i))};
        y[i]      = {static_cast<double>(i), -1.0};
        expect[i] = y[i] + a * std::conj(x[i]);
    }
    conj_axpy(n, a, x.data(), y.data());

    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(y[i] - expect[i]));
    }
    std::cout << CVec<double>::complex_lanes << " complex<double> per CVec; largest error " << worst << "\n";
    return worst < 1e-13 ? 0 : 1;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
