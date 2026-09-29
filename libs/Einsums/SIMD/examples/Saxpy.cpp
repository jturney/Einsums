//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Saxpy.cpp
/// @brief y = a * x + y over an array of any length: the shape of almost every SIMD loop.
///
/// The body processes a whole Vec per iteration. The tail, the last n % lanes elements, is loaded
/// and stored with loadu_partial and storeu_partial, which touch only those elements, so the loop
/// never reads or writes past the end of either array. The alternative, a scalar loop over the
/// tail, works too; the partial form keeps one code path and one rounding behaviour for every
/// element.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

/// y[i] = a * x[i] + y[i] for i in [0, n).
void saxpy(std::size_t n, float a, float const *x, float *y) {
    constexpr std::size_t L  = lanes<float>;
    Vec<float> const      va = broadcast(a);

    std::size_t i = 0;
    for (; i + L <= n; i += L) {
        storeu(y + i, fmadd(va, loadu(x + i), loadu(y + i)));
    }
    if (i < n) {
        std::size_t const tail = n - i;
        storeu_partial(y + i, fmadd(va, loadu_partial(x + i, tail), loadu_partial(y + i, tail)), tail);
    }
}

} // namespace

int einsums_main() {
    int failures = 0;
    // Lengths below, at and just past multiples of every lane count.
    for (std::size_t n : {0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 33, 1000, 1023}) {
        std::vector<float> x(n), y(n), expect(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i]      = static_cast<float>(i) * 0.25f;
            y[i]      = 1.0f - static_cast<float>(i);
            expect[i] = std::fma(2.5f, x[i], y[i]);
        }
        saxpy(n, 2.5f, x.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) {
            // The vector FMA rounds once, like std::fma; a backend without FMA rounds twice.
            if (std::abs(y[i] - expect[i]) > 1e-5f * (1.0f + std::abs(expect[i]))) {
                std::cout << "n = " << n << ": y[" << i << "] = " << y[i] << ", expected " << expect[i] << "\n";
                ++failures;
            }
        }
    }
    std::cout << "saxpy over " << lanes<float> << "-lane Vecs with a partial tail: " << (failures ? "FAILED" : "ok") << "\n";
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
