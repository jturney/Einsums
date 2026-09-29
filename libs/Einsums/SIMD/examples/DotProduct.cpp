//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file DotProduct.cpp
/// @brief A dot product: accumulate in Vec lanes, reduce once at the end.
///
/// Two things make it fast. The loop keeps several independent accumulators, because each FMA
/// has to wait for the previous one on the same register; four chains keep the FMA units busy
/// where one chain would stall on latency. And the horizontal sum, reduce_add, runs once after the
/// loop rather than once per iteration.
///
/// The price is a different summation order from a sequential loop, so the result differs in the
/// last bits. The check below allows for that.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

double dot(std::size_t n, double const *x, double const *y) {
    constexpr std::size_t L = lanes<double>;

    Vec<double> acc0 = broadcast(0.0), acc1 = broadcast(0.0), acc2 = broadcast(0.0), acc3 = broadcast(0.0);
    std::size_t i = 0;
    for (; i + 4 * L <= n; i += 4 * L) {
        acc0 = fmadd(loadu(x + i), loadu(y + i), acc0);
        acc1 = fmadd(loadu(x + i + L), loadu(y + i + L), acc1);
        acc2 = fmadd(loadu(x + i + 2 * L), loadu(y + i + 2 * L), acc2);
        acc3 = fmadd(loadu(x + i + 3 * L), loadu(y + i + 3 * L), acc3);
    }
    for (; i + L <= n; i += L) {
        acc0 = fmadd(loadu(x + i), loadu(y + i), acc0);
    }
    if (i < n) {
        // Partial loads zero the unused lanes, which then add nothing to the sum.
        acc0 = fmadd(loadu_partial(x + i, n - i), loadu_partial(y + i, n - i), acc0);
    }
    return reduce_add((acc0 + acc1) + (acc2 + acc3));
}

} // namespace

int einsums_main() {
    int failures = 0;
    for (std::size_t n : {0, 1, 5, 16, 63, 64, 65, 1000, 100003}) {
        std::vector<double> x(n), y(n);
        double              expect = 0.0, magnitude = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::sin(static_cast<double>(i));
            y[i] = std::cos(static_cast<double>(i) * 0.5);
            expect += x[i] * y[i];
            magnitude += std::abs(x[i] * y[i]);
        }
        double const got = dot(n, x.data(), y.data());
        // A reordered sum of n terms stays within about n ulps of the terms' magnitude.
        double const bound = 4.0 * static_cast<double>(n + 1) * 1.1e-16 * magnitude;
        std::cout << "n = " << n << ": dot = " << got << " (sequential " << expect << ")\n";
        if (std::abs(got - expect) > bound) {
            ++failures;
        }
    }
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
