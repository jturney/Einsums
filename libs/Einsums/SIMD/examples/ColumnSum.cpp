//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ColumnSum.cpp
/// @brief Column sums of a row-major matrix, two ways: strided gathers, or contiguous rows.
///
/// A column of a row-major matrix is strided: consecutive elements are a row length apart.
/// gather(base, stride) loads lanes elements that far apart into one Vec, so a column's elements
/// can be summed lanes at a time. It works, but each gathered lane touches a different cache line.
///
/// The better answer is usually to change the traversal rather than gather: summing whole rows
/// into a Vec of column accumulators reads memory contiguously. Both are here, and both are
/// checked against a scalar loop.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

/// Sum of column @p c: gather lanes rows at a time, then add the rows left over.
double column_sum_gather(double const *m, std::size_t rows, std::size_t cols, std::size_t c) {
    constexpr std::size_t L   = lanes<double>;
    Vec<double>           acc = broadcast(0.0);
    std::size_t           r   = 0;
    for (; r + L <= rows; r += L) {
        acc = acc + gather(m + r * cols + c, static_cast<std::ptrdiff_t>(cols));
    }
    double sum = reduce_add(acc);
    for (; r < rows; ++r) {
        sum += m[r * cols + c];
    }
    return sum;
}

/// Every column sum at once: each row is added into the accumulators contiguously.
void column_sums_by_row(double const *m, std::size_t rows, std::size_t cols, double *sums) {
    constexpr std::size_t L = lanes<double>;
    for (std::size_t c0 = 0; c0 < cols; c0 += L) {
        std::size_t const width = std::min(L, cols - c0);
        Vec<double>       acc   = broadcast(0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            acc = acc + loadu_partial(m + r * cols + c0, width);
        }
        storeu_partial(sums + c0, acc, width);
    }
}

} // namespace

int einsums_main() {
    std::size_t const   rows = 101, cols = 13;
    std::vector<double> m(rows * cols);
    for (std::size_t i = 0; i < m.size(); ++i) {
        m[i] = static_cast<double>(i % 17) - 8.0;
    }

    std::vector<double> by_row(cols);
    column_sums_by_row(m.data(), rows, cols, by_row.data());

    int failures = 0;
    for (std::size_t c = 0; c < cols; ++c) {
        double expect = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            expect += m[r * cols + c];
        }
        double const gathered = column_sum_gather(m.data(), rows, cols, c);
        std::cout << "column " << c << ": " << expect << " (gather " << gathered << ", by row " << by_row[c] << ")\n";
        // Small integers: every partial sum is exact, so the three must agree exactly.
        failures += (gathered != expect) + (by_row[c] != expect);
    }
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
