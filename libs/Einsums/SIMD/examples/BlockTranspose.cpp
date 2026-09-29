//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file BlockTranspose.cpp
/// @brief Transpose a matrix a lanes x lanes tile at a time, in registers.
///
/// transpose_inplace(rows) transposes the square block held in lanes Vecs, one row each, without
/// touching memory. A matrix transpose then loads a tile's rows, transposes them in registers, and
/// stores them as the destination tile's rows. Both the loads and the stores are contiguous, where
/// an element-by-element transpose writes one of the two with a stride.
///
/// Edges that do not fill a whole tile are finished with a scalar loop.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Shuffle.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

/// out (cols x rows) = transpose of in (rows x cols), both row-major.
void transpose(float const *in, std::size_t rows, std::size_t cols, float *out) {
    constexpr std::size_t L = lanes<float>;
    std::size_t const     R = rows - rows % L, C = cols - cols % L;
    for (std::size_t r0 = 0; r0 < R; r0 += L) {
        for (std::size_t c0 = 0; c0 < C; c0 += L) {
            Vec<float> tile[L];
            for (std::size_t i = 0; i < L; ++i) {
                tile[i] = loadu(in + (r0 + i) * cols + c0);
            }
            transpose_inplace(tile);
            for (std::size_t i = 0; i < L; ++i) {
                storeu(out + (c0 + i) * rows + r0, tile[i]);
            }
        }
    }
    // The ragged edges: the columns past C in every row, and the rows past R.
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = (r < R ? C : 0); c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

} // namespace

int einsums_main() {
    std::size_t const  rows = 37, cols = 21;
    std::vector<float> in(rows * cols), out(cols * rows);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>(i);
    }
    transpose(in.data(), rows, cols, out.data());

    int failures = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            failures += out[c * rows + r] != in[r * cols + c];
        }
    }
    std::cout << rows << " x " << cols << " transposed in " << lanes<float> << " x "
              << lanes<float> << " register tiles: " << (failures ? "FAILED" : "ok") << "\n";
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
