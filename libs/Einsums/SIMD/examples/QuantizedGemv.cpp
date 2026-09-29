//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file QuantizedGemv.cpp
/// @brief A quantized matrix-vector product: uint8 activations times int8 weights, int32 sums.
///
/// Quantized inference stores weights as int8 and activations as uint8, each with a float scale,
/// and accumulates their products in int32 so nothing overflows. dot_product_us(acc, a, b) does
/// the inner step in one instruction where the hardware has one: each int32 lane of acc gains the
/// sum of four uint8 x int8 products. The exact row sums then go back to float a Vec at a time with
/// convert<float>, and take the product of the two scales.
///
/// The unsigned x signed form exists with Arm FEAT_I8MM and with x86 AVX-VNNI or AVX-512 VNNI.
/// Elsewhere dot_product_us has no definition and a call fails to link, so the vector path below
/// is compiled only under the conditions that provide it, and the program says when it is absent.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Convert.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>
#include <Einsums/SIMD/Reduce.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace einsums::simd;

#if defined(__ARM_FEATURE_MATMUL_INT8) || (defined(__AVX512VNNI__) && defined(__AVX512F__) && defined(__AVX512VL__)) ||                    \
    (defined(__AVXVNNI__) && defined(__AVX2__))
#    define HAVE_DOT_PRODUCT_US 1
#else
#    define HAVE_DOT_PRODUCT_US 0
#endif

namespace {

/// y[r] = (sw * sx) * sum_k x[k] * w[r][k] for an int8 weight matrix of rows x cols, row-major.
/// @p cols must be a multiple of the uint8 lane count.
void qgemv(std::size_t rows, std::size_t cols, std::int8_t const *w, std::uint8_t const *x, float sw, float sx, float *y) {
    // The integer part: one exact int32 sum per row.
    std::vector<std::int32_t> sums(rows);
#if HAVE_DOT_PRODUCT_US
    constexpr std::size_t L = lanes<std::uint8_t>;
    for (std::size_t r = 0; r < rows; ++r) {
        Vec<std::int32_t> acc = broadcast(std::int32_t{0});
        for (std::size_t k = 0; k < cols; k += L) {
            acc = dot_product_us(acc, loadu(x + k), loadu(w + r * cols + k));
        }
        sums[r] = reduce_add(acc);
    }
#else
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t k = 0; k < cols; ++k) {
            sums[r] += static_cast<std::int32_t>(x[k]) * static_cast<std::int32_t>(w[r * cols + k]);
        }
    }
#endif

    // The float part: dequantize a Vec of row sums at a time.
    constexpr std::size_t F     = lanes<float>;
    Vec<float> const      scale = broadcast(sw * sx);
    for (std::size_t r = 0; r < rows; r += F) {
        std::size_t const width = std::min(F, rows - r);
        storeu_partial(y + r, convert<float>(loadu_partial(sums.data() + r, width)) * scale, width);
    }
}

} // namespace

int einsums_main() {
    std::size_t const         rows = 8, cols = std::size_t{4} * lanes<std::uint8_t>;
    std::vector<std::int8_t>  w(rows * cols);
    std::vector<std::uint8_t> x(cols);
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<std::int8_t>(static_cast<int>(i * 37 % 255) - 127);
    }
    for (std::size_t k = 0; k < cols; ++k) {
        x[k] = static_cast<std::uint8_t>(k * 11 % 256);
    }
    float const        sw = 0.02f, sx = 0.1f;
    std::vector<float> y(rows);
    qgemv(rows, cols, w.data(), x.data(), sw, sx, y.data());

    std::cout << (HAVE_DOT_PRODUCT_US ? "int8 dot-product instructions" : "no int8 dot-product instructions in this build; scalar loop")
              << ":\n";
    int failures = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        std::int32_t sum = 0;
        for (std::size_t k = 0; k < cols; ++k) {
            sum += static_cast<std::int32_t>(x[k]) * static_cast<std::int32_t>(w[r * cols + k]);
        }
        float const expect = static_cast<float>(sum) * (sw * sx);
        std::cout << "  y[" << r << "] = " << y[r] << "\n";
        failures += y[r] != expect; // the integer sum is exact, so the float results match exactly
    }
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
