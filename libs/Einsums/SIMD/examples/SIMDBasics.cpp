//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SIMDBasics.cpp
/// @brief Demonstrates the SIMD abstraction layer.
///
/// Shows how to:
///   - Detect the SIMD ISA at compile time
///   - Use Vec<T> for vectorized arithmetic
///   - Perform in-register matrix transpose
///   - Use gather/scatter for non-contiguous access
///   - Work with complex numbers via CVec<T>

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Gather.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/Shuffle.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <complex>
#include <iomanip>
#include <iostream>

using namespace einsums::simd;

int einsums_main() {
    // ── 1. Platform info ────────────────────────────────────────────────────
    std::cout << "SIMD Platform:\n";
    std::cout << "  native_bits:    " << native_bits << "\n";
    std::cout << "  float lanes:    " << native_lanes<float> << "\n";
    std::cout << "  double lanes:   " << native_lanes<double> << "\n";
    std::cout << "  has_sse2:       " << has_sse2 << "\n";
    std::cout << "  has_avx:        " << has_avx << "\n";
    std::cout << "  has_avx2:       " << has_avx2 << "\n";
    std::cout << "  has_avx512:     " << has_avx512 << "\n";
    std::cout << "  has_neon:       " << has_neon << "\n";
    std::cout << "  is_apple_silicon: " << is_apple_silicon << "\n\n";

    // ── 2. Basic Vec<float> arithmetic ──────────────────────────────────────
    constexpr size_t N = native_lanes<float>;
    std::cout << "Vec<float> arithmetic (" << N << " lanes):\n";

    float data_a[N], data_b[N], data_c[N];
    for (size_t i = 0; i < N; i++) {
        data_a[i] = static_cast<float>(i + 1);
        data_b[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Vec<float> a = loadu<float>(data_a);
    Vec<float> b = loadu<float>(data_b);

    // c = a * b + a  (fused multiply-add)
    Vec<float> c = fmadd(a, b, a);
    storeu(data_c, c);

    std::cout << "  a = [";
    for (size_t i = 0; i < N; i++)
        std::cout << (i ? ", " : "") << data_a[i];
    std::cout << "]\n";

    std::cout << "  b = [";
    for (size_t i = 0; i < N; i++)
        std::cout << (i ? ", " : "") << data_b[i];
    std::cout << "]\n";

    std::cout << "  fmadd(a, b, a) = [";
    for (size_t i = 0; i < N; i++)
        std::cout << (i ? ", " : "") << data_c[i];
    std::cout << "]\n\n";

    // ── 3. In-register 4x4 transpose ───────────────────────────────────────
    if constexpr (native_lanes<float> >= 4) {
        std::cout << "4x4 float transpose (in-register):\n";
        float matrix[4][4];
        for (int r = 0; r < 4; r++)
            for (int col = 0; col < 4; col++)
                matrix[r][col] = static_cast<float>(r * 4 + col);

        // Load rows into SIMD registers
        Vec<float> rows[4];
        for (int r = 0; r < 4; r++)
            rows[r] = loadu<float>(matrix[r]);

        std::cout << "  Before:\n";
        for (int r = 0; r < 4; r++) {
            float tmp[4];
            storeu(tmp, rows[r]);
            std::cout << "    [" << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << ", " << tmp[3] << "]\n";
        }

        transpose_inplace(rows);

        std::cout << "  After transpose:\n";
        for (int r = 0; r < 4; r++) {
            float tmp[4];
            storeu(tmp, rows[r]);
            std::cout << "    [" << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << ", " << tmp[3] << "]\n";
        }
        std::cout << "\n";
    }

    // ── 4. Gather with fixed stride ─────────────────────────────────────────
    std::cout << "Gather with stride 3:\n";
    float source[32];
    for (int i = 0; i < 32; i++)
        source[i] = static_cast<float>(i * 10);

    Vec<float> gathered = gather_fixed<3>(source);
    float      result[N];
    storeu(result, gathered);

    std::cout << "  source[0,3,6,9,...] = [";
    for (size_t i = 0; i < N; i++)
        std::cout << (i ? ", " : "") << result[i];
    std::cout << "]\n\n";

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
