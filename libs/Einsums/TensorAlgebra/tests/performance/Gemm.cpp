//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Performance.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

void serial_mmul(const double *A, const double *B, double *C, std::size_t N) {
    // For each row...
    for (std::size_t row = 0; row < N; row++)
        // For each col...
            for (std::size_t col = 0; col < N; col++)
                // For each element in the row/col pair...
                    for (std::size_t idx = 0; idx < N; idx++)
                        // Accumulate the partial results
                            C[row * N + col] += A[row * N + idx] * B[idx * N + col];
}

void serial_mmul_bench(benchmark::State &s) {
    // Number Dimensions of our matrix
    std::size_t const N = s.range(0);

    auto A = einsums::create_random_tensor("A", N, N);
    auto B = einsums::create_random_tensor("A", N, N);
    auto C = einsums::create_zero_tensor("A", N, N);

    // Main benchmark loop
    for (auto _ : s) {
        serial_mmul(A.data(), B.data(), C.data(), N);
    }
}
BENCHMARK(serial_mmul_bench)
    ->Arg(384)
    ->Arg(768)
    ->Arg(1152)
    ->Unit(benchmark::kMillisecond);