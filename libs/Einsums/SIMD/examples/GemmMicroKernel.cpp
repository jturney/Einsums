//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GemmMicroKernel.cpp
/// @brief C += A * B with a register-blocked micro-kernel, the core of every fast GEMM.
///
/// A naive product loads two numbers for every multiply-add, and memory, not arithmetic, sets its
/// speed. A micro-kernel keeps an MR x NR block of C in registers for the whole k loop. Each k step
/// loads one MR-tall column of A (two Vecs here) and NR numbers of B, and does MR x NR multiply-adds
/// with them: 2 * 4 = 8 FMAs for 2 vector loads and 4 broadcasts. The block is written back to C
/// once, at the end.
///
/// MR and NR are chosen so the accumulators, the A column and one broadcast fit the register file:
/// 8 accumulators + 2 + 1 = 11 vector registers, within NEON's 32 and x86's 16. A production GEMM
/// adds packing, so the A and B panels are read contiguously from cache, and cache blocking around
/// this loop; the micro-kernel itself looks like this.
///
/// All three matrices are column-major. Edges that do not fill a whole block use a scalar loop.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace einsums::simd;

namespace {

constexpr std::size_t L  = lanes<double>;
constexpr std::size_t MR = 2 * L; // rows of C per block: two Vecs
constexpr std::size_t NR = 4;     // columns of C per block

/// C[i..i+MR, j..j+NR] += A[i..i+MR, :] * B[:, j..j+NR]. Leading dimensions are the row counts.
void micro_kernel(std::size_t K, double const *A, std::size_t lda, double const *B, std::size_t ldb, double *C, std::size_t ldc) {
    Vec<double> c0[NR], c1[NR];
    for (std::size_t j = 0; j < NR; ++j) {
        c0[j] = broadcast(0.0);
        c1[j] = broadcast(0.0);
    }
    for (std::size_t k = 0; k < K; ++k) {
        Vec<double> const a0 = loadu(A + k * lda);
        Vec<double> const a1 = loadu(A + k * lda + L);
        for (std::size_t j = 0; j < NR; ++j) {
            Vec<double> const b = broadcast(B[k + j * ldb]);
            c0[j]               = fmadd(a0, b, c0[j]);
            c1[j]               = fmadd(a1, b, c1[j]);
        }
    }
    for (std::size_t j = 0; j < NR; ++j) {
        storeu(C + j * ldc, loadu(C + j * ldc) + c0[j]);
        storeu(C + j * ldc + L, loadu(C + j * ldc + L) + c1[j]);
    }
}

/// C (M x N) += A (M x K) * B (K x N), column-major.
void gemm(std::size_t M, std::size_t N, std::size_t K, double const *A, double const *B, double *C) {
    std::size_t const Mb = M - M % MR, Nb = N - N % NR;
    for (std::size_t j = 0; j < Nb; j += NR) {
        for (std::size_t i = 0; i < Mb; i += MR) {
            micro_kernel(K, A + i, M, B + j * K, K, C + i + j * M, M);
        }
    }
    // Scalar edges: the rows below Mb in the blocked columns, then every row of the columns past Nb.
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = (j < Nb ? Mb : 0); i < M; ++i) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i + k * M] * B[k + j * K];
            }
            C[i + j * M] += sum;
        }
    }
}

} // namespace

int einsums_main() {
    std::size_t const   M = 37, N = 19, K = 53;
    std::vector<double> A(M * K), B(K * N), C(M * N, 1.0), expect(M * N, 1.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = std::sin(static_cast<double>(i));
    }
    for (std::size_t i = 0; i < B.size(); ++i) {
        B[i] = std::cos(0.5 * static_cast<double>(i));
    }
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                expect[i + j * M] += A[i + k * M] * B[k + j * K];
            }
        }
    }

    gemm(M, N, K, A.data(), B.data(), C.data());

    double worst = 0.0;
    for (std::size_t i = 0; i < C.size(); ++i) {
        worst = std::max(worst, std::abs(C[i] - expect[i]));
    }
    std::cout << M << " x " << N << " x " << K << " GEMM with a " << MR << " x " << NR << " register block; largest error " << worst
              << "\n";
    return worst < 1e-12 ? 0 : 1;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
