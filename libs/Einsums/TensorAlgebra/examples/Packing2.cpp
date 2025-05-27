//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// dgemm_blocked.cpp
// Simplified blocked DGEMM implementation with packing and a microkernel stub
// Assumes padded input matrices and AVX2 register blocking (MR=4, NR=8)

#include <Einsums/Config.hpp>

#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <cstring>
#include <immintrin.h>
#include <iostream>

constexpr int MC = 256; // L3 blocking for M
constexpr int NC = 256; // L3 blocking for N
constexpr int KC = 128; // L3 blocking for K
constexpr int MR = 4;   // Register block for M (rows)
constexpr int NR = 8;   // Register block for N (columns)

// Packing A (MC x KC) in row-major
void pack_A(double const *EINSUMS_RESTRICT A, int lda, double *EINSUMS_RESTRICT A_pack) {
    A      = EINSUMS_ASSUME_ALIGNED(A, 64);
    A_pack = EINSUMS_ASSUME_ALIGNED(A_pack, 64);

    for (int i = 0; i < MC; ++i) {
        for (int k = 0; k < KC; ++k) {
            A_pack[i * KC + k] = A[i + k * lda];
        }
    }
}

// Packing B (KC x NC) in column-major
void pack_B(double const *EINSUMS_RESTRICT B, int ldb, double *EINSUMS_RESTRICT B_pack) {
    B      = EINSUMS_ASSUME_ALIGNED(B, 64);
    B_pack = EINSUMS_ASSUME_ALIGNED(B_pack, 64);

    for (int k = 0; k < KC; ++k) {
        for (int j = 0; j < NC; ++j) {
            B_pack[k * NC + j] = B[k + j * ldb];
        }
    }
}

// Microkernel: C[MR x NR] += A[MR x KC] * B[KC x NR]
void microkernel(double const *EINSUMS_RESTRICT A, double const *EINSUMS_RESTRICT B, double *EINSUMS_RESTRICT C, int ldc) {
    A = EINSUMS_ASSUME_ALIGNED(A, 64);
    B = EINSUMS_ASSUME_ALIGNED(B, 64);
    C = EINSUMS_ASSUME_ALIGNED(C, 64);

    // Very naive microkernel for demonstration only
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR; ++j) {
            double cij = 0.0;
            for (int k = 0; k < KC; ++k) {
                cij += A[i * KC + k] * B[k * NR + j];
            }
            C[i + j * ldc] += cij;
        }
    }
}

// Top-level blocked DGEMM
void dgemm(int M, int N, int K, double const *EINSUMS_RESTRICT A, int lda, double const *EINSUMS_RESTRICT B, int ldb,
           double *EINSUMS_RESTRICT C, int ldc) {

    einsums::profile::Timer timer("dgemm");

    A = EINSUMS_ASSUME_ALIGNED(A, 64);
    B = EINSUMS_ASSUME_ALIGNED(B, 64);
    C = EINSUMS_ASSUME_ALIGNED(C, 64);

    double *A_pack = (double *)aligned_alloc(64, MC * KC * sizeof(double));
    double *B_pack = (double *)aligned_alloc(64, KC * NC * sizeof(double));

    for (int jc = 0; jc < N; jc += NC) {
        for (int pc = 0; pc < K; pc += KC) {
            pack_B(B + pc + jc * ldb, ldb, B_pack);

            for (int ic = 0; ic < M; ic += MC) {
                pack_A(A + ic + pc * lda, lda, A_pack);

                for (int jr = 0; jr < NC; jr += NR) {
                    for (int ir = 0; ir < MC; ir += MR) {
                        double const *Ab = &A_pack[ir * KC];
                        double const *Bb = &B_pack[jr * KC];
                        double       *Cb = &C[(ic + ir) + (jc + jr) * ldc];

                        // __builtin_prefetch(Ab + MR * KC, 0, 1);
                        // __builtin_prefetch(Bb + KC * NR, 0, 1);

                        microkernel(Ab, Bb, Cb, ldc);
                    }
                }
            }
        }
    }

    delete[] A_pack;
    delete[] B_pack;
}

int einsums_main() {
    using namespace einsums;

    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    // double* A = new double[M * K];
    // double* B = new double[K * N];
    // double* C = new double[M * N];

    auto A = create_ones_tensor("A", M, K);
    auto B = create_ones_tensor("B", M, K);
    auto C = create_zero_tensor("C", M, K);

    // std::fill(A, A + M * K, 1.0);
    // std::fill(B, B + K * N, 1.0);
    // std::fill(C, C + M * N, 0.0);

    dgemm(M, N, K, A.data(), M, B.data(), K, C.data(), M);

    std::cout << "C[0][0] = " << C(0, 0) << std::endl;

    // delete[] A;
    // delete[] B;
    // delete[] C;

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}