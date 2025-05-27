//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities.hpp>

void naive_dgemm(int n, double const *EINSUMS_RESTRICT A, double const *EINSUMS_RESTRICT B, double *EINSUMS_RESTRICT C) {
    for (int c = 0; c < n; ++c) {
        for (int k = 0; k < n; ++k) {
            for (int r = 0; r < n; ++r) {
                C[r + c * n] += A[r + k * n] * B[k + c * n];
            }
        }
    }
}

void tiled_dgemm(int block_size, int n, double *EINSUMS_RESTRICT A, double *EINSUMS_RESTRICT B, double *EINSUMS_RESTRICT C) {
    A = EINSUMS_ASSUME_ALIGNED(A, 32);
    B = EINSUMS_ASSUME_ALIGNED(B, 32);
    C = EINSUMS_ASSUME_ALIGNED(C, 32);

    // Tiling loops
    for (int br = 0; br < n; br += block_size) {
        for (int bc = 0; bc < n; bc += block_size) {
            for (int bk = 0; bk < n; bk += block_size) {

                // Inner loops
                for (int c = 0; c < block_size; ++c) {
                    for (int k = 0; k < block_size; ++k) {
                        double const b_val = B[(bk + k) + (bc + c) * n];
                        for (int r = 0; r < block_size; ++r) {
                            C[(br + r) + (bc + c) * n] += A[(br + r) + (bk + k) * n] * b_val;
                        }
                    }
                }
            }
        }
    }
}

// Tiled-Packed DGEMM
void multiply_packed_block(double const *EINSUMS_RESTRICT a_packed, double const *EINSUMS_RESTRICT b_packed, double *EINSUMS_RESTRICT C,
                           int const n, int const block_size) {
    a_packed = EINSUMS_ASSUME_ALIGNED(a_packed, 32);
    b_packed = EINSUMS_ASSUME_ALIGNED(b_packed, 32);
    C        = EINSUMS_ASSUME_ALIGNED(C, 32);

    for (int c = 0; c < block_size; ++c) {
        for (int k = 0; k < block_size; ++k) {
            double const b_val = b_packed[c * block_size + k];
            for (int r = 0; r < block_size; ++r) {
                C[r + c * n] += a_packed[r + k * block_size] * b_val;
            }
        }
    }
}

void pack_matrix(double *EINSUMS_RESTRICT dest, double const *EINSUMS_RESTRICT src, int const n, int const block_size) {
    for (int j = 0; j < block_size; j++) {
        for (int i = 0; i < block_size; i++) {
            dest[j * block_size + i] = src[i + j * n];
        }
    }
}

void tiled_packed_extracted_dgemm(int const block_size, int const n, double const *EINSUMS_RESTRICT A, double const *EINSUMS_RESTRICT B,
                                  double *EINSUMS_RESTRICT C) {
    A = EINSUMS_ASSUME_ALIGNED(A, 32);
    B = EINSUMS_ASSUME_ALIGNED(B, 32);
    C = EINSUMS_ASSUME_ALIGNED(C, 32);

    double *a_packed = (double *)aligned_alloc(32, block_size * block_size * sizeof(double));
    double *b_packed = (double *)aligned_alloc(32, block_size * block_size * sizeof(double));

    for (int br = 0; br < n; br += block_size) {
        for (int bc = 0; bc < n; bc += block_size) {
            for (int bk = 0; bk < n; bk += block_size) {
                pack_matrix(a_packed, &A[br + bk * n], n, block_size);
                pack_matrix(b_packed, &B[bk + bc * n], n, block_size);
                multiply_packed_block(a_packed, b_packed, &C[br + bc * n], n, block_size);
            }
        }
    }

    free(a_packed);
    free(b_packed);
}

struct Arguments {
    int n;
    int block_size;
};

int einsums_main(Arguments const &arguments) {
    using namespace einsums;

    println("Arguments: n {} block_size {}", arguments.n, arguments.block_size);

    size_t i = arguments.n;

    auto A = create_incremented_tensor("A", i, i);
    auto B = create_incremented_tensor("B", i, i);
    auto C = create_zero_tensor("C", i, i);
    {
        // profile::Timer t("naive_dgemm");
        // naive_dgemm(i, A.data(), B.data(), C.data());
    }
    {
        // profile::Timer t("tiled_dgemm");
        // tiled_dgemm(arguments.block_size, i, A.data(), B.data(), C.data());
    }
    {
        profile::Timer t("tiled_packed_extracted_dgemm");
        tiled_packed_extracted_dgemm(arguments.block_size, i, A.data(), B.data(), C.data());
    }
    {
        profile::Timer t("BLAS");
        linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
    }
    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    Arguments           arguments{};
    einsums::InitParams params{};
    params.register_arguments = [&arguments](argparse::ArgumentParser &parser) {
        parser.add_argument("--n").default_value(1024).store_into(arguments.n);
        parser.add_argument("--block_size").default_value(128).store_into(arguments.block_size);
    };

    // Another alternative to use a lambda for func is to use std::bind.
    // BUT A GOTCHA is that std::bind always binds by value, and we want a reference here.
    // In this case the bind would have to be std::bind(einsums_main, std:ref(arguments));
    std::function func = [&arguments] { return einsums_main(arguments); };
    return einsums::start(func, argc, argv, params);
}