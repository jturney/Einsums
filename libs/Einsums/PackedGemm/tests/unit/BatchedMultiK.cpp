//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Batched multi-K contractions through the flatten path's gather fallback.
//
// nb > 0 always gathers (see the per-slice-HPTT finding in
// EinsumPackedGemm.hpp: the KC-fused gather measured faster than up-front
// slice transposes). These tests pin the gather across the layouts that
// experiment covered: memcpy-friendly stride-1 lead axes, strided lead axes
// (elementwise gather) with the batch axis outermost and buried mid-tensor,
// and complex elements. try_packed_gemm is called directly so the tests
// exercise this backend regardless of what einsum dispatch prefers.

#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {
constexpr size_t NB = 3; // batch_total < 4 keeps the batch loop serial
constexpr size_t NI = 7, NJ = 6, NK = 4, NL = 5;
} // namespace

TEST_CASE("Batched multi-K: stride-1 lead axes (memcpy gather)", "[PackedGemm][BatchedMultiK]") {
    // C[i,j,b] = A[i,k,l,b] * B[j,k,l,b]; m and n sit at axis 0 (stride 1),
    // so both copies are contiguous per-K memcpys.
    auto A = create_random_tensor<double>("A", NI, NK, NL, NB);
    auto B = create_random_tensor<double>("B", NJ, NK, NL, NB);
    auto C = create_zero_tensor<double>("C", NI, NJ, NB);

    auto C_ref = create_zero_tensor<double>("C_ref", NI, NJ, NB);
    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < NK; kk++) {
                    for (size_t ll = 0; ll < NL; ll++) {
                        sum += A(ii, kk, ll, bb) * B(jj, kk, ll, bb);
                    }
                }
                C_ref(ii, jj, bb) = sum;
            }
        }
    }

    bool const handled =
        packed_gemm::try_packed_gemm<false, false>(0.0, Indices{i, j, b}, &C, 1.0, Indices{i, k, l, b}, A, Indices{j, k, l, b}, B);
    REQUIRE(handled);

    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                REQUIRE_THAT(C(ii, jj, bb), Catch::Matchers::WithinRel(C_ref(ii, jj, bb), 1e-12));
            }
        }
    }
}

TEST_CASE("Batched multi-K: strided lead, batch axis outermost", "[PackedGemm][BatchedMultiK]") {
    // C[i,j,b] = A[k,i,l,b] * B[k,j,l,b]; m/n sit behind k (stride NK), so
    // the gather is elementwise. b carries the largest stride: every slice
    // is a contiguous prefix block.
    auto A = create_random_tensor<double>("A", NK, NI, NL, NB);
    auto B = create_random_tensor<double>("B", NK, NJ, NL, NB);
    auto C = create_zero_tensor<double>("C", NI, NJ, NB);

    auto C_ref = create_zero_tensor<double>("C_ref", NI, NJ, NB);
    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < NK; kk++) {
                    for (size_t ll = 0; ll < NL; ll++) {
                        sum += A(kk, ii, ll, bb) * B(kk, jj, ll, bb);
                    }
                }
                C_ref(ii, jj, bb) = sum;
            }
        }
    }

    bool const handled =
        packed_gemm::try_packed_gemm<false, false>(0.0, Indices{i, j, b}, &C, 1.0, Indices{k, i, l, b}, A, Indices{k, j, l, b}, B);
    REQUIRE(handled);

    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                REQUIRE_THAT(C(ii, jj, bb), Catch::Matchers::WithinRel(C_ref(ii, jj, bb), 1e-12));
            }
        }
    }
}

TEST_CASE("Batched multi-K: strided lead, batch axis mid-tensor", "[PackedGemm][BatchedMultiK]") {
    // C[i,j,b] = A[k,b,i,l] * B[k,b,j,l]; the batch axis sits between the K
    // and m axes and m carries stride NK*NB, so the gather is elementwise
    // and every slice offset mixes batch and K strides.
    auto A = create_random_tensor<double>("A", NK, NB, NI, NL);
    auto B = create_random_tensor<double>("B", NK, NB, NJ, NL);
    auto C = create_zero_tensor<double>("C", NI, NJ, NB);

    auto C_ref = create_zero_tensor<double>("C_ref", NI, NJ, NB);
    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < NK; kk++) {
                    for (size_t ll = 0; ll < NL; ll++) {
                        sum += A(kk, bb, ii, ll) * B(kk, bb, jj, ll);
                    }
                }
                C_ref(ii, jj, bb) = sum;
            }
        }
    }

    bool const handled =
        packed_gemm::try_packed_gemm<false, false>(0.0, Indices{i, j, b}, &C, 1.0, Indices{k, b, i, l}, A, Indices{k, b, j, l}, B);
    REQUIRE(handled);

    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                REQUIRE_THAT(C(ii, jj, bb), Catch::Matchers::WithinRel(C_ref(ii, jj, bb), 1e-12));
            }
        }
    }
}

TEST_CASE("Batched multi-K: complex elements, mid-tensor batch axis", "[PackedGemm][BatchedMultiK]") {
    using T = std::complex<double>;

    auto A = create_random_tensor<T>("A", NK, NB, NI, NL);
    auto B = create_random_tensor<T>("B", NK, NB, NJ, NL);
    auto C = create_zero_tensor<T>("C", NI, NJ, NB);

    auto C_ref = create_zero_tensor<T>("C_ref", NI, NJ, NB);
    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                T sum{0.0};
                for (size_t kk = 0; kk < NK; kk++) {
                    for (size_t ll = 0; ll < NL; ll++) {
                        sum += A(kk, bb, ii, ll) * B(kk, bb, jj, ll);
                    }
                }
                C_ref(ii, jj, bb) = sum;
            }
        }
    }

    bool const handled =
        packed_gemm::try_packed_gemm<false, false>(T{0.0}, Indices{i, j, b}, &C, T{1.0}, Indices{k, b, i, l}, A, Indices{k, b, j, l}, B);
    REQUIRE(handled);

    for (size_t bb = 0; bb < NB; bb++) {
        for (size_t ii = 0; ii < NI; ii++) {
            for (size_t jj = 0; jj < NJ; jj++) {
                REQUIRE(std::abs(C(ii, jj, bb) - C_ref(ii, jj, bb)) < 1e-10);
            }
        }
    }
}
