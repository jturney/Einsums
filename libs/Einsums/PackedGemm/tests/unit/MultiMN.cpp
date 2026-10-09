//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Tests for multi-M and multi-N PackedGemm contractions.
// These patterns previously fell to the generic O(N^3) algorithm.

#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <limits>
#include <tuple>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

namespace {
// Tolerance for "same contraction, different summation order".
//
// create_random_tensor draws from [-1, 1], so the K-sum cancels: an unlucky
// draw lands |C| near zero, where the reference triple loop and the einsum
// kernel - which accumulate the same products in different orders (and, per
// SIMD rung, with FMA vs separate mul/add) - disagree by the usual
// floating-point reassociation bound. That bound is set by the MAGNITUDE OF
// THE TERMS, not the magnitude of the result, so a fixed relative gate is
// wrong whenever the sum cancels and reads as a mysterious "0.000483 and
// 0.000483" mismatch. This is why the simd_v3 rung flaked in nightly while the
// other rungs passed on the same seed. Mirrors BatchedMultiK.cpp.
//
// So: accept either tight relative agreement (the meaningful check when the
// result is well-scaled) or agreement within the accumulation's own error
// bound.
constexpr double kRelTol = 1e-12;

double reassociation_tol(double term_magnitude) {
    // 32x headroom over the strict n*eps*scale bound; n here is the K extent (<= 6).
    return 32.0 * std::numeric_limits<double>::epsilon() * term_magnitude;
}
} // namespace

TEST_CASE("Multi-M: C[i,j,l] = A[i,j,k] * B[k,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 4, 5, 3);
    auto B = create_random_tensor<double>("B", 3, 6);
    auto C = create_zero_tensor<double>("C", 4, 5, 6);

    // Reference via generic algorithm (force generic by using a copy)
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 5, 6);
    auto C_mag = create_zero_tensor<double>("C_mag", 4, 5, 6); // sum|terms|, sets the reassociation bound
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0, mag = 0.0;
                for (size_t kk = 0; kk < 3; kk++) {
                    double const term = A(ii, jj, kk) * B(kk, ll);
                    sum += term;
                    mag += std::abs(term);
                }
                C_ref(ii, jj, ll) = sum;
                C_mag(ii, jj, ll) = mag;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), kRelTol) ||
                                                Catch::Matchers::WithinAbs(C_ref(ii, jj, ll), reassociation_tol(C_mag(ii, jj, ll))));
            }
        }
    }
}

TEST_CASE("Multi-N: C[i,j,l] = A[i,k] * B[k,j,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5, 6);
    auto C = create_zero_tensor<double>("C", 4, 5, 6);

    auto C_ref = create_zero_tensor<double>("C_ref", 4, 5, 6);
    auto C_mag = create_zero_tensor<double>("C_mag", 4, 5, 6);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0, mag = 0.0;
                for (size_t kk = 0; kk < 3; kk++) {
                    double const term = A(ii, kk) * B(kk, jj, ll);
                    sum += term;
                    mag += std::abs(term);
                }
                C_ref(ii, jj, ll) = sum;
                C_mag(ii, jj, ll) = mag;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, k}, A, Indices{k, j, l}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), kRelTol) ||
                                                Catch::Matchers::WithinAbs(C_ref(ii, jj, ll), reassociation_tol(C_mag(ii, jj, ll))));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-N: C[i,j,l,m] = A[i,j,k] * B[k,l,m]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto B = create_random_tensor<double>("B", 5, 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 4, 3, 4);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 3, 4);
    auto C_mag = create_zero_tensor<double>("C_mag", 3, 4, 3, 4);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 3; ll++) {
                for (size_t mm = 0; mm < 4; mm++) {
                    double sum = 0.0, mag = 0.0;
                    for (size_t kk = 0; kk < 5; kk++) {
                        double const term = A(ii, jj, kk) * B(kk, ll, mm);
                        sum += term;
                        mag += std::abs(term);
                    }
                    C_ref(ii, jj, ll, mm) = sum;
                    C_mag(ii, jj, ll, mm) = mag;
                }
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l, m}, &C, Indices{i, j, k}, A, Indices{k, l, m}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 3; ll++) {
                for (size_t mm = 0; mm < 4; mm++) {
                    REQUIRE_THAT(C(ii, jj, ll, mm),
                                 Catch::Matchers::WithinRel(C_ref(ii, jj, ll, mm), kRelTol) ||
                                     Catch::Matchers::WithinAbs(C_ref(ii, jj, ll, mm), reassociation_tol(C_mag(ii, jj, ll, mm))));
                }
            }
        }
    }
}

TEST_CASE("Multi-M with beta accumulate: C += A * B", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto B = create_random_tensor<double>("B", 5, 6);
    auto C = create_random_tensor<double>("C", 3, 4, 6);

    auto C_ref = Tensor<double, 3>(C); // Copy initial C
    auto C_mag = create_zero_tensor<double>("C_mag", 3, 4, 6);

    // Reference: C_ref += A * B
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0, mag = std::abs(C_ref(ii, jj, ll)); // beta*C is a term too
                for (size_t kk = 0; kk < 5; kk++) {
                    double const term = A(ii, jj, kk) * B(kk, ll);
                    sum += term;
                    mag += std::abs(term);
                }
                C_ref(ii, jj, ll) += sum;
                C_mag(ii, jj, ll) = mag;
            }
        }
    }

    // C += A*B (c_prefactor=1, ab_prefactor=1)
    tensor_algebra::einsum(1.0, Indices{i, j, l}, &C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), kRelTol) ||
                                                Catch::Matchers::WithinAbs(C_ref(ii, jj, ll), reassociation_tol(C_mag(ii, jj, ll))));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-K: C[i,j,l] = A[i,j,k,m] * B[k,m,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 2, 3);
    auto B = create_random_tensor<double>("B", 2, 3, 5);
    auto C = create_zero_tensor<double>("C", 3, 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 5);
    auto C_mag = create_zero_tensor<double>("C_mag", 3, 4, 5);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 5; ll++) {
                double sum = 0.0, mag = 0.0;
                for (size_t kk = 0; kk < 2; kk++) {
                    for (size_t mm = 0; mm < 3; mm++) {
                        double const term = A(ii, jj, kk, mm) * B(kk, mm, ll);
                        sum += term;
                        mag += std::abs(term);
                    }
                }
                C_ref(ii, jj, ll) = sum;
                C_mag(ii, jj, ll) = mag;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, j, k, m}, A, Indices{k, m, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 5; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), kRelTol) ||
                                                Catch::Matchers::WithinAbs(C_ref(ii, jj, ll), reassociation_tol(C_mag(ii, jj, ll))));
            }
        }
    }
}

TEST_CASE("Multi-M with alpha scaling", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto B = create_random_tensor<double>("B", 5, 6);
    auto C = create_zero_tensor<double>("C", 3, 4, 6);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 6);
    auto C_mag = create_zero_tensor<double>("C_mag", 3, 4, 6);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0, mag = 0.0;
                for (size_t kk = 0; kk < 5; kk++) {
                    double const term = A(ii, jj, kk) * B(kk, ll);
                    sum += term;
                    mag += std::abs(term);
                }
                C_ref(ii, jj, ll) = 2.5 * sum;
                C_mag(ii, jj, ll) = 2.5 * mag;
            }
        }
    }

    tensor_algebra::einsum(0.0, Indices{i, j, l}, &C, 2.5, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), kRelTol) ||
                                                Catch::Matchers::WithinAbs(C_ref(ii, jj, ll), reassociation_tol(C_mag(ii, jj, ll))));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-K + Multi-N: C[i,j,l,n] = A[i,j,k,m] * B[k,m,l,n]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 2, 3);
    auto B = create_random_tensor<double>("B", 2, 3, 4, 5);
    auto C = create_zero_tensor<double>("C", 3, 4, 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 4, 5);
    auto C_mag = create_zero_tensor<double>("C_mag", 3, 4, 4, 5);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 4; ll++) {
                for (size_t nn = 0; nn < 5; nn++) {
                    double sum = 0.0, mag = 0.0;
                    for (size_t kk = 0; kk < 2; kk++) {
                        for (size_t mm = 0; mm < 3; mm++) {
                            double const term = A(ii, jj, kk, mm) * B(kk, mm, ll, nn);
                            sum += term;
                            mag += std::abs(term);
                        }
                    }
                    C_ref(ii, jj, ll, nn) = sum;
                    C_mag(ii, jj, ll, nn) = mag;
                }
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l, n}, &C, Indices{i, j, k, m}, A, Indices{k, m, l, n}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 4; ll++) {
                for (size_t nn = 0; nn < 5; nn++) {
                    REQUIRE_THAT(C(ii, jj, ll, nn),
                                 Catch::Matchers::WithinRel(C_ref(ii, jj, ll, nn), kRelTol) ||
                                     Catch::Matchers::WithinAbs(C_ref(ii, jj, ll, nn), reassociation_tol(C_mag(ii, jj, ll, nn))));
                }
            }
        }
    }
}

// Multi-K where only the faster K index is contiguous in the operands, as in the
// per-pair DF ladder's second step (a,b <- Q,a,f ; Q,b,f): each operand's rows
// are K-contiguous in runs of k's extent, one run per value of l. The packers
// transpose such a block run by run when the runs average a vector's worth of
// elements, and gather it element by element when they are shorter. The cases
// cover runs inside one K block with a ragged panel of rows, a K block boundary
// falling inside a run, and runs too short for the run-wise path. Double only: on
// a rung without SME PackedGemm declines float here, and the Sort+GEMM it falls
// back to hits the TensorPermute non-unit inner stride bug (see that module's
// "a fastest axis with a non-unit stride"); PackTranspose covers the float
// transpose itself.
TEST_CASE("Multi-K in contiguous runs: C[i,j] = A[k,i,l] * B[k,j,l]", "[PackedGemm][MultiMN]") {
    using T = double;
    for (auto [nk, ni, nj, nl] :
         {std::tuple{size_t{20}, size_t{17}, size_t{19}, size_t{5}}, std::tuple{size_t{1000}, size_t{17}, size_t{9}, size_t{5}},
          std::tuple{size_t{3}, size_t{17}, size_t{19}, size_t{7}}}) {
        CAPTURE(nk, ni, nj, nl);
        auto A = create_random_tensor<T>("A", nk, ni, nl);
        auto B = create_random_tensor<T>("B", nk, nj, nl);
        // C is a slice with no unit stride, as the ladder's r2[i,j] is. A
        // contiguous C takes the flatten + GEMM route instead, which never packs.
        auto C_parent = create_zero_tensor<T>("C", 3, ni, nj);
        auto C        = C_parent(1, All, All);

        tensor_algebra::einsum(Indices{i, j}, &C, Indices{k, i, l}, A, Indices{k, j, l}, B);

        // The tolerance follows the K-sum's own error bound, which grows with the
        // number of terms; the terms' magnitude sets its scale.
        double const eps = static_cast<double>(std::numeric_limits<T>::epsilon());
        for (size_t ii = 0; ii < ni; ii++) {
            for (size_t jj = 0; jj < nj; jj++) {
                double sum = 0.0, mag = 0.0;
                for (size_t ll = 0; ll < nl; ll++) {
                    for (size_t kk = 0; kk < nk; kk++) {
                        double const term = static_cast<double>(A(kk, ii, ll)) * static_cast<double>(B(kk, jj, ll));
                        sum += term;
                        mag += std::abs(term);
                    }
                }
                double const tol = 4.0 * static_cast<double>(nk * nl) * eps * mag;
                REQUIRE_THAT(static_cast<double>(C(ii, jj)), Catch::Matchers::WithinAbs(sum, tol));
            }
        }
    }
}
