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

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;

TEST_CASE("Multi-M: C[i,j,l] = A[i,j,k] * B[k,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 4, 5, 3);
    auto B = create_random_tensor<double>("B", 3, 6);
    auto C = create_zero_tensor<double>("C", 4, 5, 6);

    // Reference via generic algorithm (force generic by using a copy)
    auto C_ref = create_zero_tensor<double>("C_ref", 4, 5, 6);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < 3; kk++) {
                    sum += A(ii, jj, kk) * B(kk, ll);
                }
                C_ref(ii, jj, ll) = sum;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), 1e-12));
            }
        }
    }
}

TEST_CASE("Multi-N: C[i,j,l] = A[i,k] * B[k,j,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5, 6);
    auto C = create_zero_tensor<double>("C", 4, 5, 6);

    auto C_ref = create_zero_tensor<double>("C_ref", 4, 5, 6);
    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < 3; kk++) {
                    sum += A(ii, kk) * B(kk, jj, ll);
                }
                C_ref(ii, jj, ll) = sum;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, k}, A, Indices{k, j, l}, B);

    for (size_t ii = 0; ii < 4; ii++) {
        for (size_t jj = 0; jj < 5; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), 1e-12));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-N: C[i,j,l,m] = A[i,j,k] * B[k,l,m]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto B = create_random_tensor<double>("B", 5, 3, 4);
    auto C = create_zero_tensor<double>("C", 3, 4, 3, 4);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 3, 4);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 3; ll++) {
                for (size_t mm = 0; mm < 4; mm++) {
                    double sum = 0.0;
                    for (size_t kk = 0; kk < 5; kk++) {
                        sum += A(ii, jj, kk) * B(kk, ll, mm);
                    }
                    C_ref(ii, jj, ll, mm) = sum;
                }
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l, m}, &C, Indices{i, j, k}, A, Indices{k, l, m}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 3; ll++) {
                for (size_t mm = 0; mm < 4; mm++) {
                    REQUIRE_THAT(C(ii, jj, ll, mm), Catch::Matchers::WithinRel(C_ref(ii, jj, ll, mm), 1e-12));
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

    // Reference: C_ref += A * B
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < 5; kk++) {
                    sum += A(ii, jj, kk) * B(kk, ll);
                }
                C_ref(ii, jj, ll) += sum;
            }
        }
    }

    // C += A*B (c_prefactor=1, ab_prefactor=1)
    tensor_algebra::einsum(1.0, Indices{i, j, l}, &C, 1.0, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), 1e-12));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-K: C[i,j,l] = A[i,j,k,m] * B[k,m,l]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 2, 3);
    auto B = create_random_tensor<double>("B", 2, 3, 5);
    auto C = create_zero_tensor<double>("C", 3, 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 5);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 5; ll++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < 2; kk++) {
                    for (size_t mm = 0; mm < 3; mm++) {
                        sum += A(ii, jj, kk, mm) * B(kk, mm, ll);
                    }
                }
                C_ref(ii, jj, ll) = sum;
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l}, &C, Indices{i, j, k, m}, A, Indices{k, m, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 5; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), 1e-10));
            }
        }
    }
}

TEST_CASE("Multi-M with alpha scaling", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 5);
    auto B = create_random_tensor<double>("B", 5, 6);
    auto C = create_zero_tensor<double>("C", 3, 4, 6);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 6);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                double sum = 0.0;
                for (size_t kk = 0; kk < 5; kk++) {
                    sum += A(ii, jj, kk) * B(kk, ll);
                }
                C_ref(ii, jj, ll) = 2.5 * sum;
            }
        }
    }

    tensor_algebra::einsum(0.0, Indices{i, j, l}, &C, 2.5, Indices{i, j, k}, A, Indices{k, l}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 6; ll++) {
                REQUIRE_THAT(C(ii, jj, ll), Catch::Matchers::WithinRel(C_ref(ii, jj, ll), 1e-12));
            }
        }
    }
}

TEST_CASE("Multi-M + Multi-K + Multi-N: C[i,j,l,n] = A[i,j,k,m] * B[k,m,l,n]", "[PackedGemm][MultiMN]") {
    auto A = create_random_tensor<double>("A", 3, 4, 2, 3);
    auto B = create_random_tensor<double>("B", 2, 3, 4, 5);
    auto C = create_zero_tensor<double>("C", 3, 4, 4, 5);

    auto C_ref = create_zero_tensor<double>("C_ref", 3, 4, 4, 5);
    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 4; ll++) {
                for (size_t nn = 0; nn < 5; nn++) {
                    double sum = 0.0;
                    for (size_t kk = 0; kk < 2; kk++) {
                        for (size_t mm = 0; mm < 3; mm++) {
                            sum += A(ii, jj, kk, mm) * B(kk, mm, ll, nn);
                        }
                    }
                    C_ref(ii, jj, ll, nn) = sum;
                }
            }
        }
    }

    tensor_algebra::einsum(Indices{i, j, l, n}, &C, Indices{i, j, k, m}, A, Indices{k, m, l, n}, B);

    for (size_t ii = 0; ii < 3; ii++) {
        for (size_t jj = 0; jj < 4; jj++) {
            for (size_t ll = 0; ll < 4; ll++) {
                for (size_t nn = 0; nn < 5; nn++) {
                    REQUIRE_THAT(C(ii, jj, ll, nn), Catch::Matchers::WithinRel(C_ref(ii, jj, ll, nn), 1e-10));
                }
            }
        }
    }
}
