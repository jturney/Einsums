//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The two shape classes PackedGemm hands straight to BLAS instead of packing:
// outer products (no link indices, a K=1 GEMM) and GEMV-shaped contractions
// (no M or no N, where packing copies the largest operand for nothing).
//
// These paths reinterpret the operands' axis groups as flat BLAS shapes, so
// what they must prove is that the flattening agrees with the contraction the
// indices describe - including when a destination prefactor is present, when
// the output's groups are ordered opposite to the operands, and when the
// tensor is a view whose axes do NOT flatten (which must decline and leave the
// result to another path).

#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <cmath>
#include <complex>

#include <Einsums/Testing.hpp>

using namespace einsums;
using namespace einsums::index;
using namespace einsums::tensor_algebra;

namespace {

// Both paths are single-product-per-element (outer) or one K-sum (gemv), so
// there is no long accumulation to reassociate; a tight gate is honest here.
constexpr double kTol = 1e-11;

} // namespace

TEST_CASE("DirectBlas - outer product ij;kl->ijkl matches the definition", "[PackedGemm][DirectBlas]") {
    size_t const n = 6;
    auto         A = create_random_tensor<double>("A", n, n);
    auto         B = create_random_tensor<double>("B", n, n);
    auto         C = create_zero_tensor<double>("C", n, n, n, n);

    einsum(0.0, Indices{i, j, k, l}, &C, 1.0, Indices{i, j}, A, Indices{k, l}, B);

    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                for (size_t d = 0; d < n; d++) {
                    REQUIRE_THAT(C(a, b, c, d), Catch::Matchers::WithinAbs(A(a, b) * B(c, d), kTol));
                }
            }
        }
    }
}

TEST_CASE("DirectBlas - outer product honors the destination prefactor", "[PackedGemm][DirectBlas]") {
    // The k=1 GEMM folds beta in while ger needs a separate pass, so the two
    // sides of that threshold must agree on what a nonzero C_prefactor means.
    size_t const n = 5;
    auto         A = create_random_tensor<double>("A", n, n);
    auto         B = create_random_tensor<double>("B", n, n);
    auto         C = create_random_tensor<double>("C", n, n, n, n);

    Tensor<double, 4> before("before", n, n, n, n);
    for (size_t q = 0; q < C.size(); q++) {
        before.data()[q] = C.data()[q];
    }

    einsum(0.5, Indices{i, j, k, l}, &C, 2.0, Indices{i, j}, A, Indices{k, l}, B);

    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                for (size_t d = 0; d < n; d++) {
                    double const want = 0.5 * before(a, b, c, d) + 2.0 * A(a, b) * B(c, d);
                    REQUIRE_THAT(C(a, b, c, d), Catch::Matchers::WithinAbs(want, kTol));
                }
            }
        }
    }
}

TEST_CASE("DirectBlas - outer product with C's groups in operand-swapped order", "[PackedGemm][DirectBlas]") {
    // "klij <- ij ; kl": B's indices lead C, so the flattened view's row axis
    // comes from B, not A. Getting this backwards transposes the result.
    size_t const n = 4;
    auto         A = create_random_tensor<double>("A", n, n);
    auto         B = create_random_tensor<double>("B", n, n);
    auto         C = create_zero_tensor<double>("C", n, n, n, n);

    einsum(0.0, Indices{k, l, i, j}, &C, 1.0, Indices{i, j}, A, Indices{k, l}, B);

    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                for (size_t d = 0; d < n; d++) {
                    REQUIRE_THAT(C(c, d, a, b), Catch::Matchers::WithinAbs(A(a, b) * B(c, d), kTol));
                }
            }
        }
    }
}

TEST_CASE("DirectBlas - gemv-shaped ijk;jk->i matches the definition", "[PackedGemm][DirectBlas]") {
    size_t const n = 7;
    auto         A = create_random_tensor<double>("A", n, n, n);
    auto         B = create_random_tensor<double>("B", n, n);
    auto         C = create_zero_tensor<double>("C", n);

    einsum(0.0, Indices{i}, &C, 1.0, Indices{i, j, k}, A, Indices{j, k}, B);

    for (size_t a = 0; a < n; a++) {
        double ref = 0.0;
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                ref += A(a, b, c) * B(b, c);
            }
        }
        REQUIRE_THAT(C(a), Catch::Matchers::WithinAbs(ref, kTol));
    }
}

TEST_CASE("DirectBlas - gemv-shaped with the contracted operand first", "[PackedGemm][DirectBlas]") {
    // "i <- jk ; jki": B supplies C here, and its link axes lead - the
    // transposed gemv. Both operand roles and both axis orders are reachable.
    size_t const n = 5;
    auto         A = create_random_tensor<double>("A", n, n);
    auto         B = create_random_tensor<double>("B", n, n, n);
    auto         C = create_zero_tensor<double>("C", n);

    einsum(0.0, Indices{i}, &C, 1.0, Indices{j, k}, A, Indices{j, k, i}, B);

    for (size_t a = 0; a < n; a++) {
        double ref = 0.0;
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                ref += A(b, c) * B(b, c, a);
            }
        }
        REQUIRE_THAT(C(a), Catch::Matchers::WithinAbs(ref, kTol));
    }
}

TEST_CASE("DirectBlas - gemv-shaped honors the destination prefactor", "[PackedGemm][DirectBlas]") {
    size_t const n = 5;
    auto         A = create_random_tensor<double>("A", n, n, n);
    auto         B = create_random_tensor<double>("B", n, n);
    auto         C = create_random_tensor<double>("C", n);

    Tensor<double, 1> before("before", n);
    for (size_t q = 0; q < C.size(); q++) {
        before.data()[q] = C.data()[q];
    }

    einsum(-1.5, Indices{i}, &C, 3.0, Indices{i, j, k}, A, Indices{j, k}, B);

    for (size_t a = 0; a < n; a++) {
        double ref = 0.0;
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                ref += A(a, b, c) * B(b, c);
            }
        }
        REQUIRE_THAT(C(a), Catch::Matchers::WithinAbs(-1.5 * before(a) + 3.0 * ref, kTol));
    }
}

TEST_CASE("DirectBlas - interleaved output declines the flattening", "[PackedGemm][DirectBlas]") {
    // "ikj <- ij ; k": A's indices are NOT contiguous within C, so no flat 2-D
    // view of C exists and neither BLAS shape can express this. The guard has
    // to reject it rather than flatten anyway - the failure mode would be a
    // silently scrambled result, not a crash.
    size_t const n = 4;
    auto         A = create_random_tensor<double>("A", n, n);
    auto         B = create_random_tensor<double>("B", n);
    auto         C = create_zero_tensor<double>("C", n, n, n);

    einsum(0.0, Indices{i, k, j}, &C, 1.0, Indices{i, j}, A, Indices{k}, B);

    for (size_t a = 0; a < n; a++) {
        for (size_t c = 0; c < n; c++) {
            for (size_t b = 0; b < n; b++) {
                REQUIRE_THAT(C(a, c, b), Catch::Matchers::WithinAbs(A(a, b) * B(c), kTol));
            }
        }
    }
}

TEST_CASE("DirectBlas - non-contiguous operand views stay correct", "[PackedGemm][DirectBlas]") {
    // A strided view does not flatten to a BLAS vector with one increment, so
    // the fast paths must decline and let another path produce the answer.
    size_t const n    = 6;
    auto         full = create_random_tensor<double>("full", n, n);
    auto         B    = create_random_tensor<double>("B", n, n);
    auto         C    = create_zero_tensor<double>("C", n / 2, n, n, n);

    // A prefix of the fastest axis: unit stride within a column, but the next
    // axis still steps by the PARENT's extent, so the view has gaps and does
    // not flatten to a single BLAS increment.
    auto Aview = full(Range{0, n / 2}, All);

    einsum(0.0, Indices{i, j, k, l}, &C, 1.0, Indices{i, j}, Aview, Indices{k, l}, B);

    for (size_t a = 0; a < n / 2; a++) {
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                for (size_t d = 0; d < n; d++) {
                    REQUIRE_THAT(C(a, b, c, d), Catch::Matchers::WithinAbs(Aview(a, b) * B(c, d), kTol));
                }
            }
        }
    }
}

TEST_CASE("DirectBlas - complex outer product and gemv", "[PackedGemm][DirectBlas]") {
    using T             = std::complex<double>;
    size_t const n      = 4;
    auto         A      = create_random_tensor<T>("A", n, n);
    auto         B      = create_random_tensor<T>("B", n, n);
    auto         Couter = create_zero_tensor<T>("Couter", n, n, n, n);

    einsum(T{0.0}, Indices{i, j, k, l}, &Couter, T{1.0}, Indices{i, j}, A, Indices{k, l}, B);

    for (size_t a = 0; a < n; a++) {
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                for (size_t d = 0; d < n; d++) {
                    T const want = A(a, b) * B(c, d);
                    REQUIRE_THAT(std::abs(Couter(a, b, c, d) - want), Catch::Matchers::WithinAbs(0.0, kTol));
                }
            }
        }
    }

    auto A3 = create_random_tensor<T>("A3", n, n, n);
    auto Cv = create_zero_tensor<T>("Cv", n);
    einsum(T{0.0}, Indices{i}, &Cv, T{1.0}, Indices{i, j, k}, A3, Indices{j, k}, B);
    for (size_t a = 0; a < n; a++) {
        T ref{};
        for (size_t b = 0; b < n; b++) {
            for (size_t c = 0; c < n; c++) {
                ref += A3(a, b, c) * B(b, c);
            }
        }
        REQUIRE_THAT(std::abs(Cv(a) - ref), Catch::Matchers::WithinAbs(0.0, kTol));
    }
}
