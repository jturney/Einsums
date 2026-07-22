//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomDefinite.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>
#include <Einsums/TensorUtilities/Diagonal.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("pow", "[linear-algebra]", float, double) {

    // pow() reconstructs A^p through an eigendecomposition, then this test
    // compares it against an independent path (a direct gemm for integer
    // powers, a hand-built reference otherwise). The two paths accumulate
    // rounding differently, so a flat WithinAbs(1e-5) was ~1.7e-6 relative at
    // these magnitudes - far too tight for float32 (it tripped under the
    // sanitizer build's altered rounding). A pure WithinRel would not work
    // either: A has entries in [-1, 1), so A@A off-diagonals are cancellation
    // sums that land near zero, where WithinRel's eps*|target| tolerance
    // collapses to ~0. The combined matcher below scales with magnitude for
    // large entries and keeps an absolute floor at zero; one pair of
    // tolerances covers both float and double (double satisfies both
    // trivially). It is spelled inline at each check because Catch2's matcher
    // operator|| holds its operands by reference - a helper returning the
    // combined matcher would dangle.
    auto const rel_tol = TestType{1e-4};
    auto const abs_tol = TestType{1e-5};

    SECTION("Integer power") {

        Tensor A = create_random_tensor<TestType>("A", 10, 10);

        // Can only handle symmetric matrices.
        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < i; j++) {
                A(i, j) = A(j, i);
            }
        }

        Tensor C = create_tensor_like(A);

        linear_algebra::gemm<false, false>(1.0, A, A, 0.0, &C);

        Tensor B = linear_algebra::pow(A, TestType{2.0});

        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                CHECK_THAT(B(i, j), Catch::Matchers::WithinRel(C(i, j), rel_tol) || Catch::Matchers::WithinAbs(C(i, j), abs_tol));
            }
        }
    }

    SECTION("Non-integer power") {
        constexpr int       size = 3;
        Tensor<TestType, 2> A{"A", size, size};
        Tensor<TestType, 2> B{"B", size, size};
        Tensor<TestType, 2> C{"C", size, size};

        std::default_random_engine engine(Catch::rngSeed());

        // Generate a positive definite matrix for A.
        auto normal = std::normal_distribution<TestType>(1, 3);

        // Set up the diagonal.
        Tensor<TestType, 1> Evals{"Evals", size};

        for (int i = 0; i < size; i++) {
            TestType val;

            // If we get zero, reroll until not zero.
            do {
                val = normal(engine);
            } while (val == TestType{0.0});

            Evals(i) = std::abs(val); // Can't have negatives in a positive definite matrix.
        }

        // Generate the eigenvectors.
        Tensor<TestType, 2> Evecs = create_random_tensor<TestType>("Evecs", size, size);

        // Set up the first vector.
        auto v1 = Evecs(All, 0);

        auto norm = linear_algebra::vec_norm(v1);
        v1 /= norm;

        // Orthogonalize.
        for (int i = 1; i < size; i++) {
            auto qi = Evecs(All, i);
            for (int j = 0; j < i; j++) {
                auto qj = Evecs(All, j);

                auto proj = linear_algebra::true_dot(qi, qj);

                // axpy(-proj, qj, &qi);
                for (int k = 0; k < size; k++) {
                    qi(k) -= proj * qj(k);
                }
            }

            while (linear_algebra::vec_norm(qi) < TestType{1e-5}) {
                qi = create_random_tensor<TestType>("new vec", size);
                for (int j = 0; j < i; j++) {
                    auto qj = Evecs(All, j);

                    auto proj = linear_algebra::true_dot(qi, qj);

                    // axpy(-proj, qj, &qi);
                    for (int k = 0; k < size; k++) {
                        qi(k) -= proj * qj(k);
                    }
                }
            }
            qi /= linear_algebra::vec_norm(qi);
        }

        // Create the test tensors.
        auto Diag1 = diagonal(Evals);

        for (int i = 0; i < size; i++) {
            Evals(i) = std::pow(Evals(i), TestType{-0.5});
        }
        auto Diag2 = diagonal(Evals);

        linear_algebra::gemm<false, true>(TestType{1.0}, Diag1, Evecs, TestType{0.0}, &C);
        linear_algebra::gemm<false, false>(TestType{1.0}, Evecs, C, TestType{0.0}, &A);
        linear_algebra::gemm<false, true>(TestType{1.0}, Diag2, Evecs, TestType{0.0}, &C);
        linear_algebra::gemm<false, false>(TestType{1.0}, Evecs, C, TestType{0.0}, &B);

        C.zero();
        C = linear_algebra::pow(A, TestType{-0.5});

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                CHECK_THAT(B(i, j), Catch::Matchers::WithinRel(C(i, j), rel_tol) || Catch::Matchers::WithinAbs(C(i, j), abs_tol));
                CHECK_THAT(C(j, i), Catch::Matchers::WithinRel(C(i, j), rel_tol) || Catch::Matchers::WithinAbs(C(i, j), abs_tol));
                CHECK_THAT(B(j, i), Catch::Matchers::WithinRel(B(i, j), rel_tol) || Catch::Matchers::WithinAbs(B(i, j), abs_tol));
            }
        }
    }

    SECTION("Integer power blocks") {

        BlockTensor<TestType, 2> A("A", 4, 0, 1, 2);

        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            A[i] = create_random_tensor("block", A.block_dim(i), A.block_dim(i));
            // Can only handle symmetric matrices.
            for (int j = 0; j < A.block_dim(i); j++) {
                for (int k = 0; k < j; k++) {
                    A[i](j, k) = A[i](k, j);
                }
            }
        }

        BlockTensor B = einsums::linear_algebra::pow(A, TestType{2.0});
        BlockTensor C = create_tensor_like(A);
        C.zero();

        linear_algebra::gemm<false, false>(1.0, A, A, 0.0, &C);

        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                CHECK_THAT(B(i, j), Catch::Matchers::WithinRel(C(i, j), rel_tol) || Catch::Matchers::WithinAbs(C(i, j), abs_tol));
            }
        }
    }

    SECTION("Non-integer power blocks") {

        BlockTensor<TestType, 2> A("A", 4, 0, 1, 2);

        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            A[i] = create_random_definite("block", A.block_dim(i), A.block_dim(i));
        }

        BlockTensor<TestType, 2> B      = einsums::linear_algebra::pow(A, TestType{-0.5});
        Tensor<TestType, 2>      A_base = (Tensor<TestType, 2>)A;
        Tensor<TestType, 2>      C      = einsums::linear_algebra::pow(A_base, TestType{-0.5});

        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                CHECK_THAT(B(i, j), Catch::Matchers::WithinRel(C(i, j), rel_tol) || Catch::Matchers::WithinAbs(C(i, j), abs_tol));
            }
        }
    }
}