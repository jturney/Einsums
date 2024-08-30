//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/modules/tensor_utilities.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

// TEMPLATE_TEST_CASE("pow", "[linear-algebra]", float, double) {
//
//     //    using TestType = double;
//
//     SECTION("Integer power") {
//
//         tensor A = create_random_tensor<TestType>("A", 10, 10);
//         // Can only handle symmetric matrices.
//         for (int i = 0; i < A.dim(0); i++) {
//             for (int j = 0; j < i; j++) {
//                 A(i, j) = A(j, i);
//             }
//         }
//
//         tensor B = linear_algebra::pow(A, TestType{2.0});
//         tensor C = create_tensor_like(A);
//
//         gemm<false, false>(1.0, A, A, 0.0, &C);
//
//         for (int i = 0; i < A.dim(0); i++) {
//             for (int j = 0; j < A.dim(1); j++) {
//                 CHECK_THAT(B(i, j), Catch::Matchers::WithinAbs(C(i, j), TestType{1e-6}));
//             }
//         }
//     }
//
//     SECTION("Non-integer power") {
//         constexpr int       size = 3;
//         tensor<TestType, 2> A{"A", size, size};
//         tensor<TestType, 2> B{"B", size, size};
//         tensor<TestType, 2> C{"C", size, size};
//
//         std::default_random_engine engine(Catch::rngSeed());
//
//         // Generate a positive definite matrix for A.
//         auto normal = std::normal_distribution<TestType>(1, 3);
//
//         // Set up the diagonal.
//         tensor<TestType, 1> Evals{"Evals", size};
//
//         for (int i = 0; i < size; i++) {
//             TestType val;
//
//             // If we get zero, reroll until not zero.
//             do {
//                 val = normal(engine);
//             } while (val == TestType{0.0});
//
//             Evals(i) = std::abs(val); // Can't have negatives in a positive definite matrix.
//         }
//
//         // Generate the eigenvectors.
//         tensor<TestType, 2> Evecs = create_random_tensor<TestType>("Evecs", size, size);
//
//         // Set up the first vector.
//         auto v1 = Evecs(all, 0);
//
//         auto norm = vec_norm(v1);
//         v1 /= norm;
//
//         // Orthogonalize.
//         for (int i = 1; i < size; i++) {
//             auto qi = Evecs(all, i);
//             for (int j = 0; j < i; j++) {
//                 auto qj = Evecs(all, j);
//
//                 auto proj = true_dot(qi, qj);
//
//                 // axpy(-proj, qj, &qi);
//                 for (int k = 0; k < size; k++) {
//                     qi(k) -= proj * qj(k);
//                 }
//             }
//
//             while (vec_norm(qi) < TestType{1e-6}) {
//                 qi = create_random_tensor<TestType>("new vec", size);
//                 for (int j = 0; j < i; j++) {
//                     auto qj = Evecs(all, j);
//
//                     auto proj = true_dot(qi, qj);
//
//                     // axpy(-proj, qj, &qi);
//                     for (int k = 0; k < size; k++) {
//                         qi(k) -= proj * qj(k);
//                     }
//                 }
//             }
//             qi /= linear_algebra::vec_norm(qi);
//         }
//
//         // Create the test tensors.
//         auto Diag1 = diagonal(Evals);
//
//         for (int i = 0; i < size; i++) {
//             Evals(i) = std::pow(Evals(i), TestType{0.5});
//         }
//         auto Diag2 = diagonal(Evals);
//
//         linear_algebra::gemm<false, true>(TestType{1.0}, Diag1, Evecs, TestType{0.0}, &C);
//         linear_algebra::gemm<false, false>(TestType{1.0}, Evecs, C, TestType{0.0}, &A);
//         linear_algebra::gemm<false, true>(TestType{1.0}, Diag2, Evecs, TestType{0.0}, &C);
//         linear_algebra::gemm<false, false>(TestType{1.0}, Evecs, C, TestType{0.0}, &B);
//
//         C.zero();
//         C = linear_algebra::pow(A, TestType{0.5});
//
//         for (int i = 0; i < size; i++) {
//             for (int j = 0; j < size; j++) {
//                 CHECK_THAT(B(i, j), Catch::Matchers::WithinAbs(C(i, j), TestType{1e-6}));
//                 CHECK_THAT(C(j, i), Catch::Matchers::WithinAbs(C(i, j), TestType{1e-6}));
//                 CHECK_THAT(B(j, i), Catch::Matchers::WithinAbs(B(i, j), TestType{1e-6}));
//             }
//         }
//     }
//
//     SECTION("Integer power blocks") {
//
//         block_tensor<TestType, 2> A("A", 4, 0, 1, 2);
//
//         for (int i = 0; i < A.num_blocks(); i++) {
//             if (A.block_dim(i) == 0) {
//                 continue;
//             }
//             A[i] = create_random_tensor("block", A.block_dim(i), A.block_dim(i));
//             // Can only handle symmetric matrices.
//             for (int j = 0; j < A.block_dim(i); j++) {
//                 for (int k = 0; k < j; k++) {
//                     A[i](j, k) = A[i](k, j);
//                 }
//             }
//         }
//
//         block_tensor B = einsums::linear_algebra::pow(A, TestType{2.0});
//         block_tensor C = create_tensor_like(A);
//         C.zero();
//
//         gemm<false, false>(1.0, A, A, 0.0, &C);
//
//         for (int i = 0; i < A.dim(0); i++) {
//             for (int j = 0; j < A.dim(1); j++) {
//                 CHECK_THAT(B(i, j), Catch::Matchers::WithinAbs(C(i, j), TestType{1e-6}));
//             }
//         }
//     }
//
//     SECTION("Non-integer power blocks") {
//
//         block_tensor<TestType, 2> A("A", 4, 0, 1, 2);
//
//         for (int i = 0; i < A.num_blocks(); i++) {
//             if (A.block_dim(i) == 0) {
//                 continue;
//             }
//             A[i] = create_random_definite("block", A.block_dim(i), A.block_dim(i));
//         }
//
//         BlockTensor<TestType, 2> B      = einsums::linear_algebra::pow(A, TestType{0.5});
//         Tensor<TestType, 2>      A_base = (Tensor<TestType, 2>)A;
//         Tensor<TestType, 2>      C      = einsums::linear_algebra::pow(A_base, TestType{0.5});
//
//         for (int i = 0; i < A.dim(0); i++) {
//             for (int j = 0; j < A.dim(1); j++) {
//                 CHECK_THAT(B(i, j), Catch::Matchers::WithinAbs(C(i, j), TestType{1e-6}));
//             }
//         }
//     }
// }