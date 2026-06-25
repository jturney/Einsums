//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1: parallel RuntimeTensor coverage for linear_algebra::gemm.
//
// gemm.cpp covers static Tensor<T, 2>. This file exercises the runtime-rank
// overloads added in C.9: linear_algebra::gemm taking BasicTensorConcept
// arguments where at least one operand has Rank == dynamic_rank. Mirrors
// the static-rank tests so any future regression in the runtime path
// surfaces with a focused C++ failure rather than a Python error.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <cstring>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

// Fill a runtime tensor's underlying buffer from a flat vector. Same column-
// major layout as the static-rank tests' ``vector_data() = ...`` assignment.
template <typename T>
void fill_runtime(RuntimeTensor<T> &t, std::vector<T> const &flat) {
    REQUIRE(t.size() == flat.size());
    std::memcpy(t.data(), flat.data(), flat.size() * sizeof(T));
}

template <typename T>
auto runtime_data(RuntimeTensor<T> const &t) -> std::vector<T> {
    std::vector<T> out(t.size());
    std::memcpy(out.data(), t.data(), t.size() * sizeof(T));
    return out;
}

} // namespace

TEMPLATE_TEST_CASE("RuntimeTensor gemm — fixed reference values", "[linear-algebra][runtime]", float, double) {
    // Row-major to match the reference values lifted from gemm.cpp.
    RuntimeTensor<TestType> A("A", {3, 3}, /*row_major=*/true);
    RuntimeTensor<TestType> B("B", {3, 3}, /*row_major=*/true);
    RuntimeTensor<TestType> C("C", {3, 3}, /*row_major=*/true);

    REQUIRE(A.rank() == 2);
    REQUIRE(A.dim(0) == 3);
    REQUIRE(A.dim(1) == 3);

    fill_runtime(A, {TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6), TestType(7), TestType(8), TestType(9)});
    fill_runtime(
        B, {TestType(11), TestType(22), TestType(33), TestType(44), TestType(55), TestType(66), TestType(77), TestType(88), TestType(99)});

    // NN (no transpose, no transpose): reference values from the static gemm.cpp test.
    linear_algebra::gemm<false, false>(TestType{1}, A, B, TestType{0}, &C);
    CHECK_THAT(runtime_data(C),
               Catch::Matchers::Equals(std::vector<TestType>{TestType(330), TestType(396), TestType(462), TestType(726), TestType(891),
                                                             TestType(1056), TestType(1122), TestType(1386), TestType(1650)}));

    // TN
    linear_algebra::gemm<true, false>(TestType{1}, A, B, TestType{0}, &C);
    CHECK_THAT(runtime_data(C),
               Catch::Matchers::Equals(std::vector<TestType>{TestType(726), TestType(858), TestType(990), TestType(858), TestType(1023),
                                                             TestType(1188), TestType(990), TestType(1188), TestType(1386)}));

    // NT
    linear_algebra::gemm<false, true>(TestType{1}, A, B, TestType{0}, &C);
    CHECK_THAT(runtime_data(C),
               Catch::Matchers::Equals(std::vector<TestType>{TestType(154), TestType(352), TestType(550), TestType(352), TestType(847),
                                                             TestType(1342), TestType(550), TestType(1342), TestType(2134)}));

    // TT
    linear_algebra::gemm<true, true>(TestType{1}, A, B, TestType{0}, &C);
    CHECK_THAT(runtime_data(C),
               Catch::Matchers::Equals(std::vector<TestType>{TestType(330), TestType(726), TestType(1122), TestType(396), TestType(891),
                                                             TestType(1386), TestType(462), TestType(1056), TestType(1650)}));
}

TEMPLATE_TEST_CASE("RuntimeTensor gemm — complex dtypes", "[linear-algebra][runtime]", std::complex<float>, std::complex<double>) {
    using R = typename TestType::value_type;

    RuntimeTensor<TestType> A("A", {2, 3});
    RuntimeTensor<TestType> B("B", {3, 4});
    RuntimeTensor<TestType> C("C", {2, 4});

    // Random fill via copy from static tensors so we cross-check against a
    // known-good static-tensor gemm result.
    auto Astatic = create_random_tensor<TestType>("A_s", 2, 3);
    auto Bstatic = create_random_tensor<TestType>("B_s", 3, 4);
    auto Cref    = create_zero_tensor<TestType>("C_ref", 2, 4);

    std::memcpy(A.data(), Astatic.data(), A.size() * sizeof(TestType));
    std::memcpy(B.data(), Bstatic.data(), B.size() * sizeof(TestType));

    linear_algebra::gemm<false, false>(TestType{1, 0}, Astatic, Bstatic, TestType{0, 0}, &Cref);
    linear_algebra::gemm<false, false>(TestType{1, 0}, A, B, TestType{0, 0}, &C);

    auto       got = runtime_data(C);
    auto const ref = std::vector<TestType>(Cref.data(), Cref.data() + Cref.size());
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(std::abs(got[i] - ref[i]) < R{1e-5});
    }
}

TEMPLATE_TEST_CASE("RuntimeTensor gemm — beta scaling accumulates onto C", "[linear-algebra][runtime]", float, double) {
    RuntimeTensor<TestType> A("A", {2, 2}, /*row_major=*/true);
    RuntimeTensor<TestType> B("B", {2, 2}, /*row_major=*/true);
    RuntimeTensor<TestType> C("C", {2, 2}, /*row_major=*/true);

    // Row-major fill: A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = [[1,1],[1,1]].
    fill_runtime(A, {TestType(1), TestType(2), TestType(3), TestType(4)});
    fill_runtime(B, {TestType(5), TestType(6), TestType(7), TestType(8)});
    fill_runtime(C, {TestType(1), TestType(1), TestType(1), TestType(1)});

    // C := 2 * A @ B + 3 * C
    // A @ B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
    //       = [[19, 22],[43, 50]]
    // 2 * A @ B + 3*1 = [[41, 47],[89, 103]] (row-major flat: 41, 47, 89, 103)
    linear_algebra::gemm<false, false>(TestType{2}, A, B, TestType{3}, &C);
    CHECK_THAT(runtime_data(C), Catch::Matchers::Equals(std::vector<TestType>{TestType(41), TestType(47), TestType(89), TestType(103)}));
}

TEST_CASE("RuntimeTensor gemm — rank mismatch surfaces a clean error", "[linear-algebra][runtime]") {
    RuntimeTensor<double> A("A", {3, 3, 3}); // rank-3, invalid for gemm
    RuntimeTensor<double> B("B", {3, 3});
    RuntimeTensor<double> C("C", {3, 3});

    REQUIRE_THROWS_AS((linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C)), einsums::rank_error);
}
