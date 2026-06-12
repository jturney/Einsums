//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1: runtime-tensor coverage for linear_algebra::ger (rank-1 update
// A := A + alpha * x * y^T). LinearAlgebra has no existing CPU unit test
// for ger of any kind; this file establishes baseline coverage for the
// runtime overload added in C.10.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstring>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

namespace {

template <typename T>
auto runtime_data(RuntimeTensor<T> const &t) -> std::vector<T> {
    std::vector<T> out(t.size());
    std::memcpy(out.data(), t.data(), t.size() * sizeof(T));
    return out;
}

template <typename T>
void fill_runtime(RuntimeTensor<T> &t, std::vector<T> const &flat) {
    REQUIRE(t.size() == flat.size());
    std::memcpy(t.data(), flat.data(), flat.size() * sizeof(T));
}

} // namespace

TEMPLATE_TEST_CASE("RuntimeTensor ger — rank-1 update with alpha=1, A starts zero", "[linear-algebra][runtime]", float, double) {
    // 3×2 outer product of x = [1, 2, 3] and y = [4, 5]:
    //   x ⊗ y = [[4, 5], [8, 10], [12, 15]]
    RuntimeTensor<TestType> x("x", {3}, /*row_major=*/true);
    RuntimeTensor<TestType> y("y", {2}, /*row_major=*/true);
    RuntimeTensor<TestType> A("A", {3, 2}, /*row_major=*/true);

    fill_runtime(x, {TestType(1), TestType(2), TestType(3)});
    fill_runtime(y, {TestType(4), TestType(5)});
    fill_runtime(A, std::vector<TestType>(6, TestType(0)));

    linear_algebra::ger(TestType{1}, x, y, &A);

    CHECK_THAT(runtime_data(A), Catch::Matchers::Equals(std::vector<TestType>{TestType(4), TestType(5), TestType(8), TestType(10),
                                                                              TestType(12), TestType(15)}));
}

TEMPLATE_TEST_CASE("RuntimeTensor ger — alpha scaling and accumulation onto non-zero A", "[linear-algebra][runtime]", float, double) {
    // Start A = 1, do A += 2 * x * y^T  with x = [1, 1] and y = [3, 7].
    //   x ⊗ y = [[3, 7], [3, 7]];  2 * (x ⊗ y) = [[6, 14], [6, 14]]
    //   A = 1 + 2*(x⊗y) = [[7, 15], [7, 15]]
    RuntimeTensor<TestType> x("x", {2}, /*row_major=*/true);
    RuntimeTensor<TestType> y("y", {2}, /*row_major=*/true);
    RuntimeTensor<TestType> A("A", {2, 2}, /*row_major=*/true);

    fill_runtime(x, {TestType(1), TestType(1)});
    fill_runtime(y, {TestType(3), TestType(7)});
    fill_runtime(A, {TestType(1), TestType(1), TestType(1), TestType(1)});

    linear_algebra::ger(TestType{2}, x, y, &A);

    CHECK_THAT(runtime_data(A), Catch::Matchers::Equals(std::vector<TestType>{TestType(7), TestType(15), TestType(7), TestType(15)}));
}

TEST_CASE("RuntimeTensor ger — rank mismatch surfaces a clean error", "[linear-algebra][runtime]") {
    RuntimeTensor<double> x("x", {3});
    RuntimeTensor<double> y("y", {3, 3}); // y must be rank-1
    RuntimeTensor<double> A("A", {3, 3});

    REQUIRE_THROWS(linear_algebra::ger(1.0, x, y, &A));
}
