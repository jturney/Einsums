//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Unit tests for the rank-1 updates ger (A += alpha x y^T) and gerc (A += alpha x y^H), through
// the runtime-tensor overloads and on every layout of A.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <array>
#include <complex>
#include <cstring>
#include <limits>
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

// Column-major BLAS sees a row-major A as A^T = alpha conj(y) x^T, so the row-major branch cannot
// swap x and y and call gerc, which would conjugate x instead of y. ger and gerc meet A in column-
// and row-major order, contiguous for BLAS and with every stride doubled for the loop kernel, with
// strided x and y and m != n, so a swapped vector, a misplaced conjugate or a wrong stride shows.
TEMPLATE_TEST_CASE("ger and gerc on every layout of A", "[linear-algebra][ger]", float, double, std::complex<float>, std::complex<double>) {
    using T        = TestType;
    using R        = RemoveComplexT<T>;
    size_t const m = 3, n = 4;

    auto const value = [](double re, double im) {
        if constexpr (IsComplexV<T>) {
            return T(static_cast<R>(re), static_cast<R>(im));
        } else {
            return static_cast<T>(re);
        }
    };

    // x and y step by 2 and 3.
    std::vector<T> x_data(2 * m), y_data(3 * n);
    for (size_t i = 0; i < m; i++) {
        x_data[2 * i] = value(i + 1.0, 0.5 - i);
    }
    for (size_t j = 0; j < n; j++) {
        y_data[3 * j] = value(0.25 * j, j - 1.5);
    }
    detail::TensorImpl<T> const x(x_data.data(), std::array{m}, std::array<size_t, 1>{2});
    detail::TensorImpl<T> const y(y_data.data(), std::array{n}, std::array<size_t, 1>{3});
    T const                     alpha = value(0.75, -1.25);

    struct Layout {
        char const           *name;
        std::array<size_t, 2> strides;
    };
    for (auto const &[name, strides] : {Layout{"column-major", {1, m}}, Layout{"row-major", {n, 1}},
                                        Layout{"strided column-major", {2, 2 * m}}, Layout{"strided row-major", {2 * n, 2}}}) {
        for (bool const conjugate : {false, true}) {
            INFO(name << " A, " << (conjugate ? "gerc" : "ger"));
            std::vector<T>        a_data(4 * m * n);
            detail::TensorImpl<T> A(a_data.data(), std::array{m, n}, strides);
            REQUIRE(A.is_gemmable() == (strides[0] == 1 || strides[1] == 1));
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    A.subscript(i, j) = value(double(i) - double(j), 0.5 * double(i * j));
                }
            }
            std::vector<T> const a0 = a_data;

            if (conjugate) {
                linear_algebra::detail::gerc(alpha, x, y, &A);
            } else {
                linear_algebra::detail::ger(alpha, x, y, &A);
            }

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    INFO("A(" << i << ", " << j << ")");
                    T y_j = y_data[3 * j];
                    if constexpr (IsComplexV<T>) {
                        y_j = conjugate ? std::conj(y_j) : y_j;
                    }
                    T const want = a0[i * strides[0] + j * strides[1]] + alpha * x_data[2 * i] * y_j;
                    CHECK(std::abs(A.subscript(i, j) - want) <= 16 * std::numeric_limits<R>::epsilon() * (1 + std::abs(want)));
                }
            }
        }
    }
}
