//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Phase D.1 — parallel RuntimeTensor coverage for linear_algebra::gemv.
// Mirrors gemv.cpp's static-rank tests against the runtime overload added
// in C.10. Tests both column-major and row-major layouts since the runtime
// path defers layout to the underlying TensorImpl.

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <complex>
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

TEMPLATE_TEST_CASE("RuntimeTensor gemv — both transpose flags, both layouts", "[linear-algebra][runtime]", float, double) {
    auto run_for_layout = [](bool row_major) {
        RuntimeTensor<TestType> A("A", {3, 3}, row_major);
        RuntimeTensor<TestType> x("x", {3}, row_major);
        RuntimeTensor<TestType> y("y", {3}, row_major);

        fill_runtime(A,
                     {TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6), TestType(7), TestType(8), TestType(9)});
        fill_runtime(x, {TestType(11), TestType(22), TestType(33)});

        // gemv<TransA=true>: y = A^T @ x
        linear_algebra::gemv<true>(TestType{1}, A, x, TestType{0}, &y);
        if (A.impl().is_column_major()) {
            CHECK_THAT(runtime_data(y), Catch::Matchers::Equals(std::vector<TestType>{TestType(154), TestType(352), TestType(550)}));
        } else {
            CHECK_THAT(runtime_data(y), Catch::Matchers::Equals(std::vector<TestType>{TestType(330), TestType(396), TestType(462)}));
        }

        // gemv<TransA=false>: y = A @ x
        linear_algebra::gemv<false>(TestType{1}, A, x, TestType{0}, &y);
        if (A.impl().is_column_major()) {
            CHECK_THAT(runtime_data(y), Catch::Matchers::Equals(std::vector<TestType>{TestType(330), TestType(396), TestType(462)}));
        } else {
            CHECK_THAT(runtime_data(y), Catch::Matchers::Equals(std::vector<TestType>{TestType(154), TestType(352), TestType(550)}));
        }
    };

    SECTION("column-major") {
        run_for_layout(false);
    }
    SECTION("row-major") {
        run_for_layout(true);
    }
}

TEST_CASE("RuntimeTensor gemv — rank-error on non-matrix A", "[linear-algebra][runtime]") {
    RuntimeTensor<double> A("A", {3, 3, 3}); // rank-3 ⇒ invalid
    RuntimeTensor<double> x("x", {3});
    RuntimeTensor<double> y("y", {3});

    REQUIRE_THROWS_AS((linear_algebra::gemv<false>(1.0, A, x, 0.0, &y)), einsums::rank_error);
}
