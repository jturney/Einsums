//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Testing/TensorCompare.hpp>

#include <complex>
#include <limits>

#include <Einsums/Testing.hpp>

using einsums::Tensor;
using einsums::TensorView;
using einsums::testing::bytes_of;
using einsums::testing::compare_tensors;
using einsums::testing::require_tensors_close;

namespace {

Tensor<double, 2> filled(size_t rows, size_t cols, double base) {
    Tensor<double, 2> t("t", rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t(i, j) = base + static_cast<double>(10 * i + j);
        }
    }
    return t;
}

} // namespace

TEST_CASE("compare_tensors - identical tensors are close", "[testing][compare]") {
    auto const a = filled(3, 4, 1.0);
    auto const b = filled(3, 4, 1.0);

    auto const result = compare_tensors(a, b, {.rtol = 0, .atol = 0});
    CHECK(result.close());
    CHECK(result.checked == 12);
    require_tensors_close(a, b, {.rtol = 0, .atol = 0});
}

TEST_CASE("compare_tensors - applies atol + rtol * |want|", "[testing][compare]") {
    auto       want = filled(2, 2, 100.0);
    auto const got  = filled(2, 2, 100.0);
    want(1, 1) += 1e-9; // |diff| 1e-9 at a value near 111

    CHECK_FALSE(compare_tensors(got, want, {.rtol = 1e-12, .atol = 0}).close());
    CHECK(compare_tensors(got, want, {.rtol = 1e-10, .atol = 0}).close());
    CHECK(compare_tensors(got, want, {.rtol = 0, .atol = 1e-8}).close());

    SECTION("near zero only the absolute term can pass") {
        Tensor<double, 1> zero("zero", 1), tiny("tiny", 1);
        zero(0) = 0.0;
        tiny(0) = 1e-13;
        CHECK_FALSE(compare_tensors(tiny, zero, {.rtol = 1e-6, .atol = 0}).close());
        CHECK(compare_tensors(tiny, zero, {.rtol = 0, .atol = 1e-12}).close());
    }
}

TEST_CASE("compare_tensors - a NaN is never close", "[testing][compare]") {
    auto const want = filled(2, 3, 0.0);
    auto       got  = filled(2, 3, 0.0);
    got(0, 2)       = std::numeric_limits<double>::quiet_NaN();

    auto const result = compare_tensors(got, want, {.rtol = 1e300, .atol = 1e300});
    CHECK_FALSE(result.close());
    CHECK(result.failures == 1);
}

TEST_CASE("compare_tensors - names the first element out of tolerance", "[testing][compare]") {
    auto const want = filled(3, 4, 0.0);
    auto       got  = filled(3, 4, 0.0);
    got(1, 2) += 1.0;
    got(2, 3) += 1.0;

    auto const result = compare_tensors(got, want, {.rtol = 0, .atol = 1e-12});
    CHECK(result.failures == 2);
    CHECK_THAT(result.describe(), Catch::Matchers::ContainsSubstring("2 of 12 elements out of tolerance"));
    CHECK_THAT(result.describe(), Catch::Matchers::ContainsSubstring("index (1, 2)"));
}

TEST_CASE("compare_tensors - different extents are a mismatch, not a comparison", "[testing][compare]") {
    auto const a = filled(3, 4, 0.0);
    auto const b = filled(4, 3, 0.0);

    auto const result = compare_tensors(a, b, {.rtol = 1, .atol = 1});
    CHECK_FALSE(result.close());
    CHECK(result.checked == 0);
    CHECK_THAT(result.describe(), Catch::Matchers::ContainsSubstring("got extents (3, 4) but want (4, 3)"));
}

TEST_CASE("compare_tensors - complex values compare by magnitude", "[testing][compare]") {
    Tensor<std::complex<double>, 1> want("want", 2), got("got", 2);
    want(0) = {1.0, 1.0};
    want(1) = {-2.0, 0.5};
    got(0)  = {1.0, 1.0 + 3e-12};
    got(1)  = {-2.0, 0.5};

    CHECK(compare_tensors(got, want, {.rtol = 0, .atol = 5e-12}).close());
    CHECK_FALSE(compare_tensors(got, want, {.rtol = 0, .atol = 1e-12}).close());
}

TEST_CASE("compare_tensors and bytes_of - a view is walked by its strides", "[testing][compare]") {
    auto const                  parent = filled(5, 6, 0.0);
    TensorView<double, 2> const view{parent, einsums::Dim<2>{2, 3}, einsums::Offset<2>{1, 2}};
    Tensor<double, 2>           packed("packed", 2, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            packed(i, j) = parent(i + 1, j + 2);
        }
    }

    CHECK(compare_tensors(view, packed, {.rtol = 0, .atol = 0}).close());
    CHECK(bytes_of(view) == bytes_of(packed));
}

TEST_CASE("bytes_of - tells -0.0 from 0.0", "[testing][compare]") {
    Tensor<double, 1> a("a", 1), b("b", 1);
    a(0) = 0.0;
    b(0) = -0.0;

    CHECK(compare_tensors(a, b, {.rtol = 0, .atol = 0}).close());
    CHECK(bytes_of(a) != bytes_of(b));
}
