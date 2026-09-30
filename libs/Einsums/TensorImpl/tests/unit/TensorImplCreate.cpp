//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("TensorImpl Creation", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    SECTION("Default constructor") {
        detail::TensorImpl<TestType> impl;

        REQUIRE(impl.data() == nullptr);
        REQUIRE(impl.rank() == 0);
        REQUIRE(impl.size() == 0);
    }

    SECTION("Row major constructor and copy constructor.") {
        std::vector<std::remove_cv_t<TestType>> test_data{TestType{1.0}, TestType{2.0}, TestType{3.0}, TestType{4.0}};
        detail::TensorImpl<TestType>            impl(test_data.data(), {2, 2}, true);

        REQUIRE(impl.data() == test_data.data());
        REQUIRE(impl.rank() == 2);
        REQUIRE(impl.size() == 4);
        REQUIRE(impl.dim(0) == 2);
        REQUIRE(impl.dim(1) == 2);
        REQUIRE_THROWS(impl.dim(2));
        REQUIRE(impl.stride(0) == 2);
        REQUIRE(impl.stride(1) == 1);

        detail::TensorImpl<TestType> impl_copy = impl;

        REQUIRE(impl.data() == test_data.data());
        REQUIRE(impl.rank() == 2);
        REQUIRE(impl.size() == 4);
        REQUIRE(impl.dim(0) == 2);
        REQUIRE(impl.dim(1) == 2);
        REQUIRE_THROWS(impl.dim(2));
        REQUIRE(impl.stride(0) == 2);
        REQUIRE(impl.stride(1) == 1);

        REQUIRE(impl_copy.data() == test_data.data());
        REQUIRE(impl_copy.rank() == 2);
        REQUIRE(impl_copy.size() == 4);
        REQUIRE(impl_copy.dim(0) == 2);
        REQUIRE(impl_copy.dim(1) == 2);
        REQUIRE_THROWS(impl_copy.dim(2));
        REQUIRE(impl_copy.stride(0) == 2);
        REQUIRE(impl_copy.stride(1) == 1);
    }

    SECTION("Column major constructor and move constructor.") {
        std::vector<std::remove_cv_t<TestType>> test_data{TestType{1.0}, TestType{2.0}, TestType{3.0}, TestType{4.0}};
        detail::TensorImpl<TestType>            impl(test_data.data(), {2, 2}, false);

        REQUIRE(impl.data() == test_data.data());
        REQUIRE(impl.rank() == 2);
        REQUIRE(impl.size() == 4);
        REQUIRE(impl.dim(0) == 2);
        REQUIRE(impl.dim(1) == 2);
        REQUIRE_THROWS(impl.dim(2));
        REQUIRE(impl.stride(0) == 1);
        REQUIRE(impl.stride(1) == 2);

        detail::TensorImpl<TestType> impl_copy(std::move(impl));

        REQUIRE(impl.data() == nullptr);
        REQUIRE(impl.rank() == 0);
        REQUIRE(impl.size() == 0);
        REQUIRE(impl.dim(0) == 0);
        REQUIRE(impl.stride(0) == 0);

        REQUIRE(impl_copy.data() == test_data.data());
        REQUIRE(impl_copy.rank() == 2);
        REQUIRE(impl_copy.size() == 4);
        REQUIRE(impl_copy.dim(0) == 2);
        REQUIRE(impl_copy.dim(1) == 2);
        REQUIRE_THROWS(impl_copy.dim(2));
        REQUIRE(impl_copy.stride(0) == 1);
        REQUIRE(impl_copy.stride(1) == 2);
    }

    SECTION("Strides specified and assignments.") {
        std::vector<std::remove_cv_t<TestType>> test_data{TestType{1.0}, TestType{2.0}, TestType{3.0}, TestType{4.0}};
        detail::TensorImpl<TestType>            impl(test_data.data(), {2, 2}, {2, 1});

        REQUIRE(impl.data() == test_data.data());
        REQUIRE(impl.rank() == 2);
        REQUIRE(impl.size() == 4);
        REQUIRE(impl.dim(0) == 2);
        REQUIRE(impl.dim(1) == 2);
        REQUIRE_THROWS(impl.dim(2));
        REQUIRE(impl.stride(0) == 2);
        REQUIRE(impl.stride(1) == 1);

        detail::TensorImpl<TestType> impl_copy;

        impl_copy = impl;

        REQUIRE(impl.data() == test_data.data());
        REQUIRE(impl.rank() == 2);
        REQUIRE(impl.size() == 4);
        REQUIRE(impl.dim(0) == 2);
        REQUIRE(impl.dim(1) == 2);
        REQUIRE_THROWS(impl.dim(2));
        REQUIRE(impl.stride(0) == 2);
        REQUIRE(impl.stride(1) == 1);

        REQUIRE(impl_copy.data() == test_data.data());
        REQUIRE(impl_copy.rank() == 2);
        REQUIRE(impl_copy.size() == 4);
        REQUIRE(impl_copy.dim(0) == 2);
        REQUIRE(impl_copy.dim(1) == 2);
        REQUIRE_THROWS(impl_copy.dim(2));
        REQUIRE(impl_copy.stride(0) == 2);
        REQUIRE(impl_copy.stride(1) == 1);

        impl_copy = std::move(impl);

        REQUIRE(impl.data() == nullptr);
        REQUIRE(impl.rank() == 0);
        REQUIRE(impl.size() == 0);
        REQUIRE(impl.dim(0) == 0);
        REQUIRE(impl.stride(0) == 0);

        REQUIRE(impl_copy.data() == test_data.data());
        REQUIRE(impl_copy.rank() == 2);
        REQUIRE(impl_copy.size() == 4);
        REQUIRE(impl_copy.dim(0) == 2);
        REQUIRE(impl_copy.dim(1) == 2);
        REQUIRE_THROWS(impl_copy.dim(2));
        REQUIRE(impl_copy.stride(0) == 2);
        REQUIRE(impl_copy.stride(1) == 1);
    }
}

TEMPLATE_TEST_CASE("TensorImpl view creation", "[tensor]", float, double, std::complex<float>, std::complex<double>, int) {
    std::vector<std::remove_cv_t<TestType>> test_data(27);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                test_data[i * 9 + j * 3 + k] = i * 9 + j * 3 + k;
            }
        }
    }

    SECTION("Row major") {
        detail::TensorImpl<TestType> base(test_data.data(), {3, 3, 3}, true);

        auto view = base.subscript(Range{0, 2}, 1, All);

        REQUIRE(view.data() == base.data() + base.stride(1));
        REQUIRE(view.rank() == 2);
        REQUIRE(view.stride(0) == 9);
        REQUIRE(view.stride(1) == 1);
        REQUIRE(view.dim(0) == 2);
        REQUIRE(view.dim(1) == 3);
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> base(test_data.data(), {3, 3, 3}, false);

        auto view = base.subscript(Range{0, 2}, 1, All);

        REQUIRE(view.data() == base.data() + base.stride(1));
        REQUIRE(view.rank() == 2);
        REQUIRE(view.stride(0) == 1);
        REQUIRE(view.stride(1) == 9);
        REQUIRE(view.dim(0) == 2);
        REQUIRE(view.dim(1) == 3);
    }
}

TEST_CASE("TensorImpl layout inferred from the axes stepped along", "[tensor]") {
    // An extent-1 axis is never stepped along, so its stride is whatever a view inherited. The
    // layout used to be read off the first and last strides regardless, so (6, 1, 2) over extents
    // (1, 2, 3), a column-major block, was called row major.
    std::vector<double> data(64);
    auto const          layout = [&](std::vector<size_t> dims, std::vector<size_t> strides) {
        return detail::TensorImpl<double>(data.data(), std::move(dims), std::move(strides)).is_row_major();
    };

    CHECK_FALSE(layout({1, 2, 3}, {6, 1, 2}));
    CHECK_FALSE(layout({2, 3, 1}, {1, 2, 1}));
    CHECK(layout({3, 2, 1}, {2, 1, 6}));
    CHECK(layout({1, 3, 2}, {1, 2, 1}));

    // With one axis stepped along, a unit stride makes it the minor axis and any other the major.
    CHECK(layout({1, 3}, {2, 1}));
    CHECK_FALSE(layout({3, 1}, {1, 2}));
    CHECK_FALSE(layout({1, 3}, {1, 5}));
    CHECK(layout({3, 1}, {4, 1}));

    // Ordinary layouts are unchanged.
    CHECK_FALSE(layout({2, 3}, {1, 2}));
    CHECK(layout({2, 3}, {3, 1}));

    // The layout views agree with the flag. They compared the first and last strides on their own,
    // so the row-major vector (1, 1, 6) with strides (1, 1, 1) came back from to_column_major
    // unreversed and still row major, and permute read its axes from the wrong end.
    detail::TensorImpl<double> const vec(data.data(), std::vector<size_t>{1, 1, 6}, std::vector<size_t>{1, 1, 1});
    REQUIRE(vec.is_row_major());
    auto const col = vec.to_column_major();
    CHECK(col.is_column_major());
    CHECK(col.dim(0) == 6);
    CHECK(vec.to_row_major().dim(2) == 6);
}

TEST_CASE("TensorImpl leading dimension of a matrix with an extent-1 axis", "[tensor]") {
    // A diagonal folded out of an (i, i, j) operand with i of extent 1 is the (1, 3) matrix with
    // strides (2, 1). Its major axis is never stepped along, so that 2 says nothing, but get_lda
    // handed it to BLAS as the leading dimension, below the minor extent of 3, and gemv rejected
    // the call: the string einsum "i <- j ; iij" failed this way.
    std::vector<double> data(6);

    SECTION("row-major, major axis of extent 1") {
        detail::TensorImpl<double> const impl(data.data(), std::vector<size_t>{1, 3}, std::vector<size_t>{2, 1});
        REQUIRE(impl.is_row_major());
        CHECK(impl.get_lda() == 3);
        size_t lda = 0;
        REQUIRE(impl.is_gemmable(&lda));
        CHECK(lda == 3);
    }

    SECTION("column-major, major axis of extent 1") {
        detail::TensorImpl<double> const impl(data.data(), std::vector<size_t>{3, 1}, std::vector<size_t>{1, 2});
        REQUIRE(impl.is_column_major());
        CHECK(impl.get_lda() == 3);
    }

    SECTION("a minor axis of extent 1 keeps its stride, the step between the elements") {
        detail::TensorImpl<double> const impl(data.data(), std::vector<size_t>{1, 3}, std::vector<size_t>{1, 2});
        REQUIRE(impl.is_column_major());
        CHECK(impl.get_lda() == 2);
    }

    SECTION("a major axis stepped with a stride other than one") {
        // (2, 1) with strides (4, 1) is two elements four apart: row major, leading dimension 4,
        // whatever the extent-1 axis's stride claims.
        detail::TensorImpl<double> const impl(data.data(), std::vector<size_t>{2, 1}, std::vector<size_t>{4, 1});
        REQUIRE(impl.is_row_major());
        CHECK(impl.get_lda() == 4);
    }

    SECTION("an ordinary matrix") {
        detail::TensorImpl<double> const impl(data.data(), std::vector<size_t>{2, 3}, std::vector<size_t>{1, 2});
        CHECK(impl.get_lda() == 2);
    }
}
