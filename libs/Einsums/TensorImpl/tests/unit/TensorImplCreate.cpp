//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("TensorImlp Creation", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
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
