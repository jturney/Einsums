//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/modules/tensor_utilities.hpp>
#include <einsums/testing.hpp>

#include <catch2/catch_all.hpp>

using namespace einsums;

template <typename T>
void test_syev() {
    auto A = create_tensor<T>("A", 3, 3);
    auto x = create_tensor<T>("x", 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((x.dim(0) == 3));

    A.vector_data() = vector_type<T>{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};

    einsums::linear_algebra::syev(&A, &x);

    CHECK_THAT(x(0), Catch::Matchers::WithinRel(-0.515729, 0.00001));
    CHECK_THAT(x(1), Catch::Matchers::WithinRel(+0.170915, 0.00001));
    CHECK_THAT(x(2), Catch::Matchers::WithinRel(+11.344814, 0.00001));
}

TEMPLATE_TEST_CASE("syev", "[linear-algebra]", float, double) {
    test_syev<TestType>();
}

TEMPLATE_TEST_CASE("definite and semidefinite", "[linear-algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("positive definite") {
        auto A = create_random_definite<TestType>("A", size, size);

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        for (int i = 0; i < size; i++) {
            REQUIRE(B(i) >= -WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
    }

    SECTION("negative definite") {
        auto A = create_random_definite<TestType>("A", size, size, TestType{-1.0});

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        for (int i = 0; i < size; i++) {
            REQUIRE(B(i) <= WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
    }

    SECTION("positive semi-definite") {
        auto A = create_random_semidefinite<TestType>("A", size, size);

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        int zeros = 0;
        for (int i = 0; i < size; i++) {
            if (WithinStrict(TestType(0.0), TestType{10000.0}).match(B(i))) {
                zeros++;
            }
            REQUIRE(B(i) >= -WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
        REQUIRE(zeros >= 1);
    }

    SECTION("negative semi-definite") {
        auto A = create_random_semidefinite<TestType>("A", size, size, TestType{-1.0});

        auto B = create_tensor<TestType>("B", size);

        syev(&A, &B);

        int zeros = 0;
        for (int i = 0; i < size; i++) {
            if (WithinStrict(TestType(0.0), TestType{10000.0}).match(B(i))) {
                zeros++;
            }
            REQUIRE(B(i) <= WithinStrict(TestType(0.0), TestType{10000.0}).get_error());
        }
        REQUIRE(zeros >= 1);
    }
}
