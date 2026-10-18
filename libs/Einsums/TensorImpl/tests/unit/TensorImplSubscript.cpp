//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Tensor impl subscripting.", "[tensor]", float, double, int) {
    std::vector<TestType> vector_data(27);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                vector_data[i * 9 + j * 3 + k] = i * 9 + j * 3 + k;
            }
        }
    }

    SECTION("Row major") {
        detail::TensorImpl<TestType> impl(vector_data.data(), {3, 3, 3}, true);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(impl.subscript(i, j, k), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript({i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript(BufferVector<int>{i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(i, j, k), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check({i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                }
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            impl.subscript_no_check(1, 1, 1) = TestType{10};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            impl.subscript_no_check({1, 1, 1}) = TestType{11};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            impl.subscript_no_check(BufferVector<int>{1, 1, 1}) = TestType{12};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            impl.subscript(1, 1, 1) = TestType{10};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            impl.subscript({1, 1, 1}) = TestType{11};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            impl.subscript(BufferVector<int>{1, 1, 1}) = TestType{12};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
        }
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> impl(vector_data.data(), {3, 3, 3}, false);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(impl.subscript(k, j, i), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript({k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript(BufferVector<int>{k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(k, j, i), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check({k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                }
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            impl.subscript_no_check(1, 1, 1) = TestType{10};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            impl.subscript_no_check({1, 1, 1}) = TestType{11};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            impl.subscript_no_check(BufferVector<int>{1, 1, 1}) = TestType{12};
            REQUIRE_THAT(impl.subscript_no_check(1, 1, 1), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            impl.subscript(1, 1, 1) = TestType{10};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
            impl.subscript({1, 1, 1}) = TestType{11};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
            impl.subscript(BufferVector<int>{1, 1, 1}) = TestType{12};
            REQUIRE_THAT(impl.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript({1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
            REQUIRE_THAT(impl.subscript(BufferVector<int>{1, 1, 1}), Catch::Matchers::WithinAbsMatcher(12, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("Tensor impl const subscripting.", "[tensor]", float, double, int) {
    std::vector<TestType> vector_data(27);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                vector_data[i * 9 + j * 3 + k] = i * 9 + j * 3 + k;
            }
        }
    }

    SECTION("Row major") {
        detail::TensorImpl<TestType> const impl(vector_data.data(), {3, 3, 3}, true);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(impl.subscript(i, j, k), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript({i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript(BufferVector<int>{i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(i, j, k), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check({i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{i, j, k}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                }
            }
        }
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> const impl(vector_data.data(), {3, 3, 3}, false);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(impl.subscript(k, j, i), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript({k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript(BufferVector<int>{k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(k, j, i), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check({k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                    REQUIRE_THAT(impl.subscript_no_check(BufferVector<int>{k, j, i}), Catch::Matchers::WithinAbs(9 * i + 3 * j + k, 1e-6));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("TensorImpl view subscript", "[tensor]", float, double, int) {
    std::vector<TestType> test_data(27);

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

        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 3; k++) {
                REQUIRE_THAT(view.subscript(i, k), Catch::Matchers::WithinAbsMatcher(9 * i + 3 + k, 1e-6));
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            view.subscript(1, 1) = TestType{10};

            REQUIRE_THAT(base.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
        }
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> base(test_data.data(), {3, 3, 3}, false);

        auto view = base.subscript(Range{0, 2}, 1, All);

        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 2; k++) {
                REQUIRE_THAT(view.subscript(k, i), Catch::Matchers::WithinAbsMatcher(9 * i + 3 + k, 1e-6));
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            view.subscript(1, 1) = TestType{10};

            REQUIRE_THAT(base.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("TensorImpl tied view subscript", "[tensor]", float, double, int) {
    std::vector<TestType> test_data(27);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                test_data[i * 9 + j * 3 + k] = i * 9 + j * 3 + k;
            }
        }
    }

    SECTION("Row major") {
        detail::TensorImpl<TestType> base(test_data.data(), {3, 3, 3}, true);

        auto view = base.tie_indices(0, 2);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(view.subscript(i, j), Catch::Matchers::WithinAbsMatcher(9 * i + 3 * j + i, 1e-6));
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            view.subscript(1, 1) = TestType{10};

            REQUIRE_THAT(base.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
        }
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> base(test_data.data(), {3, 3, 3}, false);

        auto view = base.tie_indices(0, 2);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(view.subscript(j, i), Catch::Matchers::WithinAbsMatcher(9 * i + 3 * j + i, 1e-6));
            }
        }

        if constexpr (!std::is_const_v<TestType>) {
            view.subscript(1, 1) = TestType{10};

            REQUIRE_THAT(base.subscript(1, 1, 1), Catch::Matchers::WithinAbsMatcher(10, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("TensorImpl const tied view subscript", "[tensor]", float, double, int) {
    std::vector<TestType> test_data(27);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                test_data[i * 9 + j * 3 + k] = i * 9 + j * 3 + k;
            }
        }
    }

    SECTION("Row major") {
        detail::TensorImpl<TestType> const base(test_data.data(), {3, 3, 3}, true);

        auto view = base.tie_indices(0, 2);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(view.subscript(i, j), Catch::Matchers::WithinAbsMatcher(9 * i + 3 * j + i, 1e-6));
            }
        }
    }

    SECTION("Column major") {
        detail::TensorImpl<TestType> const base(test_data.data(), {3, 3, 3}, false);

        auto view = base.tie_indices(0, 2);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(view.subscript(j, i), Catch::Matchers::WithinAbsMatcher(9 * i + 3 * j + i, 1e-6));
            }
        }
    }
}
