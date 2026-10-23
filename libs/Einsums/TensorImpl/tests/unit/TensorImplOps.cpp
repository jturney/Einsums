//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>
#include <Einsums/Testing/Layouts.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Full-Full", "[tensor]", float, double, int) {
    BufferVector<std::remove_cv_t<TestType>> input_data(27), output_data(27);

    for (int i = 0; i < 27; i++) {
        input_data[i]  = i + 1;
        output_data[i] = 11 * (i + 1);
    }

    SECTION("Row-Row") {
        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, true);

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, true);

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(12 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(10 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Row-Column") {
        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, false);

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, true);

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) + (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) - (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) * (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(
                                         ((TestType)(11 * (i * 9 + 3 * j + k + 1))) / ((TestType)(k * 9 + 3 * j + i + 1)), 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Column-Row") {
        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, true);

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, false);

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) + (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) - (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) * (k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(
                                         ((TestType)(11 * (i * 9 + j * 3 + k + 1))) / ((TestType)(k * 9 + 3 * j + i + 1)), 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher((k * 9 + 3 * j + i + 1), 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Column-Column") {
        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, false);

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, false);

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(12 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(10 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1) * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher((i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("View-View", "[tensor]", float, double, int) {
    BufferVector<std::remove_cv_t<TestType>> input_data(64), output_data(64);

    for (int i = 0; i < 64; i++) {
        input_data[i]  = i + 1;
        output_data[i] = 11 * (i + 1);
    }

    SECTION("Row Major") {

        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, {16, 4, 1});

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, {16, 4, 1});

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(12 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(10 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 16 + 4 * j + k + 1) * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Column Major") {

        detail::TensorImpl<TestType> const input(input_data.data(), {3, 3, 3}, {1, 4, 16});

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, {1, 4, 16});

        SECTION("Add") {
            detail::add_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(12 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(10 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(11 * (i * 16 + 4 * j + k + 1) * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(input, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher((i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Full-Scalar", "[tensor]", float, double, int) {
    BufferVector<TestType> output_data(27);
    TestType               scalar = 11;

    for (int i = 0; i < 27; i++) {
        output_data[i] = (i + 1);
    }

    SECTION("Row") {
        detail::TensorImpl<TestType> output(output_data.data(), {3, 3, 3}, true);

        SECTION("Add") {
            detail::add_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((i * 9 + 3 * j + k + 1) + 11, 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((i * 9 + 3 * j + k + 1) - 11, 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(((TestType)(i * 9 + 3 * j + k + 1)) / ((TestType)11), 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Column-Row") {
        detail::TensorImpl<TestType> output(output_data.data(), {3, 3, 3}, false);

        SECTION("Add") {
            detail::add_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11 + (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher((i * 9 + 3 * j + k + 1) - 11, 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11 * (i * 9 + 3 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(((TestType)(i * 9 + j * 3 + k + 1)) / (TestType)11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("View-Scalar", "[tensor]", float, double, int) {
    BufferVector<TestType> output_data(64);
    TestType               scalar = 11;

    for (int i = 0; i < 64; i++) {
        output_data[i] = (i + 1);
    }

    SECTION("Row major") {

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, {16, 4, 1});

        SECTION("Add") {
            detail::add_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11 + (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher((i * 16 + 4 * j + k + 1) - 11, 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k),
                                     Catch::Matchers::WithinAbsMatcher(((TestType)(i * 16 + 4 * j + k + 1)) / (TestType)11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(i, j, k), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }
    }

    SECTION("Column major") {

        detail::TensorImpl<std::remove_cv_t<TestType>> output(output_data.data(), {3, 3, 3}, {1, 4, 16});

        SECTION("Add") {
            detail::add_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11 + (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Subtract") {
            detail::sub_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher((i * 16 + 4 * j + k + 1) - 11, 1e-6));
                    }
                }
            }
        }

        SECTION("Multiply") {
            detail::mult_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11 * (i * 16 + 4 * j + k + 1), 1e-6));
                    }
                }
            }
        }

        SECTION("Divide") {
            detail::div_assign(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i),
                                     Catch::Matchers::WithinAbsMatcher(((TestType)(i * 16 + 4 * j + k + 1)) / (TestType)11, 1e-6));
                    }
                }
            }
        }

        SECTION("Copy") {
            detail::copy_to(scalar, output);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        REQUIRE_THAT(output.subscript(k, j, i), Catch::Matchers::WithinAbsMatcher(11, 1e-6));
                    }
                }
            }
        }
    }
}

// A dense block copied into a WINDOW of a larger buffer. Every stride differs
// except the innermost, which the two share -- the case impl_copy answers with a
// run of copies rather than an element-by-element walk. What must hold is that
// each element lands at its own index and that nothing outside the window moves.
TEMPLATE_TEST_CASE("Block into a window", "[tensor]", float, double, int) {
    using T = std::remove_cv_t<TestType>;

    constexpr size_t D0 = 5, D1 = 6, D2 = 7; // buffer
    constexpr size_t B0 = 2, B1 = 3, B2 = 2; // block
    constexpr size_t O0 = 1, O1 = 2, O2 = 3; // where the block goes

    BufferVector<T> block_data(B0 * B1 * B2);
    for (size_t i = 0; i < block_data.size(); i++) {
        block_data[i] = static_cast<T>(i + 1);
    }

    SECTION("Column-major") {
        BufferVector<T>           buffer(D0 * D1 * D2, T{0});
        std::vector<size_t> const bstride{1, D0, D0 * D1};
        size_t const              base = O0 * bstride[0] + O1 * bstride[1] + O2 * bstride[2];

        detail::TensorImpl<T> const block(block_data.data(), {B0, B1, B2}, false);
        detail::TensorImpl<T>       window(buffer.data() + base, std::vector<size_t>{B0, B1, B2}, bstride);
        detail::copy_to(block, window);

        for (size_t k = 0; k < D2; k++) {
            for (size_t j = 0; j < D1; j++) {
                for (size_t i = 0; i < D0; i++) {
                    bool const inside = i >= O0 && i < O0 + B0 && j >= O1 && j < O1 + B1 && k >= O2 && k < O2 + B2;
                    T const    want   = inside ? block.subscript(i - O0, j - O1, k - O2) : T{0};
                    REQUIRE(buffer[i + j * D0 + k * D0 * D1] == want);
                }
            }
        }
    }

    SECTION("Row-major") {
        BufferVector<T>           buffer(D0 * D1 * D2, T{0});
        std::vector<size_t> const bstride{D1 * D2, D2, 1};
        size_t const              base = O0 * bstride[0] + O1 * bstride[1] + O2 * bstride[2];

        detail::TensorImpl<T> const block(block_data.data(), {B0, B1, B2}, true);
        detail::TensorImpl<T>       window(buffer.data() + base, std::vector<size_t>{B0, B1, B2}, bstride);
        detail::copy_to(block, window);

        for (size_t i = 0; i < D0; i++) {
            for (size_t j = 0; j < D1; j++) {
                for (size_t k = 0; k < D2; k++) {
                    bool const inside = i >= O0 && i < O0 + B0 && j >= O1 && j < O1 + B1 && k >= O2 && k < O2 + B2;
                    T const    want   = inside ? block.subscript(i - O0, j - O1, k - O2) : T{0};
                    REQUIRE(buffer[i * D1 * D2 + j * D2 + k] == want);
                }
            }
        }
    }

    // The two layouts disagree about which axis is innermost, so there is no
    // shared run to exploit and the element-by-element walk has to answer it.
    SECTION("Column-major into row-major") {
        BufferVector<T>             out_data(B0 * B1 * B2, T{0});
        detail::TensorImpl<T> const block(block_data.data(), {B0, B1, B2}, false);
        detail::TensorImpl<T>       out(out_data.data(), {B0, B1, B2}, true);
        detail::copy_to(block, out);

        for (size_t i = 0; i < B0; i++) {
            for (size_t j = 0; j < B1; j++) {
                for (size_t k = 0; k < B2; k++) {
                    REQUIRE(out.subscript(i, j, k) == block.subscript(i, j, k));
                }
            }
        }
    }

    // Strides that differ by a constant factor: the whole walk is one strided
    // vector on both sides, so no axis is left to loop over.
    SECTION("Into every other element") {
        BufferVector<T>             out_data(2 * B0 * B1 * B2, T{0});
        detail::TensorImpl<T> const block(block_data.data(), {B0, B1, B2}, false);
        detail::TensorImpl<T>       out(out_data.data(), std::vector<size_t>{B0, B1, B2}, std::vector<size_t>{2, 2 * B0, 2 * B0 * B1});
        detail::copy_to(block, out);

        for (size_t i = 0; i < out_data.size(); i++) {
            REQUIRE(out_data[i] == (i % 2 == 0 ? block_data[i / 2] : T{0}));
        }
    }
}

namespace {

using einsums::testing::for_each_index;
using einsums::testing::for_each_layout_combination;
using einsums::testing::Layout;
using einsums::testing::layout_close;
using einsums::testing::layout_value;
using einsums::testing::logical_index;
using einsums::testing::shape_name;

/// Applies @p op(in, out) on every pair of layouts of every shape and compares each element of out
/// against @p ref(out before, in), and every gap of out against the sentinel.
template <typename Out, typename In, typename Op, typename Ref>
void check_every_layout_pair(Op op, Ref ref) {
    for_each_layout_combination<Out, In>([&](std::vector<size_t> const &dims, Layout<Out> out, Layout<In> in) {
        INFO("shape " << shape_name(dims) << ", out " << out.name << ", in " << in.name);
        auto       out_impl = out.impl();
        auto const in_impl  = in.impl();
        op(in_impl, out_impl);

        size_t      wrong = 0;
        std::string first;
        for_each_index(dims, [&](std::vector<size_t> const &index) {
            size_t const k    = logical_index(dims, index);
            Out const    want = ref(layout_value<Out>(1, k), layout_value<In>(2, k));
            if (!layout_close(out.at(index), want) && wrong++ == 0) {
                first = fmt::format("element {} is {}, want {}", k, out.at(index), want);
            }
        });
        INFO(first);
        CHECK(wrong == 0);
        CHECK(out.gaps_untouched());
    });
}

/// As @ref check_every_layout_pair for an operation on one tensor in place.
template <typename T, typename Op, typename Ref>
void check_every_layout(Op op, Ref ref) {
    check_every_layout_pair<T, T>([&](auto const &, auto &out) { op(out); }, [&](T out, T) { return ref(out); });
}

template <typename T>
T layout_scalar(double re, double im) {
    if constexpr (IsComplexV<T>) {
        return T(static_cast<RemoveComplexT<T>>(re), static_cast<RemoveComplexT<T>>(im));
    } else {
        return static_cast<T>(re);
    }
}

} // namespace

// Each elementwise operation picks one of three paths by layout: one vector call when both operands
// are a single vector with the same strides, loops over vector calls when they share strides but are
// not one vector, and an element walk otherwise. Every layout of the output meets every layout of the
// input here, on shapes with extent-1 axes, rank 0 and an empty extent, and on a type BLAS does not take.
TEMPLATE_TEST_CASE("elementwise operations on every pair of layouts", "[tensor][layouts]", float, double, std::complex<float>,
                   std::complex<double>, int) {
    using T       = TestType;
    T const alpha = std::is_integral_v<T> ? T(3) : layout_scalar<T>(1.5, -0.5);

    SECTION("axpy") {
        check_every_layout_pair<T, T>([&](auto const &in, auto &out) { detail::impl_axpy(alpha, in, out); },
                                      [&](T out, T in) { return out + alpha * in; });
    }
    SECTION("mult") {
        check_every_layout_pair<T, T>([](auto const &in, auto &out) { detail::impl_mult(in, out); }, [](T out, T in) { return out * in; });
    }
    SECTION("div") {
        check_every_layout_pair<T, T>([](auto const &in, auto &out) { detail::impl_div(in, out); }, [](T out, T in) { return out / in; });
    }
    SECTION("copy") {
        check_every_layout_pair<T, T>([](auto const &in, auto &out) { detail::impl_copy(in, out); }, [](T, T in) { return in; });
    }
    SECTION("scal") {
        check_every_layout<T>([&](auto &out) { detail::impl_scal(alpha, out); }, [&](T out) { return alpha * out; });
    }
    SECTION("scal by zero") {
        check_every_layout<T>([](auto &out) { detail::impl_scal(T(0), out); }, [](T) { return T(0); });
    }
    SECTION("div_scalar") {
        check_every_layout<T>([&](auto &out) { detail::impl_div_scalar(alpha, out); }, [&](T out) { return out / alpha; });
    }
    SECTION("scalar_add") {
        check_every_layout<T>([&](auto &out) { detail::impl_scalar_add(alpha, out); }, [&](T out) { return out + alpha; });
    }
    SECTION("scalar_copy") {
        check_every_layout<T>([&](auto &out) { detail::impl_scalar_copy(alpha, out); }, [&](T) { return alpha; });
    }
    SECTION("conj") {
        check_every_layout<T>([](auto &out) { detail::impl_conj(out); },
                              [](T out) {
                                  if constexpr (IsComplexV<T>) {
                                      return std::conj(out);
                                  } else {
                                      return out;
                                  }
                              });
    }
}

TEMPLATE_TEST_CASE("complex-to-real operations on every pair of layouts", "[tensor][layouts]", float, double) {
    using R = TestType;
    using C = std::complex<R>;

    SECTION("real") {
        check_every_layout_pair<R, C>([](auto const &in, auto &out) { detail::impl_real(in, out); }, [](R, C in) { return std::real(in); });
    }
    SECTION("imag") {
        check_every_layout_pair<R, C>([](auto const &in, auto &out) { detail::impl_imag(in, out); }, [](R, C in) { return std::imag(in); });
    }
    SECTION("abs") {
        check_every_layout_pair<R, C>([](auto const &in, auto &out) { detail::impl_abs(in, out); }, [](R, C in) { return std::abs(in); });
    }
}

// An input of another type converts on the way, so these never reach a same-type vector call.
TEST_CASE("mixed-type elementwise operations on every pair of layouts", "[tensor][layouts]") {
    SECTION("copy float into double") {
        check_every_layout_pair<double, float>([](auto const &in, auto &out) { detail::impl_copy(in, out); },
                                               [](double, float in) { return double(in); });
    }
    SECTION("axpy double into complex double") {
        using Z = std::complex<double>;
        check_every_layout_pair<Z, double>([](auto const &in, auto &out) { detail::impl_axpy(2.0, in, out); },
                                           [](Z out, double in) { return out + 2.0 * in; });
    }
    SECTION("mult by int into double") {
        check_every_layout_pair<double, int>([](auto const &in, auto &out) { detail::impl_mult(in, out); },
                                             [](double out, int in) { return out * in; });
    }
}
