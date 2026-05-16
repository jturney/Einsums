//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
