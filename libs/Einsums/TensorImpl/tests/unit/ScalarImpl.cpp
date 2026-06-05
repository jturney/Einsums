//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Scalars", "[tensor]", float, double, std::complex<float>, std::complex<double>, int) {
    std::remove_cv_t<TestType>   value = 11.0;
    detail::TensorImpl<TestType> tensor(&value, {}, true);

    SECTION("Getters and setters") {
        REQUIRE(tensor.data() == &value);
        REQUIRE(tensor.dim(0) == 1);
        REQUIRE(tensor.stride(0) == 0);
        REQUIRE(tensor.rank() == 0);
    }

    SECTION("Subscripting") {
        REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(std::real(value), 1e-6));
        REQUIRE_THAT(std::real(tensor.subscript(BufferVector<size_t>{})), Catch::Matchers::WithinAbs(std::real(value), 1e-6));
        REQUIRE_THAT(std::real(tensor.subscript_no_check()), Catch::Matchers::WithinAbs(std::real(value), 1e-6));
        REQUIRE_THAT(std::real(tensor.subscript_no_check(BufferVector<size_t>{})), Catch::Matchers::WithinAbs(std::real(value), 1e-6));
    }

    if constexpr (!std::is_const_v<TestType>) {

        SECTION("Assignment") {
            tensor.subscript() = 1.0;

            REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(1, 1e-6));
        }

        SECTION("Operations") {
            std::remove_cv_t<TestType> const value2 = 2.0;

            SECTION("Add") {
                detail::add_assign(value2, tensor);

                REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(13, 1e-6));
            }

            SECTION("Subtract") {
                detail::sub_assign(value2, tensor);

                REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(9, 1e-6));
            }

            SECTION("Multiply") {
                detail::mult_assign(value2, tensor);

                REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(22, 1e-6));
            }

            SECTION("Divide") {
                detail::div_assign(value2, tensor);

                REQUIRE_THAT(std::real(tensor.subscript()),
                             Catch::Matchers::WithinAbs(std::real(static_cast<TestType>(11) / static_cast<TestType>(2)), 1e-6));
            }

            SECTION("Copy") {
                detail::copy_to(value2, tensor);

                REQUIRE_THAT(std::real(tensor.subscript()), Catch::Matchers::WithinAbs(2, 1e-6));
            }
        }
    }
}