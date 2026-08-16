//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("element transform", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    config::set(option::BufferSize, std::string("1GB"));

    SECTION("tensor") {
        Tensor A     = create_random_tensor<TestType>("A", 32, 32, 32, 32);
        Tensor Acopy = A;

        if constexpr (IsComplexV<TestType>) {
            element_transform(&A, [](TestType val) -> TestType { return TestType{1.0, 1.0} / val; });
        } else {
            element_transform(&A, [](TestType val) -> TestType { return TestType{1.0} / val; });
        }

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        if constexpr (IsComplexV<TestType>) {
                            REQUIRE_THAT(A(w, x, y, z), CheckWithinRel(TestType{1.0, 1.0} / Acopy(w, x, y, z), 0.001));
                        } else {
                            REQUIRE_THAT(A(w, x, y, z), CheckWithinRel(TestType{1.0} / Acopy(w, x, y, z), 0.001));
                            // REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(1.0 / Acopy(w, x, y, z), 0.001));
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("element", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    int _i = 5;

    SECTION("1") {
        Tensor A     = create_random_tensor<TestType>("A", _i, _i, _i, _i);
        Tensor Acopy = A;

        Tensor B = create_random_tensor<TestType>("B", _i, _i, _i, _i);

        element([](TestType const &Aval, TestType const &Bval) { return Aval + Bval; }, &A, B);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _i; x++) {
                for (int y = 0; y < _i; y++) {
                    for (int z = 0; z < _i; z++) {
                        REQUIRE_THAT(A(w, x, y, z), CheckWithinRel(Acopy(w, x, y, z) + B(w, x, y, z), 0.001));
                        // REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z), 0.001));
                    }
                }
            }
        }
    }

    SECTION("2") {
        Tensor A     = create_random_tensor<TestType>("A", _i, _i, _i, _i);
        Tensor Acopy = A;

        Tensor B = create_random_tensor<TestType>("B", _i, _i, _i, _i);
        Tensor C = create_random_tensor<TestType>("C", _i, _i, _i, _i);

        element([](TestType const &Aval, TestType const &Bval, TestType const &Cval) { return Aval + Bval + Cval; }, &A, B, C);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _i; x++) {
                for (int y = 0; y < _i; y++) {
                    for (int z = 0; z < _i; z++) {
                        REQUIRE_THAT(A(w, x, y, z), CheckWithinRel(Acopy(w, x, y, z) + B(w, x, y, z) + C(w, x, y, z), 0.001));
                        // REQUIRE_THAT(A(w, x, y, z), Catch::Matchers::WithinAbs(Acopy(w, x, y, z) + B(w, x, y, z) + C(w, x, y, z),
                        // 0.001));
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("einsum element", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    int const _i{5}, _j{5};

    SECTION("1") {
        Tensor C  = create_tensor<TestType>("C", _i, _j);
        Tensor C0 = create_tensor<TestType>("C", _i, _j);

        Tensor B = create_random_tensor<TestType>("B", _i, _j);
        Tensor A = create_random_tensor<TestType>("A", _i, _j);

        element([](TestType const & /*Cval*/, TestType const &Aval, TestType const &Bval) { return Aval * Bval; }, &C0, A, B);

        einsum(Indices{i, j}, &C, Indices{i, j}, A, Indices{i, j}, B, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::DIRECT);

        // std::stringstream stream;

        // fprintln(stream, C);

        // INFO(stream.str());

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                CHECK_THAT(C(w, x), CheckWithinRel(C0(w, x), 1.0e-5));
                // REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(C0(w, x), 1.0e-5));
            }
        }
    }

    SECTION("2") {
        Tensor C          = create_random_tensor<TestType>("C", _i, _j);
        Tensor C0         = C;
        Tensor testresult = create_zero_tensor<TestType>("result", _i, _j);

        Tensor A = create_random_tensor<TestType>("A", _i, _j);

        element([](TestType const &Cval, TestType const &Aval) { return Cval * Aval; }, &C, A);

        einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::DIRECT);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                CHECK_THAT(C(w, x), CheckWithinRel(testresult(w, x), 1.0e-5));
                // REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
            }
        }
    }

    SECTION("3") {
        Tensor parentC  = create_random_tensor<TestType>("parentC", _i, _i, _i, _j);
        Tensor parentC0 = parentC;
        Tensor parentA  = create_random_tensor<TestType>("parentA", _i, _i, _i, _j);

        auto   C          = parentC(3, All, All, 4);
        auto   C0         = parentC0(3, All, All, 4);
        Tensor testresult = create_zero_tensor<TestType>("result", _i, _j);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                testresult(w, x) = C(w, x);
            }
        }

        auto A = parentA(1, 2, All, All);

        element([](TestType const &Cval, TestType const &Aval) { return Cval * Aval; }, &C, A);

        einsum(Indices{i, j}, &testresult, Indices{i, j}, C0, Indices{i, j}, A, &alg_choice);
        REQUIRE(alg_choice == tensor_algebra::detail::DIRECT);

        for (int w = 0; w < _i; w++) {
            for (int x = 0; x < _j; x++) {
                REQUIRE_THAT(C(w, x), CheckWithinRel(testresult(w, x), 1.0e-5));
                // REQUIRE_THAT(C(w, x), Catch::Matchers::WithinAbs(testresult(w, x), 1.0e-5));
            }
        }
    }
}
