//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("direct product", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    SECTION("Rank 1 tensors") {
        constexpr int size = 21;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C = create_tensor<TestType>("C", size);
        C.zero();
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        if constexpr (IsComplexV<TestType>) {
            for (int i = 0; i < size; i++) {
                C_test(i) = TestType{1.0, 1.0} * A(i) * B(i);
            }
            direct_product(TestType{1.0, 1.0}, A, B, TestType{0.0}, &C);
        } else {

            for (int i = 0; i < size; i++) {
                C_test(i) = A(i) * B(i);
            }
            direct_product(TestType{1.0}, A, B, TestType{0.0}, &C);
        }

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i)));
            }
        }
    }

    SECTION("Rank 1 tensors small") {
        constexpr int size = 4;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C      = create_tensor<TestType>("C", size);
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i) * B(i);
        }

        direct_product(TestType{1.0}, A, B, TestType{0.0}, &C);

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i)));
            };
        }
    }

    SECTION("Rank 1 tensor views") {
        constexpr int size = 21;

        Tensor<TestType, 2> A = create_random_tensor<TestType>(true, "A", size, 1);
        Tensor<TestType, 2> B = create_random_tensor<TestType>(true, "B", size, 2);

        Tensor<TestType, 2> C(true, "C", size, 3);
        Tensor<TestType, 1> C_test("C", size);

        auto A_view = A(All, 0);
        auto B_view = B(All, 0);
        auto C_view = C(All, 0);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i, 0) * B(i, 0);
        }

        direct_product(TestType{1.0}, A_view, B_view, TestType{0.0}, &C_view);

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i, 0))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i, 0))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i, 0)));
            }
        }
    }

    SECTION("Rank 1 tensor views small") {
        constexpr int size = 4;

        Tensor<TestType, 2> A = create_random_tensor<TestType>(true, "A", size, 2);
        Tensor<TestType, 2> B = create_random_tensor<TestType>(true, "B", size, 2);

        Tensor<TestType, 2> C(true, "C", size, 2);
        Tensor<TestType, 1> C_test("C", size);

        auto A_view = A(All, 0);
        auto B_view = B(All, 0);
        auto C_view = C(All, 0);

        for (int i = 0; i < size; i++) {
            C_test(i) = A(i, 0) * B(i, 0);
        }

        direct_product(TestType{1.0}, A_view, B_view, TestType{0.0}, &C_view);

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinRel(std::real(C(i, 0))));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinRel(std::imag(C(i, 0))));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i, 0)));
            }
        }
    }

    // SECTION("Rank 2 tensors") {
    //     constexpr int size = 10;
    //     Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
    //     Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

    //     TestType test{0.0};

    //     for (int i = 0; i < size; i++) {
    //         for (int j = 0; j < size; j++) {
    //             test += A(i, j) * B(i, j);
    //         }
    //     }

    //     auto dot_res = dot(A, B);

    //     REQUIRE_THAT(dot_res, einsums::WithinStrict(test, TestType{100000.0}));
    // }

    // SECTION("Rank 2 tensor views") {
    //     Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
    //     Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

    //     for(int i = 0; i < size; i++) {
    //         auto A_view = A(AllT(), Range{i, i + 1});
    //         auto B_view = B(AllT(), Range{i, i + 1});

    //         TestType test{0.0};

    //         for(int j = 0; j < size; j++) {
    //             test += A(j, i) * B(i, j);
    //         }

    //         auto dot_res = dot(A_view, B_view);

    //         REQUIRE(std::abs(dot_res - test) < static_cast<TestType>(EINSUMS_ZERO));
    //     }
    // }
}

TEST_CASE("mixed dots", "[linear-algebra]") {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("Rank 1 tensors") {
        Tensor<float, 1>  A = create_random_tensor<float>("A", size);
        Tensor<double, 1> B = create_random_tensor<double>("B", size);

        double test{0.0};

        for (int i = 0; i < size; i++) {
            test += A(i) * B(i);
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::WithinStrict(test, double{100000.0}));
    }
}

TEMPLATE_TEST_CASE("Disk direct products", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    config::set(option::WorkBufferSize, std::string("1024"));

    SECTION("Rank 1 tensors") {
        constexpr int size = 2000;

        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        Tensor<TestType, 1> C = create_tensor<TestType>("C", size);
        C.zero();
        Tensor<TestType, 1> C_test = create_tensor<TestType>("C", size);

        DiskTensor<TestType, 1> A_disk{fmt::format("/dirprod/rank1/{}/A", type_name<TestType>()), size};
        DiskTensor<TestType, 1> B_disk{fmt::format("/dirprod/rank1/{}/B", type_name<TestType>()), size};
        DiskTensor<TestType, 1> C_disk{fmt::format("/dirprod/rank1/{}/C", type_name<TestType>()), size};

        A_disk(All).get() = A;
        B_disk(All).get() = B;

        if constexpr (IsComplexV<TestType>) {
            for (int i = 0; i < size; i++) {
                C_test(i) = TestType{1.0, 1.0} * A(i) * B(i);
            }
            direct_product(TestType{1.0, 1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);
        } else {

            for (int i = 0; i < size; i++) {
                C_test(i) = A(i) * B(i);
            }
            direct_product(TestType{1.0}, A_disk, B_disk, TestType{0.0}, &C_disk);
        }

        C = C_disk(All).get();

        for (int i = 0; i < size; i++) {
            if constexpr (IsComplexV<TestType>) {
                REQUIRE_THAT(std::real(C_test(i)), Catch::Matchers::WithinAbs(std::real(C(i)), (RemoveComplexT<TestType>)1e-6));
                REQUIRE_THAT(std::imag(C_test(i)), Catch::Matchers::WithinAbs(std::imag(C(i)), (RemoveComplexT<TestType>)1e-6));
            } else {
                REQUIRE_THAT(C_test(i), Catch::Matchers::WithinRel(C(i), (TestType)1.0e-6));
            }
        }
    }
}
