//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("dot", "[linear-algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("Rank 1 tensors") {
        Tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        Tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            test += A(i) * B(i);
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }

    SECTION("Rank 1 tensor views") {
        Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        for (int i = 0; i < size; i++) {
            auto A_view = A(All, i);
            auto B_view = B(i, All);

            TestType test{0.0};

            for (int j = 0; j < size; j++) {
                test += A(j, i) * B(i, j);
            }

            auto dot_res = dot(A_view, B_view);

            REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
        }
    }

    SECTION("Rank 2 tensors") {
        Tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        Tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                test += A(i, j) * B(i, j);
            }
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }

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

        REQUIRE_THAT(dot_res, einsums::CheckWithinRel(test));
    }
}

TEMPLATE_TEST_CASE("Disk dot", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    config::set(option::WorkBufferSize, std::string("1024"));

    SECTION("Rank 1 Tensors") {
        constexpr int size = 2000;

        auto A = create_random_tensor<TestType>("A", size);
        auto B = create_random_tensor<TestType>("A", size);

        DiskTensor<TestType, 1> A_disk{fmt::format("/dot/rank1/{}/A", type_name<TestType>()), size},
            B_disk{fmt::format("/dot/rank1/{}/B", type_name<TestType>()), size};

        A_disk(All) = A;
        B_disk(All) = B;

        auto expected = einsums::linear_algebra::dot(A, B);
        auto got      = einsums::linear_algebra::dot(A_disk, B_disk);

        REQUIRE_THAT(got, einsums::CheckWithinRel(expected));
    }

    SECTION("Rank 2 Tensors") {
        constexpr size_t size1 = 100, size2 = 51;
        auto             A = create_random_tensor<TestType>("A", size1, size2);
        auto             B = create_random_tensor<TestType>("B", size1, size2);

        DiskTensor<TestType, 2> A_disk{fmt::format("/dot/rank2/{}/A", type_name<TestType>()), size1, size2},
            B_disk{fmt::format("/dot/rank2/{}/B", type_name<TestType>()), size1, size2};

        A_disk(All, All) = A;
        B_disk(All, All) = B;

        auto expected = einsums::linear_algebra::dot(A, B);
        auto got      = einsums::linear_algebra::dot(A_disk, B_disk);

        REQUIRE_THAT(got, einsums::CheckWithinRel(expected));
    }
}