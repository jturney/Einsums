//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/linear_algebra/linear_algebra.hpp>
#include <einsums/modules/tensor_utilities.hpp>
#include <einsums/testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("dot", "[linear-algebra]", float, double) {
    using namespace einsums;
    using namespace einsums::linear_algebra;

    constexpr int size = 10;

    SECTION("Rank 1 tensors") {
        tensor<TestType, 1> A = create_random_tensor<TestType>("A", size);
        tensor<TestType, 1> B = create_random_tensor<TestType>("B", size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            test += A(i) * B(i);
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::WithinStrict(test, TestType{100000.0}));
    }

    SECTION("Rank 1 tensor views") {
        tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        for (int i = 0; i < size; i++) {
            auto A_view = A(all, i);
            auto B_view = B(i, all);

            TestType test{0.0};

            for (int j = 0; j < size; j++) {
                test += A(j, i) * B(i, j);
            }

            auto dot_res = dot(A_view, B_view);

            REQUIRE_THAT(dot_res, einsums::WithinStrict(test, TestType{100000.0}));
        }
    }

    SECTION("Rank 2 tensors") {
        tensor<TestType, 2> A = create_random_tensor<TestType>("A", size, size);
        tensor<TestType, 2> B = create_random_tensor<TestType>("B", size, size);

        TestType test{0.0};

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                test += A(i, j) * B(i, j);
            }
        }

        auto dot_res = dot(A, B);

        REQUIRE_THAT(dot_res, einsums::WithinStrict(test, TestType{100000.0}));
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
