//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/tensor/tensor.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE("creation") {
    using namespace einsums;

    tensor A("A", 1, 1);
    tensor B("B", 1, 1);

    REQUIRE((A.dim(0) == 1 && A.dim(1) == 1));
    REQUIRE((B.dim(0) == 1 && B.dim(1) == 1));

    A.resize(dim{3, 3});
    B.resize(dim{3, 3});

    auto C = create_tensor<double>("C", 3, 3);
    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));

    A.zero();
    B.zero();
    C.zero();

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(einsums::vector_type<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(einsums::vector_type<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    CHECK_THAT(C.vector_data(), Catch::Matchers::Equals(einsums::vector_type<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));

    // Set A and B to identity
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    CHECK_THAT(A.vector_data(), Catch::Matchers::Equals(einsums::vector_type<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    CHECK_THAT(B.vector_data(), Catch::Matchers::Equals(einsums::vector_type<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
}

TEST_CASE("TensorView creation", "[tensor]") {
    // With the aid of deduction guides we can choose to not specify the rank on the tensor
    einsums::tensor      A("A", 3, 3, 3);
    einsums::tensor_view viewA(A, einsums::dim{3, 9});

    // Since we are changing the underlying datatype to float the deduction guides will not work.
    einsums::tensor      fA("A", 3, 3, 3);
    einsums::tensor_view fviewA(fA, einsums::dim{3, 9});

    for (int i = 0, ijk = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++, ijk++)
                A(i, j, k) = ijk;

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
    REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

    for (int i = 0, ij = 0; i < 3; i++)
        for (int j = 0; j < 9; j++, ij++)
            REQUIRE(viewA(i, j) == ij);
}

// TEST_CASE("TensorView Ranges") {
//     using namespace einsums;
//
//     SECTION("Subviews") {
//         auto                 C = einsums::create_random_tensor("C", 3, 3);
//         einsums::tensor_view viewC(C, einsums::Dim{2, 2}, einsums::Offset{1, 1}, einsums::Stride{3, 1});
//
//         // einsums::println("C strides: %zu %zu\n", C.strides()[0], C.strides()[1]);
//
//         REQUIRE(C(1, 1) == viewC(0, 0));
//         REQUIRE(C(1, 2) == viewC(0, 1));
//         REQUIRE(C(2, 1) == viewC(1, 0));
//         REQUIRE(C(2, 2) == viewC(1, 1));
//     }
//
//     SECTION("Subviews 2") {
//         auto C = einsums::create_random_tensor("C", 3, 3);
//         // std::array<einsums::Range, 2> test;
//         einsums::TensorView viewC = C(einsums::Range{1, 3}, einsums::Range{1, 3});
//
//         // einsums::println(C);
//         // einsums::println(viewC);
//
//         REQUIRE(C(1, 1) == viewC(0, 0));
//         REQUIRE(C(1, 2) == viewC(0, 1));
//         REQUIRE(C(2, 1) == viewC(1, 0));
//         REQUIRE(C(2, 2) == viewC(1, 1));
//     }
//
//     // SECTION("Subviews 3") {
//     //     auto C = create_random_tensor("C", 3, 3, 3, 3);
//     //     auto viewC = C(0, 0, Range{1, 3}, Range{1, 3});
//
//     //     println(C);
//     //     println(viewC);
//     // }
// }
