//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>

#include <complex>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Identity", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto I = einsums::create_identity_tensor<TestType>("I", 3, 3);

    REQUIRE(I(0, 0) == TestType{1.0});
    REQUIRE(I(0, 1) == TestType{0.0});
    REQUIRE(I(0, 2) == TestType{0.0});
    REQUIRE(I(1, 0) == TestType{0.0});
    REQUIRE(I(1, 1) == TestType{1.0});
    REQUIRE(I(1, 2) == TestType{0.0});
    REQUIRE(I(2, 0) == TestType{0.0});
    REQUIRE(I(2, 1) == TestType{0.0});
    REQUIRE(I(2, 2) == TestType{1.0});
}

TEMPLATE_TEST_CASE("Identity - 3d", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto I = einsums::create_identity_tensor<TestType>("I", 3, 3, 3);

    auto   strides  = I.strides();
    size_t elements = I.size();

    for (size_t item = 0; item < elements; item++) {
        std::array<size_t, 3> index;
        sentinel_to_indices(item, strides, index);

        auto [i, j, k] = index;

        if (i == j && j == k) {
            REQUIRE(I(i, j, k) == TestType{1.0});
        } else {
            REQUIRE(I(i, j, k) == TestType{0.0});
        }
    }
}

TEMPLATE_TEST_CASE("Identity - rectangular", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    // The diagonal runs to the shortest axis. This used to walk dim(0) instead, so a shape whose
    // first axis is the longer one threw std::out_of_range: create_identity_tensor("c", 6, 3)
    // tried to set (3, 3) on a tensor whose second axis stops at 3. Both orderings are checked,
    // because the bug was invisible whenever dim(0) happened to be smallest.
    SECTION("taller than wide") {
        auto I = einsums::create_identity_tensor<TestType>("I", 6, 3);

        for (size_t i = 0; i < 6; i++) {
            for (size_t j = 0; j < 3; j++) {
                REQUIRE(I(i, j) == (i == j ? TestType{1.0} : TestType{0.0}));
            }
        }
    }

    SECTION("wider than tall") {
        auto I = einsums::create_identity_tensor<TestType>("I", 3, 6);

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 6; j++) {
                REQUIRE(I(i, j) == (i == j ? TestType{1.0} : TestType{0.0}));
            }
        }
    }

    SECTION("rank 3, shortest axis last") {
        auto I = einsums::create_identity_tensor<TestType>("I", 5, 4, 2);

        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 4; j++) {
                for (size_t k = 0; k < 2; k++) {
                    REQUIRE(I(i, j, k) == ((i == j && j == k) ? TestType{1.0} : TestType{0.0}));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Identity - runtime rank", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    // The shape-vector overload, matching create_zero_tensor and create_random_tensor. Without
    // it an identity was the one creator in that group handing back a statically ranked tensor.
    SECTION("square") {
        einsums::RuntimeTensor<TestType> I = einsums::create_identity_tensor<TestType>("I", {3, 3});

        REQUIRE(I.rank() == 2);
        REQUIRE(I.dim(0) == 3);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                REQUIRE(I(i, j) == (i == j ? TestType{1.0} : TestType{0.0}));
            }
        }
    }

    SECTION("rectangular, either ordering") {
        einsums::RuntimeTensor<TestType> tall = einsums::create_identity_tensor<TestType>("tall", {6, 3});
        einsums::RuntimeTensor<TestType> wide = einsums::create_identity_tensor<TestType>("wide", {3, 6});

        for (size_t i = 0; i < 6; i++) {
            for (size_t j = 0; j < 3; j++) {
                REQUIRE(tall(i, j) == (i == j ? TestType{1.0} : TestType{0.0}));
                REQUIRE(wide(j, i) == (i == j ? TestType{1.0} : TestType{0.0}));
            }
        }
    }

    SECTION("rank 3") {
        einsums::RuntimeTensor<TestType> I = einsums::create_identity_tensor<TestType>("I", {3, 3, 3});

        REQUIRE(I.rank() == 3);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    REQUIRE(I(i, j, k) == ((i == j && j == k) ? TestType{1.0} : TestType{0.0}));
                }
            }
        }
    }

    SECTION("a zero extent leaves nothing to set") {
        einsums::RuntimeTensor<TestType> I = einsums::create_identity_tensor<TestType>("I", {4, 0});
        REQUIRE(I.rank() == 2);
        REQUIRE(I.size() == 0);
    }
}
