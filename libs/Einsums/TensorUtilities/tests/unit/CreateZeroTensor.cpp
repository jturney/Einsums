//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <complex>
#include <vector>

#include <Einsums/Testing.hpp>

// These assert what create_zero_tensor no longer does explicitly.
//
// It used to construct a tensor and then call zero() on it, which wrote every
// element a second time; a dimensioned constructor allocates a std::vector,
// which value-initializes, so the buffer is already zero when it returns. The
// redundant pass is gone and this is what holds the guarantee: a change to
// storage that stopped zeroing on construction fails here rather than showing
// up as garbage in whoever read the tensor next.

TEMPLATE_TEST_CASE("CreateZeroTensor is zero without an explicit fill", "[tensor]", float, double, std::complex<float>,
                   std::complex<double>) {
    auto A = einsums::create_zero_tensor<TestType>("A", 4, 5);

    REQUIRE(A.dim(0) == 4);
    REQUIRE(A.dim(1) == 5);
    for (size_t i = 0; i < A.dim(0); i++) {
        for (size_t j = 0; j < A.dim(1); j++) {
            REQUIRE(A(i, j) == TestType{0.0});
        }
    }
}

TEMPLATE_TEST_CASE("CreateZeroTensor is zero at rank 3", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto A = einsums::create_zero_tensor<TestType>("A", 3, 4, 5);

    for (size_t i = 0; i < A.dim(0); i++) {
        for (size_t j = 0; j < A.dim(1); j++) {
            for (size_t k = 0; k < A.dim(2); k++) {
                REQUIRE(A(i, j, k) == TestType{0.0});
            }
        }
    }
}

TEMPLATE_TEST_CASE("CreateZeroTensor is zero in both layouts", "[tensor]", float, double) {
    for (bool const row_major : {true, false}) {
        auto A = einsums::create_zero_tensor<TestType>(row_major, "A", 4, 5);

        for (size_t i = 0; i < A.dim(0); i++) {
            for (size_t j = 0; j < A.dim(1); j++) {
                REQUIRE(A(i, j) == TestType{0.0});
            }
        }
    }
}

TEMPLATE_TEST_CASE("CreateZeroTensor runtime-rank overload is zero", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    auto A = einsums::create_zero_tensor<TestType>("A", std::vector<size_t>{3, 4, 5});

    REQUIRE(A.rank() == 3);
    for (size_t i = 0; i < A.size(); i++) {
        REQUIRE(A.data()[i] == TestType{0.0});
    }
}

TEST_CASE("CreateZeroTensor is zero past one allocation page", "[tensor]") {
    // Small tensors can land in memory that happens to be zero already, which
    // would let a broken guarantee pass. This is large enough to be a fresh
    // mapping and is checked in full.
    auto A = einsums::create_zero_tensor<double>("A", std::vector<size_t>{256, 256});

    for (size_t i = 0; i < A.size(); i++) {
        REQUIRE(A.data()[i] == 0.0);
    }
}

TEST_CASE("CreateZeroTensor tolerates a zero extent", "[tensor]") {
    auto A = einsums::create_zero_tensor<double>("A", std::vector<size_t>{0, 6});

    REQUIRE(A.dim(0) == 0);
    REQUIRE(A.dim(1) == 6);
    REQUIRE(A.size() == 0);
}
