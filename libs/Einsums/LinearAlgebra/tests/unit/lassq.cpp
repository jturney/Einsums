//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <random>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Base Tensors", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using T      = TestType;
    auto A       = create_random_tensor<T>("a", 10);
    auto scale   = RemoveComplexT<T>{1.0};
    auto result  = RemoveComplexT<T>{0.0};
    auto result0 = RemoveComplexT<T>{0.0};

    linear_algebra::sum_square(A, &scale, &result);

    for (int i = 0; i < 10; i++) {
        if constexpr (IsComplexV<T>) {
            result0 += A(i).real() * A(i).real() + A(i).imag() * A(i).imag();
        } else {
            result0 += A(i) * A(i);
        }
    }

    REQUIRE_THAT(result * scale * scale, Catch::Matchers::WithinAbs(result0, 0.00001));
}

TEMPLATE_TEST_CASE("Block Tensors", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    BlockTensor<TestType, 2> A{"A", 4, 0, 1, 2};

    A.block(0) = create_random_tensor<TestType>("A tile 1", 4, 4);
    A.block(2) = create_random_tensor<TestType>("A tile 3", 1, 1);
    A.block(3) = create_random_tensor<TestType>("A tile 4", 2, 2);

    RemoveComplexT<TestType> scale{0.0}, sumsq{0.0}, result{0.0};

    linear_algebra::sum_square(A, &scale, &sumsq);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            TestType value = A(i, j);
            if constexpr (IsComplexV<TestType>) {
                result += value.real() * value.real() + value.imag() * value.imag();
            } else {
                result += value * value;
            }
        }
    }

    REQUIRE_THAT(scale * scale * sumsq, Catch::Matchers::WithinAbs(result, 0.00001));
}

TEMPLATE_TEST_CASE("Tiled Tensors", "[linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    TiledTensor<TestType, 2> A{"A", std::vector<int>{4, 0, 1, 2}};

    std::uniform_int_distribution random(0, 1);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == 1 || j == 1) {
                continue;
            }

            if (random(einsums::random_engine) == 1) {
                A.tile(i, j) = create_random_tensor<TestType>("A tile", A.tile_size(0)[i], A.tile_size(1)[j]);
            }
        }
    }

    RemoveComplexT<TestType> scale{0.0}, sumsq{0.0}, result{0.0};

    linear_algebra::sum_square(A, &scale, &sumsq);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            TestType value = A(i, j);
            if constexpr (IsComplexV<TestType>) {
                result += value.real() * value.real() + value.imag() * value.imag();
            } else {
                result += value * value;
            }
        }
    }

    REQUIRE_THAT(scale * scale * sumsq, Catch::Matchers::WithinAbs(result, 0.00001));
}

// A type the vendor has no lassq for takes a hand-written branch, which summed into a local and
// never wrote it back: sumsq came out unchanged, so the sum of squares was zero.
TEST_CASE("sum_square over a type without a vendor lassq", "[linear-algebra]") {
    Tensor<int, 1> A("A", 4);
    for (int i = 0; i < 4; i++) {
        A(i) = i + 1; // 1 + 4 + 9 + 16 = 30
    }

    int scale = 1, sumsq = 0;
    linear_algebra::sum_square(A, &scale, &sumsq);
    REQUIRE(scale * scale * sumsq == 30);

    // Accumulating onto a prior total, as lassq does.
    linear_algebra::sum_square(A, &scale, &sumsq);
    REQUIRE(scale * scale * sumsq == 60);
}
