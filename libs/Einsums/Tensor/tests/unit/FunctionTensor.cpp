//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/FunctionTensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

double prod(std::array<ptrdiff_t, 2> const &vals) {
    return (vals[0] + 1) * (vals[1] + 1);
}

TEST_CASE("Function Tensor") {
    auto A = einsums::FuncPointerTensor<double, 2>("A", prod, 10, 10);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            CHECK_THAT(A(i, j), Catch::Matchers::WithinAbs((i + 1) * (j + 1), 1e-7));
        }
    }

    auto B = A(einsums::All, einsums::Range{5, 10});

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK_THAT(B(i, j), Catch::Matchers::WithinAbs(A(i, j + 5), 1e-7));
        }
    }
}

TEST_CASE("Function Tensor Mixed Einsum") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    auto A = einsums::FuncPointerTensor<double, 2>("A", prod, 10, 10);
    auto B = einsums::create_random_tensor("A", 10, 10);

    auto C = einsums::Tensor<double, 2>("C", 10, 10);

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::j}, A, Indices{index::i, index::j}, B));

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE_THAT(C(i, j), Catch::Matchers::WithinAbs(prod({i, j}) * B(i, j), 1e-10));
        }
    }
}
