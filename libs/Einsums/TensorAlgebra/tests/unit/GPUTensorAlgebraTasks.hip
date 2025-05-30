//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Testing.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateIdentity.hpp>

TEST_CASE("Test dependence GPU timing", "[jobs][gpu]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    gpu::device_synchronize();

    auto A = DeviceTensor<double, 2>("A", einsums::detail::DEV_ONLY, 100, 100);
    auto B = DeviceTensor<double, 2>("B", einsums::detail::DEV_ONLY, 100, 100);

    A      = create_random_tensor("A", 100, 100);
    B      = create_random_tensor("B", 100, 100);
    auto C = DeviceTensor<double, 2>("C", einsums::detail::DEV_ONLY, 100, 100),
         D = DeviceTensor<double, 2>("D", einsums::detail::DEV_ONLY, 100, 100);
    C.zero();
    D.zero();

    SECTION("Sequential") {
        profile::push("Sequential GPU");

        for (int sentinel = 0; sentinel < 10; sentinel++) {
            REQUIRE_NOTHROW(einsums::tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B));
            REQUIRE_NOTHROW(einsums::tensor_algebra::einsum(Indices{i, j}, &D, Indices{i, k}, A, Indices{k, j}, B));
        }

        profile::pop();
    }

    SECTION("Tasked") {
#pragma omp parallel
#pragma omp single
        {
            profile::push("Tasked GPU");

            for (int sentinel = 0; sentinel < 10; sentinel++) {
#pragma omp task depend(in : A, B), depend(out : C)
                { REQUIRE_NOTHROW(einsums::tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B)); }

#pragma omp task depend(in : A, B), depend(out : D)
                { REQUIRE_NOTHROW(einsums::tensor_algebra::einsum(Indices{i, j}, &D, Indices{i, k}, A, Indices{k, j}, B)); }
            }

            profile::pop();
        }
    }
}