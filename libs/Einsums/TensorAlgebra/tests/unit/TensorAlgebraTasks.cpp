//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("Test dependence timing", "[jobs]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    auto A = create_random_tensor("A", 100, 100);
    auto B = create_random_tensor("B", 100, 100);
    auto C = create_tensor("C", 100, 100);
    auto D = create_tensor("D", 100, 100);

    SECTION("Sequential") {
        WAGGLE_ZONE("Sequential");

        for (int sentinel = 0; sentinel < 10; sentinel++) {
            einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
            einsum(Indices{i, j}, &D, Indices{i, k}, A, Indices{k, j}, B);
        }
    }

    SECTION("Tasked") {
#pragma omp parallel
#pragma omp single
        {
            WAGGLE_ZONE("Tasked");

            for (int sentinel = 0; sentinel < 10; sentinel++) {
#pragma omp task depend(in : A, B), depend(out : C)
                {
                    C.lock();
                    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
                    C.unlock();
                }

#pragma omp task depend(in : A, B), depend(out : D)
                {
                    D.try_lock();
                    einsum(Indices{i, j}, &D, Indices{i, k}, A, Indices{k, j}, B);
                    D.unlock();
                }
            }
        }
    }
}