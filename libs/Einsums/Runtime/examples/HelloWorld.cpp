//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>

int einsums_main() {
    using namespace einsums;
    using namespace einsums::index;

    println("Hello world!");

    auto A = create_random_tensor("A", 3, 3);
    auto B = create_random_tensor("B", 3, 3);
    auto C = create_random_tensor("C", 3, 3);

    tensor_algebra::einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

#pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        println("Hello world {}", i);
    }

    return finalize();
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}