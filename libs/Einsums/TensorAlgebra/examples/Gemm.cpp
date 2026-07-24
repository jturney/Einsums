//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

int einsums_main() {
    using namespace einsums;

    size_t i{10};

    auto A = create_random_tensor("A", i, i);
    auto B = create_random_tensor("B", i, i);
    auto C = create_zero_tensor("C", i, i);

    tensor_algebra::einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::i}, B);

    einsums::println(A);
    einsums::println(B);
    einsums::println(C);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}