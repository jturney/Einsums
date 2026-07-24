//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>

int einsums_main() {
    using namespace einsums;

    size_t i{10};

    auto              A = create_random_tensor("A", i);
    auto              B = create_random_tensor("B", i);
    Tensor<double, 0> C;

    tensor_algebra::einsum(Indices{}, &C, Indices{index::i}, A, Indices{index::i}, B);

    einsums::println(A);
    einsums::println(B);
    einsums::println(C);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}