//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

namespace {
auto einsums_main() -> int {
    using namespace einsums;

    size_t const i{10};

    auto A = create_random_tensor("A", i, i);
    auto B = create_random_tensor("B", i, i);
    auto C = create_zero_tensor("C", i, i);

    tensor_algebra::einsum(Indices{index::i, index::j}, &C, Indices{index::i, index::k}, A, Indices{index::k, index::j}, B);

    println(A);
    println(B);
    println(C);

    finalize();
    return EXIT_SUCCESS;
}
}

auto main(int argc, char **argv) -> int {
    return einsums::start(einsums_main, argc, argv);
}