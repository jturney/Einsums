//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>

int einsums_main() {
    using namespace einsums;

    size_t i{10};

    auto A = create_random_tensor("A", i);
    auto B = create_random_tensor("B", i);

    double C = linear_algebra::dot(A, B);

    einsums::println(A);
    einsums::println(B);
    einsums::println("C = {}", C);

    finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}