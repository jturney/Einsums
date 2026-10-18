//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>

int main(int argc, char **argv) {
    einsums::initialize(argc, argv);

    einsums::println("Hello world!");

    einsums::finalize();

    return EXIT_SUCCESS;
}
