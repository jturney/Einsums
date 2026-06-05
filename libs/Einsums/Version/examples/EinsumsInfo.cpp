//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Version.hpp>

int main() {
    using namespace einsums;

    einsums::println(complete_version());

    return EXIT_SUCCESS;
}