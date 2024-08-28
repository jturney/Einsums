// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/profile/section.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE("section") {

#pragma omp parallel for default(none)
    for (int i = 0; i < 100; i++) {
        einsums::profile::section _section("test section");
    }
}