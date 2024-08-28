//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1437: einsums::init() should strip einsums-related flags from argv

#include <einsums/init.hpp>
#include <einsums/testing.hpp>

#include <cstdlib>

bool invoked_main = false;

int my_einsums_main(int argc, char **) {
    // all einsums command line arguments should have been stripped here
    EINSUMS_TEST_EQ(argc, 1);

    invoked_main = true;
    einsums::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    EINSUMS_TEST_LT(1, argc);

    EINSUMS_TEST_EQ(einsums::init(&my_einsums_main, argc, argv), 0);
    EINSUMS_TEST(invoked_main);

    return 0;
}
