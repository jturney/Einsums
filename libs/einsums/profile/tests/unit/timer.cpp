// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/profile/timer.hpp>

#include <snitch/snitch.hpp>

TEST_CASE("timer") {
    einsums::profile::timer::initialize();

#pragma omp parallel for default(none)
    for (int i = 0; i < 100; i++) {
        einsums::profile::timer::push("test timer i = {}", i);
        einsums::profile::timer::pop();
    }

    // einsums::profile::timer::report();
    einsums::profile::timer::finalize();
}
