//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#define CATCH_CONFIG_RUNNER

#include <einsums/modules/profile.hpp>

#include <catch2/catch_all.hpp>

int main(int argc, char **argv) {
    Catch::StringMaker<float>::precision  = 10;
    Catch::StringMaker<double>::precision = 17;

    // Initialize einsums runtime
    // einsums::initialize();
    einsums::profile::timer::initialize();

    Catch::Session session;
    session.applyCommandLine(argc, argv);
    // auto cli = session.cli();
    // session.cli(cli);
    int result = session.run();

    // Shutdown einsums runtime
    einsums::profile::timer::finalize();
    // einsums::finalize(false);

    return result;
}
