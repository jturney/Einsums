//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#define CATCH_CONFIG_RUNNER

#include <einsums/init.hpp>
#include <einsums/logging.hpp>
#include <einsums/modules/profile.hpp>

#include <catch2/catch_all.hpp>

int einsums_main() {
    EINSUMS_LOG(info, "in einsums_main")
    Catch::StringMaker<float>::precision  = 10;
    Catch::StringMaker<double>::precision = 17;

    Catch::Session session;
    // session.applyCommandLine(argc, argv);
    // auto cli = session.cli();
    // session.cli(cli);
    int result = session.run();
    return result;
}

int main(int argc, char **argv) {

    // Initialize einsums runtime
    // einsums::initialize();
    einsums::profile::timer::initialize();

    einsums::init(einsums_main, argc, argv);

    // Shutdown einsums runtime
    einsums::profile::timer::finalize();
    // einsums::finalize(false);

    return 0;
}
