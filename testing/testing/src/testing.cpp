//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#define SNITCH_IMPLEMENTATION
#define SNITCH_DEFINE_MAIN 0
#include "einsums/core/InitializeFinalize.hpp"

#include "snitch/snitch.hpp"

auto main(int argc, char *argv[]) -> int {

    // Parse command-line arguments
    std::optional<snitch::cli::input> args = snitch::cli::parse_arguments(argc, argv);

    if (!args) {
        // Parsing failed.
        return 1;
    }

    // Configure snitch using command-line options
    snitch::tests.configure(*args);

    einsums::initialize();

    // Actually run the tests
    int result = snitch::tests.run_tests(*args) ? 0 : 1;

    einsums::finalize();

    return result;
}