//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Runtime.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <catch2/catch_session.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

namespace {
auto einsums_main(int argc, char *const *const argv) -> int {
    int result;
#pragma omp parallel
    {
#pragma omp single
        {
            Catch::Session session;
            session.applyCommandLine(argc, argv);

            Catch::StringMaker<float>::precision  = std::numeric_limits<float>::digits10;
            Catch::StringMaker<double>::precision = std::numeric_limits<double>::digits10;

            einsums::seed_random(session.config().rngSeed());

            result = session.run();
            einsums::finalize();
        }
    }
    return result;
}
}

auto main(int argc, char **argv) -> int {
    return einsums::start(einsums_main, argc, argv);
}
