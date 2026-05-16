//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Utilities/Random.hpp>

#if defined(EINSUMS_HAVE_MPI)
#    include <mpi.h>
#endif

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/internal/catch_context.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

namespace {
int einsums_main(int /*argc*/, char *const *const argv) {
    int result = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            auto const               &config = einsums::runtime_config();
            Catch::Session            session;
            std::vector<char const *> args;
            args.reserve(config.unknown_arguments().size() + 1);
            args.push_back(argv[0]);
            for (auto const &s : config.unknown_arguments())
                args.push_back(s.c_str());
            session.applyCommandLine(static_cast<int>(args.size()), args.data());

#if defined(EINSUMS_HAVE_MPI)
            // For MPI: broadcast rank 0's random seed so all ranks run tests in the same order.
            // All ranks must execute the same collective at the same time.
            {
                auto seed = session.configData().rngSeed;
                MPI_Bcast(&seed, sizeof(seed), MPI_BYTE, 0, MPI_COMM_WORLD);
                session.configData().rngSeed = seed;
            }
#endif

            Catch::StringMaker<float>::precision  = std::numeric_limits<float>::digits10;
            Catch::StringMaker<double>::precision = std::numeric_limits<double>::digits10;
            auto seed                             = session.config().rngSeed();

            einsums::seed_random(seed);

            { result = session.run(); }
        }
    }

    einsums::finalize();

    return result;
}
} // namespace

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
