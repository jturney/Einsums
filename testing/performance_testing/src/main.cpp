//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Runtime.hpp>

#include <benchmark/benchmark.h>

namespace {
auto einsums_main(int argc, char ** argv) -> int {
#pragma omp parallel
    {
#pragma omp single
        {
            benchmark::MaybeReenterWithoutASLR(argc, argv);
            benchmark::Initialize(&argc, argv);
            benchmark::ReportUnrecognizedArguments(argc, argv);
            benchmark::RunSpecifiedBenchmarks();
            benchmark::Shutdown();

            einsums::finalize();
        }
    }
    return EXIT_SUCCESS;
}
}

auto main(int argc, char **argv) -> int {
    return einsums::start(einsums_main, argc, argv);
}
