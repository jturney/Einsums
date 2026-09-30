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
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <Einsums/Testing.hpp>
#include <catch2/catch_all.hpp>

namespace {

// The einsums engine is otherwise seeded once, from the clock, when the
// runtime starts. Reseeding it at every pass through a test case makes each
// test's draws a function of the printed run seed and the test's own name, so
// a failure reproduces with "<binary> '<test name>' --rng-seed N" no matter
// which tests ran before it or which section is selected.
class ReseedEinsumsRandomEngine final : public Catch::EventListenerBase {
  public:
    using Catch::EventListenerBase::EventListenerBase;

    void testCasePartialStarting(Catch::TestCaseInfo const &info, uint64_t /*part_number*/) override {
        einsums::seed_random(einsums::testing::test_case_seed(Catch::getSeed(), info.name));
    }
};

} // namespace

CATCH_REGISTER_LISTENER(ReseedEinsumsRandomEngine)

namespace {
int einsums_main(int /*argc*/, char *const *const argv) {
    // Catch2's session runs on a single thread — every test that needs OMP
    // parallelism spawns its own region from within its body. The earlier
    // shape of this function wrapped session.run() in
    //   #pragma omp parallel { #pragma omp single { result = session.run(); } }
    // which spawned an OMP team just to run the test session on the master.
    // That wrapping doesn't add anything functionally, but it does create a
    // TSan false-positive on every test binary: the `int result = 0;`
    // initialization happens-before the in-single assignment via OMP's
    // barrier, but libgomp is uninstrumented so TSan flags it as a race.
    // Drop the OMP layer; tests that need it still get OMP via their own
    // bodies.
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

    int const result = session.run();

    einsums::finalize();

    // Catch2 answers 4 when every test case it ran skipped, which ctest would count as a failure.
    // Hand back 77 instead, the skip code every Catch2 test registration and simd_rung_guard share,
    // so a binary with nothing to check on this machine reports "Skipped".
    constexpr int catch_all_skipped = 4;
    constexpr int ctest_skipped     = 77;
    return result == catch_all_skipped ? ctest_skipped : result;
}
} // namespace

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
