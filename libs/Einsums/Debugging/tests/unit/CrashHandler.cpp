//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Debugging/CrashHandler.hpp>

#include <string>

#include <catch2/catch_all.hpp>

// What is testable here without faulting on purpose is the lifecycle. The report
// itself only happens on a real crash, which would take the test process with it, so
// the value of these cases is that install/remove can be driven in any order from any
// caller - which is what a handler installed from einsums::initialize() and torn down
// by a later one has to survive.
//
// The reporting path is covered where it actually matters: a crash in CI writes the
// dump the Windows job uploads.

TEST_CASE("crash handler installs and removes", "[debugging][crash-handler]") {
    SECTION("install then remove") {
        REQUIRE_NOTHROW(einsums::util::install_crash_handler());
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
    }

    SECTION("install is idempotent") {
        REQUIRE_NOTHROW(einsums::util::install_crash_handler());
        REQUIRE_NOTHROW(einsums::util::install_crash_handler());
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
    }

    SECTION("remove without install is a no-op") {
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
    }

    SECTION("a dump directory is accepted") {
        REQUIRE_NOTHROW(einsums::util::install_crash_handler(std::string{"."}));
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
    }

    SECTION("an over-long dump directory is refused rather than overrunning") {
        // MAX_PATH is the buffer the handler copies into. Anything longer has to be
        // dropped, leaving the default working-directory behavior, because the copy
        // happens here and the crash that reads it happens much later.
        REQUIRE_NOTHROW(einsums::util::install_crash_handler(std::string(8192, 'x')));
        REQUIRE_NOTHROW(einsums::util::remove_crash_handler());
    }
}
