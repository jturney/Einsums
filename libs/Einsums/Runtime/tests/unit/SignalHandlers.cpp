//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Testing.hpp>

#if !defined(EINSUMS_WINDOWS)
#    include <array>
#    include <cerrno>
#    include <csignal>
#    include <unistd.h>

/// A consumer that stops reading early is ordinary use, not a crash.
///
/// SIGPIPE used to be installed on the same handler as SIGSEGV. That handler
/// symbolizes a backtrace, and the write that raised the signal was inside
/// stdio holding the lock the symbolizer then waited for, so any Einsums
/// program piped into `head` hung forever rather than ending: no output, no
/// diagnostic, and immune to SIGTERM.
EINSUMS_TEST_CASE("SIGPIPE is ignored rather than handled", "[runtime][signals]") {
    struct sigaction current{};
    REQUIRE(sigaction(SIGPIPE, nullptr, &current) == 0);
    CHECK(current.sa_handler == SIG_IGN);
}

/// The behavior the disposition buys, and the half that would have caught the
/// original bug: on the old code this case did not fail, it took down the test
/// binary through the fatal handler.
EINSUMS_TEST_CASE("a write to a pipe with no reader returns EPIPE", "[runtime][signals]") {
    std::array<int, 2> fds{-1, -1};
    REQUIRE(pipe(fds.data()) == 0);

    // The consumer goes away, as `head` does once it has its lines.
    REQUIRE(close(fds[0]) == 0);

    errno              = 0;
    auto const written = write(fds[1], "x", 1);

    CHECK(written == -1);
    CHECK(errno == EPIPE);

    close(fds[1]);
}

#else

EINSUMS_TEST_CASE("SIGPIPE has no Windows equivalent", "[runtime][signals]") {
    SUCCEED("Windows has no SIGPIPE; a closed pipe surfaces as a failed write already.");
}

#endif
