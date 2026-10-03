//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A program that never calls finalize leaves the profiler's destructor to drain the rings and stop
// the server during static destruction. It used to abort there ("mutex lock failed"): the server
// reported its shutdown through a diagnostics mutex that was first used after the profiler was
// built, and so had already been destroyed. Passes when the process exits cleanly.

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.hpp>

#include <cstdint>

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

namespace {

#ifndef _WIN32
/// A port nothing is listening on (the kernel's pick).
std::int64_t free_port() {
    int const   fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t  len       = sizeof(addr);
    bool const ok        = ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0 &&
                           ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0;
    ::close(fd);
    return ok ? ntohs(addr.sin_port) : 0;
}
#endif

} // namespace

int main() {
    // The profiler is built here, before anything it reports through has been used.
    waggle::set_enabled(true);
    {
        WAGGLE_ZONE("exit: before the server");
    }

#ifndef _WIN32
    // A running server, so the destructor shuts one down and reports doing so.
    if (auto const port = free_port(); port != 0) {
        waggle::configure({.server = true, .port = port});
    }
#endif
    // The first diagnostic of the run, after the profiler was built: a refused setting.
    waggle::configure({.max_distinct_children = 3});
    waggle::configure({.max_distinct_children = 4});

    // Left in the ring for the destructor's final drain.
    {
        WAGGLE_ZONE("exit: left for the drain");
        waggle::annotate("key", "value");
    }
    return 0;
}
