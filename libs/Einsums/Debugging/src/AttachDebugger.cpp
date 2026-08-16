//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Debugging/AttachDebugger.hpp>

#include <array>
#include <iostream>
#include <string>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <Windows.h>
#endif

EINSUMS_NAMESPACE_BEGIN(util)

namespace {

std::string hostname() {
#if defined(EINSUMS_WINDOWS)
    char  hostname[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD size = sizeof(hostname);
    if (GetComputerNameA(hostname, &size)) {
        return std::string(hostname);
    } else {
        return "Unknown";
    }
#else
    std::array<char, 256> hostname{};
    if (gethostname(hostname.data(), hostname.size()) == 0) {
        return std::string(hostname.data());
    }
    return "Unknown";
#endif
}
} // namespace

void attach_debugger() {
#if defined(EINSUMS_WINDOWS)
    DebugBreak();
#elif defined(_POSIX_VERSION) && defined(EINSUMS_HAVE_UNISTD_H)
    int volatile i = 0;
    std::cerr << "PID: " << getpid() << " on " << hostname()
              << " ready for attaching debugger. Once attached set i = 1 "
                 "and continue"
              << std::endl;
    while (i == 0) {
        sleep(1);
    }
#endif
}

EINSUMS_NAMESPACE_END(util)