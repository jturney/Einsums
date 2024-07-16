// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#include <einsums/debugging/attach_debugger.hpp>

#include <iostream>

#if defined(EINSUMS_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#include <Windows.h>
#endif

namespace einsums::debugging {

void attach_debugger() {
#if defined(_POSIX_VERSION) && defined(EINSUMS_HAVE_UNISTD_H)
    volatile int i = 0;
    std::cerr << "PID: " << getpid()
              << " ready for attaching debugger. Once attached set i = 1 and continue"
              << std::endl;
    while (i == 0) { sleep(1); }
#elif defined(EINSUMS_WINDOWS)
    DebugBreak();
#endif
}

}