//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

// The 'environ' should be declared in some cases. E.g. Linux man page says:
// (This variable must be declared in the user program, but is declared in
// the header file unistd.h in case the header files came from libc4 or libc5,
// and in case they came from glibc and _GNU_SOURCE was defined.)
// To be safe, declare it here.

#if defined(__linux) || defined(linux) || defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#elif defined(__APPLE__)
// It appears that on Mac OS X the 'environ' variable is not
// available to dynamically linked libraries.
// See: http://article.gmane.org/gmane.comp.lib.boost.devel/103843
// See: http://lists.gnu.org/archive/html/bug-guile/2004-01/msg00013.html
#    include <unistd.h>
// The proper include for this is crt_externs.h, however it's not
// available on iOS. The right replacement is not known. See
// https://svn.boost.org/trac/boost/ticket/5053
extern "C" {
extern char ***_NSGetEnviron(void);
}
#    define environ (*_NSGetEnviron())
#elif defined(EINSUMS_WINDOWS)
#    include <winsock2.h>
#    define environ _environ
#elif defined(__FreeBSD__)
// On FreeBSD the environment is available for executables only, so needs to be
// handled explicitly (e.g. see einsums_init_impl.hpp)
// The variable is defined in .../runtime/src/custom_exception_info.cpp
extern EINSUMS_EXPORT char **freebsd_environ;
#else
extern char **environ;
#endif
