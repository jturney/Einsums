//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#if (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) && !defined(__bgq__) &&         \
    !defined(__powerpc__) && !defined(__s390x__) && !defined(__arm__) && !defined(__arm64__) && !defined(__aarch64__)

#    if defined(__x86_64__) || defined(__amd64__)
#        include "swapcontext64.ipp"
#    elif defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
#        include "swapcontext32.ipp"
#    else
#        error You are trying to use x86 context switching on a non-x86 platform. Your \
    platform may be supported with the CMake option \
    EINSUMS_WITH_BOOST_CONTEXT=ON (requires Boost.Context).
#    endif

#endif
