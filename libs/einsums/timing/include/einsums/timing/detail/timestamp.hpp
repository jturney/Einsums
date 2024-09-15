//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#if defined(EINSUMS_MSVC)
#    include <einsums/timing/detail/timestamp/msvc.hpp>
#elif defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64)
#    if (defined(EINSUMS_HAVE_RDTSC) || defined(EINSUMS_HAVE_RDTSCP)) && !defined(EINSUMS_NVHPC_VERSION)
#        include <einsums/timing/detail/timestamp/linux_x86_64.hpp>
#    else
#        include <einsums/timing/detail/timestamp/linux_generic.hpp>
#    endif
#elif defined(i386) || defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__i386) ||              \
    defined(_M_IX86) || defined(__X86__) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__)
#    if (defined(EINSUMS_HAVE_RDTSC) || defined(EINSUMS_HAVE_RDTSCP)) && !defined(EINSUMS_NVHPC_VERSION)
#        include <einsums/timing/detail/timestamp/linux_x86_32.hpp>
#    else
#        include <einsums/timing/detail/timestamp/linux_generic.hpp>
#    endif
#elif (defined(__ANDROID__) && defined(ANDROID))
#    include <einsums/timing/detail/timestamp/linux_generic.hpp>
#elif defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
#    include <einsums/timing/detail/timestamp/linux_generic.hpp>
#elif defined(__ppc__) || defined(__ppc) || defined(__powerpc__)
#    include <einsums/timing/detail/timestamp/linux_generic.hpp>
#elif defined(__s390x__)
#    include <einsums/timing/detail/timestamp/linux_generic.hpp>
#elif defined(__bgq__)
#    include <einsums/timing/detail/timestamp/bgq.hpp>
#else
#    error Unsupported platform.
#endif
