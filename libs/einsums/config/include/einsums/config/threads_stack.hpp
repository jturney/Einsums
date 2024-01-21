//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/compiler_specific.hpp>
#include <einsums/config/debug.hpp>
#include <einsums/config/defines.hpp>

///////////////////////////////////////////////////////////////////////////////

#if !defined(EINSUMS_THREADS_STACK_OVERHEAD)
#    if defined(EINSUMS_DEBUG)
#        if defined(EINSUMS_GCC_VERSION)
#            define EINSUMS_THREADS_STACK_OVERHEAD 0x3000
#        else
#            define EINSUMS_THREADS_STACK_OVERHEAD 0x2800
#        endif
#    else
#        if defined(EINSUMS_INTEL_VERSION)
#            define EINSUMS_THREADS_STACK_OVERHEAD 0x2800
#        else
#            define EINSUMS_THREADS_STACK_OVERHEAD 0x800
#        endif
#    endif
#endif

#if !defined(EINSUMS_SMALL_STACK_SIZE)
#    if defined(__has_feature)
#        if __has_feature(address_sanitizer)
#            define EINSUMS_SMALL_STACK_SIZE 0x40000 // 256kByte
#        endif
#    endif
#endif

#if !defined(EINSUMS_SMALL_STACK_SIZE)
#    if defined(EINSUMS_WINDOWS) && !defined(EINSUMS_HAVE_BOOST_CONTEXT)
#        define EINSUMS_SMALL_STACK_SIZE 0x4000 // 16kByte
#    else
#        if defined(EINSUMS_DEBUG)
#            define EINSUMS_SMALL_STACK_SIZE 0x20000 // 128kByte
#        else
#            if defined(__powerpc__) || defined(__INTEL_COMPILER)
#                define EINSUMS_SMALL_STACK_SIZE 0x20000 // 128kByte
#            else
#                define EINSUMS_SMALL_STACK_SIZE 0x10000 // 64kByte
#            endif
#        endif
#    endif
#endif

#if !defined(EINSUMS_MEDIUM_STACK_SIZE)
#    define EINSUMS_MEDIUM_STACK_SIZE 0x0020000 // 128kByte
#endif
#if !defined(EINSUMS_LARGE_STACK_SIZE)
#    define EINSUMS_LARGE_STACK_SIZE 0x0200000 // 2MByte
#endif
#if !defined(EINSUMS_HUGE_STACK_SIZE)
#    define EINSUMS_HUGE_STACK_SIZE 0x2000000 // 32MByte
#endif
