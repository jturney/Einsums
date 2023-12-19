//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/CompilerSpecific.hpp>
#include <einsums/config/Debug.hpp>

// enable auto-linking for supported platforms
#if defined(EINSUMS_MSVC) || defined(__BORLANDC__) || (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) ||                \
    (defined(__ICL) && defined(_MSC_EXTENSIONS) && (EINSUMS_MSVC >= 1200))

#    if !defined(EINSUMS_AUTOLINK_LIB_NAME)
#        error "Macro EINSUMS_AUTOLINK_LIB_NAME not set (internal error)"
#    endif

#    if defined(EINSUMS_DEBUG)
#        pragma comment(lib, EINSUMS_AUTOLINK_LIB_NAME "d"                                                                                 \
                                                       ".lib")
#    else
#        pragma comment(lib, EINSUMS_AUTOLINK_LIB_NAME ".lib")
#    endif

#endif

#undef EINSUMS_AUTOLINK_LIB_NAME
