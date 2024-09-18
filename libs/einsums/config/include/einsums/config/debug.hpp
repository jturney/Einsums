//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)
/// Defined if einsums is compiled in debug mode.
#    define EINSUMS_DEBUG
/// Evaluates to ``debug`` if compiled in debug mode, ``release`` otherwise.
#    define EINSUMS_BUILD_TYPE
#else

#    if defined(EINSUMS_DEBUG)
#        define EINSUMS_BUILD_TYPE debug
#    else
#        define EINSUMS_BUILD_TYPE release
#    endif
#endif
