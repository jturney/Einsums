//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(DOXYGEN)
/// Marks a class or function to be exported from einsums or imported if it is
/// consumed.
#    define EINSUMS_EXPORT
#else

/*! \def _EXPORT
    Internal macro that defines export visibility based on platform.
*/
#    if defined(_WIN32) || defined(__CYGWIN__)
#        define EINSUMS_SYMBOL_EXPORT __declspec(dllexport)
#    else
#        define EINSUMS_SYMBOL_EXPORT __attribute__((visibility("default")))
#    endif

/*! \def EINSUMS_EXPORT
    Macro used to decorate classes/functions/structs to be "visible"
    outside the einsums library. Default visibility is set to hidden
    on some platforms.
*/
#    if defined(EINSUMS_LIBRARY)
#        define EINSUMS_EXPORT EINSUMS_SYMBOL_EXPORT
#    elif defined(EINSUMS_STATIC_LIBRARY)
#        define EINSUMS_EXPORT
#    else
#        define EINSUMS_EXPORT EINSUMS_SYMBOL_EXPORT
#    endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#    define EINSUMS_ALWAYS_EXPORT EINSUMS_SYMBOL_EXPORT
#    define EINSUMS_ALWAYS_IMPORT EINSUMS_SYMBOL_IMPORT
#endif
