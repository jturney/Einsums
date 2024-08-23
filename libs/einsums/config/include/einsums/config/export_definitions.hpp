// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------
//

#pragma once

#include <einsums/config/defines.hpp>

#if defined(DOXYGEN)
/// Marks a class or function to be exported from einsums or imported if it is comsumed.
#    define EINSUMS_EXPORT
#else

#    if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#        if !defined(EINSUMS_MODULE_STATIC_LINKING)
#            define EINSUMS_EXPORT_SYMBOL   __declspec(dllexport)
#            define EINSUMS_IMPORT_SYMBOL   __declspec(dllimport)
#            define EINSUMS_INTERNAL_SYMBOL /* empty */
#        endif
#    elif defined(__NVCC__) || defined(__CUDACC__)
#        define EINSUMS_EXPORT_SYMBOL   /* empty */
#        define EINSUMS_IMPORT_SYMBOL   /* empty */
#        define EINSUMS_INTERNAL_SYMBOL /* empty */
#    elif defined(EINSUMS_HAVE_ELF_HIDDEN_VISIBILITY)
#        define EINSUMS_EXPORT_SYMBOL   __attribute__((visibility("default")))
#        define EINSUMS_IMPORT_SYMBOL   __attribute__((visibility("default")))
#        define EINSUMS_INTERNAL_SYMBOL __attribute__((visibility("hidden")))
#    endif

// make sure we have reasonable defaults
#    if !defined(EINSUMS_EXPORT_SYMBOL)
#        define EINSUMS_EXPORT_SYMBOL /* empty */
#    endif
#    if !defined(EINSUMS_IMPORT_SYMBOL)
#        define EINSUMS_IMPORT_SYMBOL /* empty */
#    endif
#    if !defined(EINSUMS_INTERNAL_SYMBOL)
#        define EINSUMS_INTERNAL_SYMBOL /* empty */
#    endif

/////////////////////////////////////////////////////////////////////////////
#    if defined(EINSUMS_EXPORTS)
#        define EINSUMS_EXPORT EINSUMS_EXPORT_SYMBOL
#    else
#        define EINSUMS_EXPORT EINSUMS_IMPORT_SYMBOL
#    endif

/////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#    define EINSUMS_ALWAYS_EXPORT EINSUMS_EXPORT_SYMBOL
#    define EINSUMS_ALWAYS_IMPORT EINSUMS_IMPORT_SYMBOL

#endif
