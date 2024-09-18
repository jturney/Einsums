//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/defines.hpp>

#if defined(DOXYGEN)
/// Marks a class or function to be exported from einsums or imported if it is
/// consumed.
#    define EINSUMS_EXPORT
#else

#    if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#        if !defined(EINSUMS_MODULE_STATIC_LINKING)
#            define EINSUMS_SYMBOL_EXPORT   __declspec(dllexport)
#            define EINSUMS_SYMBOL_IMPORT   __declspec(dllimport)
#            define EINSUMS_SYMBOL_INTERNAL /* empty */
#        endif
#    elif defined(__NVCC__) || defined(__CUDACC__)
#        define EINSUMS_SYMBOL_EXPORT   /* empty */
#        define EINSUMS_SYMBOL_IMPORT   /* empty */
#        define EINSUMS_SYMBOL_INTERNAL /* empty */
#    elif defined(EINSUMS_HAVE_ELF_HIDDEN_VISIBILITY)
#        define EINSUMS_SYMBOL_EXPORT   __attribute__((visibility("default")))
#        define EINSUMS_SYMBOL_IMPORT   __attribute__((visibility("default")))
#        define EINSUMS_SYMBOL_INTERNAL __attribute__((visibility("hidden")))
#    endif

// make sure we have reasonable defaults
#    if !defined(EINSUMS_SYMBOL_EXPORT)
#        define EINSUMS_SYMBOL_EXPORT /* empty */
#    endif
#    if !defined(EINSUMS_SYMBOL_IMPORT)
#        define EINSUMS_SYMBOL_IMPORT /* empty */
#    endif
#    if !defined(EINSUMS_SYMBOL_INTERNAL)
#        define EINSUMS_SYMBOL_INTERNAL /* empty */
#    endif

///////////////////////////////////////////////////////////////////////////////
#    if defined(EINSUMS_EXPORTS)
#        define EINSUMS_EXPORT EINSUMS_SYMBOL_EXPORT
#    else
#        define EINSUMS_EXPORT EINSUMS_SYMBOL_IMPORT
#    endif

///////////////////////////////////////////////////////////////////////////////
// helper macro for symbols which have to be exported from the runtime and all
// components
#    define EINSUMS_ALWAYS_EXPORT EINSUMS_SYMBOL_EXPORT
#    define EINSUMS_ALWAYS_IMPORT EINSUMS_SYMBOL_IMPORT
#endif
