//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file
/// What every Waggle header needs: the export macro, the namespace, and preprocessor helpers.

// The collector is always a shared library, so a symbol is exported while building it and imported
// everywhere else. Only Windows needs the import side spelled out.
#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(WAGGLE_EXPORTS)
#        define WAGGLE_EXPORT __declspec(dllexport)
#    else
#        define WAGGLE_EXPORT __declspec(dllimport)
#    endif
#else
#    define WAGGLE_EXPORT __attribute__((visibility("default")))
#endif

// An inline namespace carries the ABI version, so collectors built against incompatible versions
// have different symbol names and cannot be mixed in one call.
#define WAGGLE_NAMESPACE_BEGIN                                                                                                             \
    namespace waggle {                                                                                                                     \
    inline namespace v0 {
#define WAGGLE_NAMESPACE_END                                                                                                               \
    }                                                                                                                                      \
    }

#define WAGGLE_PP_CAT_IMPL(a, b) a##b
/// Paste two tokens after expanding them, for names built from __LINE__.
#define WAGGLE_PP_CAT(a, b) WAGGLE_PP_CAT_IMPL(a, b)
