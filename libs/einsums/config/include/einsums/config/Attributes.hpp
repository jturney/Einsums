//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/CompilerSpecific.hpp>
#include <einsums/config/defines.hpp>

#if defined(DOXYGEN)

/// Function attribute to tell compiler not to inline the function.
#    define EINSUMS_NOINLINE

/// Marks an entity as deprecated. The argument \c x specifies a custom message
/// that is included in the compiler warning. For more details see
/// `<https://en.cppreference.com/w/cpp/language/attributes/deprecated>`__.
#    define EINSUMS_DEPRECATED(x)

/// Indicates that this data member need not have an address distinct from all
/// other non-static data members of its class.
/// For more details see
/// `https://en.cppreference.com/w/cpp/language/attributes/no_unique_address`__.
#    define EINSUMS_NO_UNIQUE_ADDRESS
#else

///////////////////////////////////////////////////////////////////////////////
#    if defined(EINSUMS_MSVC)
#        define EINSUMS_NOINLINE __declspec(noinline)
#    elif defined(__GNUC__)
#        if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
// nvcc doesn't always parse __noinline
#            define EINSUMS_NOINLINE __attribute__((noinline))
#        else
#            define EINSUMS_NOINLINE __attribute__((__noinline__))
#        endif
#    else
#        define EINSUMS_NOINLINE
#    endif

///////////////////////////////////////////////////////////////////////////////
// handle [[deprecated]]
#    if EINSUMS_HAVE_DEPRECATION_WARNINGS && !defined(EINSUMS_INTEL_VERSION)
#        define EINSUMS_DEPRECATED_MSG "This functionality is deprecated and will be removed in the future."
#        define EINSUMS_DEPRECATED(x)  [[deprecated(x)]]
#    endif

#    if !defined(EINSUMS_DEPRECATED)
#        define EINSUMS_DEPRECATED(x)
#    endif

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#    if defined(_MSC_VER)
#        define EINSUMS_EMPTY_BASES __declspec(empty_bases)
#    else
#        define EINSUMS_EMPTY_BASES
#    endif

///////////////////////////////////////////////////////////////////////////////
// handle [[no_unique_address]]
#    if defined(EINSUMS_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE) && !defined(EINSUMS_HAVE_CUDA)
#        define EINSUMS_NO_UNIQUE_ADDRESS [[no_unique_address]]
#    else
#        define EINSUMS_NO_UNIQUE_ADDRESS
#    endif

///////////////////////////////////////////////////////////////////////////////
// handle empty_bases
#    if defined(_MSC_VER)
#        define EINSUMS_EMPTY_BASES __declspec(empty_bases)
#    else
#        define EINSUMS_EMPTY_BASES
#    endif

#endif
