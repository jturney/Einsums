//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#
#/* EINSUMS_PP_CONFIG_FLAGS */
#
#define EINSUMS_PP_CONFIG_STRICT() 0x0001
#define EINSUMS_PP_CONFIG_IDEAL()  0x0002
#
#define EINSUMS_PP_CONFIG_MSVC() 0x0004
#define EINSUMS_PP_CONFIG_MWCC() 0x0008
#define EINSUMS_PP_CONFIG_BCC()  0x0010
#define EINSUMS_PP_CONFIG_EDG()  0x0020
#define EINSUMS_PP_CONFIG_DMC()  0x0040
#
#if !defined(EINSUMS_PP_CONFIG_FLAGS)
#    if defined(_GCCXML_)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_STRICT())
#    elif defined(_WAVE_)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_STRICT())
#    elif defined(_MWERKS_) && _MWERKS_ >= 0x3200
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_STRICT())
#    elif defined(_EDG_) || defined(_EDG_VERSION_)
#        if defined(_MSC_VER) && (defined(_clang_) || defined(_INTELLISENSE_) || _EDG_VERSION_ >= 308)
#            define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_MSVC())
#        else
#            define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_EDG() | EINSUMS_PP_CONFIG_STRICT())
#        endif
#    elif defined(_MWERKS_)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_MWCC())
#    elif defined(_DMC_)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_DMC())
#    elif defined(_BORLANDC_) && _BORLANDC_ >= 0x581
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_STRICT())
#    elif defined(_BORLANDC_) || defined(_IBMC_) || defined(_IBMCPP_) || defined(_SUNPRO_CC)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_BCC())
#    elif defined(_MSC_VER)
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_MSVC())
#    else
#        define EINSUMS_PP_CONFIG_FLAGS() (EINSUMS_PP_CONFIG_STRICT())
#    endif
#endif
#
#/* EINSUMS_PP_CONFIG_EXTENDED_LINE_INFO */
#
#if !defined(EINSUMS_PP_CONFIG_EXTENDED_LINE_INFO)
#    define EINSUMS_PP_CONFIG_EXTENDED_LINE_INFO 0
#endif
#
#/* EINSUMS_PP_CONFIG_ERRORS */
#
#if !defined(EINSUMS_PP_CONFIG_ERRORS)
#    ifdef EINSUMS_DEBUG
#        define EINSUMS_PP_CONFIG_ERRORS 1
#    else
#        define EINSUMS_PP_CONFIG_ERRORS 0
#    endif
#endif
#
#/* EINSUMS_UTIL_VARIADICS */
#
#define EINSUMS_UTIL_VARIADICS_MSVC 0
#if !defined EINSUMS_UTIL_VARIADICS
#    /* variadic support explicitly disabled for all untested compilers */
#    if defined _GCCXML_ || defined _CUDACC_ || defined _PATHSCALE_ || defined _DMC_ || defined _CODEGEARC_ || defined _BORLANDC_ ||       \
        defined _MWERKS_ || (defined _SUNPRO_CC && _SUNPRO_CC < 0x5120) || defined _HP_aCC && !defined _EDG_ || defined _MRC_ ||           \
        defined _SC_ || defined _IBMCPP_ || defined _PGI
#        define EINSUMS_UTIL_VARIADICS 0
#        /* VC++ (C/C++) and Intel C++ Compiler >= 17.0 with MSVC */
#    elif defined _MSC_VER && _MSC_VER >= 1400 &&                                                                                          \
        (defined(_clang_) || !defined _EDG_ || defined(_INTELLISENSE_) || defined(_INTEL_COMPILER) && _INTEL_COMPILER >= 1700)
#        define EINSUMS_UTIL_VARIADICS 1
#        undef EINSUMS_UTIL_VARIADICS_MSVC
#        define EINSUMS_UTIL_VARIADICS_MSVC 1
#        /* Wave (C/C++), GCC (C++) */
#    elif defined _WAVE_ && _WAVE_HAS_VARIADICS_ || defined _GNUC_ && defined _GXX_EXPERIMENTAL_CXX0X_ && _GXX_EXPERIMENTAL_CXX0X_
#        define EINSUMS_UTIL_VARIADICS 1
#        /* EDG-based (C/C++), GCC (C), and unknown (C/C++) */
#    elif !defined _cplusplus && _STDC_VERSION_ >= 199901L || _cplusplus >= 201103L
#        define EINSUMS_UTIL_VARIADICS 1
#    else
#        define EINSUMS_UTIL_VARIADICS 0
#    endif
#elif !EINSUMS_UTIL_VARIADICS + 1 < 2
#    undef EINSUMS_UTIL_VARIADICS
#    define EINSUMS_UTIL_VARIADICS 1
#    if defined _MSC_VER && _MSC_VER >= 1400 &&                                                                                            \
        (defined(_clang_) || defined(_INTELLISENSE_) || (defined(_INTEL_COMPILER) && _INTEL_COMPILER >= 1700) ||                           \
         !(defined _EDG_ || defined _GCCXML_ || defined _CUDACC_ || defined _PATHSCALE_ || defined _DMC_ || defined _CODEGEARC_ ||         \
           defined _BORLANDC_ || defined _MWERKS_ || defined _SUNPRO_CC || defined _HP_aCC || defined _MRC_ || defined _SC_ ||             \
           defined _IBMCPP_ || defined _PGI))
#        undef EINSUMS_UTIL_VARIADICS_MSVC
#        define EINSUMS_UTIL_VARIADICS_MSVC 1
#    endif
#else
#    undef EINSUMS_UTIL_VARIADICS
#    define EINSUMS_UTIL_VARIADICS 0
#endif
#
