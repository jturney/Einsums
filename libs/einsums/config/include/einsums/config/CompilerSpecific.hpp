//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config/defines.hpp>

#if defined(DOXYGEN)
/// Returns the GCC version einsums is compiled with. Only set if compiled with GCC.
#    define EINSUMS_GCC_VERSION
/// Returns the Clang version einsums is compiled with. Only set if compiled with Clang.
#    define EINSUMS_CLANG_VERSION
/// Returns the Intel Compler version einsus is compiled with. Only st if compiled with the Intel Compiler.
#    define EINSUMS_INTEL_VERSION
/// This macro is set if the compilation is with MSVC.
#    define EINSUMS_MSVC
/// This macro is set if the compilation is with Mingw.
#    define EINSUMS_MINGW
/// This macro is set if the compilation is for Windows.
#    define EINSUMS_WINDOWS
#else

#    if defined(__GNUC__)

// macros to facilitate handling of compiler-specific issues
#        define EINSUMS_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#        define EINSUMS_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1

#        undef EINSUMS_CLANG_VERSION
#        undef EINSUMS_INTEL_VERSION

#    else

#        undef EINSUMS_GCC_VERSION

#    endif

#    if defined(__clang__)

#        define EINSUMS_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)

#        undef EINSUMS_INTEL_VERSION

#    else

#        undef EINSUMS_CLANG_VERSION

#    endif

#    if defined(__INTEL_COMPILER)
#        define EINSUMS_INTEL_VERSION __INTEL_COMPILER
#        if defined(_WIN32) || (_WIN64)
#            define EINSUMS_INTEL_WIN EINSUMS_INTEL_VERSION
// suppress a couple of benign warnings
// template parameter "..." is not used in declaring the parameter types of
// function template "..."
#            pragma warning disable 488
// invalid redeclaration of nested class
#            pragma warning disable 1170
// decorated name length exceeded, name was truncated
#            pragma warning disable 2586
#        endif
#    else

#        undef EINSUMS_INTEL_VERSION

#    endif

#    if defined(_MSC_VER)
#        define EINSUMS_WINDOWS
#        define EINSUMS_MSVC _MSC_VER
#        define EINSUMS_MSVC_WARNING_PRAGMA
#        if defined(__NVCC__)
#            define EINSUMS_MSVC_NVCC
#        endif
#        define EINSUMS_CDECL __cdecl
#    endif

#    if defined(__MINGW32__)
#        define EINSUMS_WINDOWS
#        define EINSUMS_MINGW
#    endif

// Detecting CUDA compilation mode
// Detecting nvhpc
// The CUDA version will also be defined so we end this block with with #endif,
// not #elif
#    if defined(__NVCOMPILER)
#        define EINSUMS_NVHPC_VERSION                                                                                  \
            (__NVCOMPILER_MAJOR__ * 10000 + __NVCOMPILER_MINOR__ * 100 + __NVCOMPILER_PATCHLEVEL__)
#    endif

// Detecting NVCC/CUDA
#    if defined(__NVCC__) || defined(__CUDACC__)
// NVCC build version numbers can be high (without limit?) so we leave it out
// from the version definition
#        define EINSUMS_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
#        define EINSUMS_COMPUTE_CODE
#        if defined(__CUDA_ARCH__)
// nvcc compiling CUDA code, device mode.
#            define EINSUMS_COMPUTE_DEVICE_CODE
#        endif
// Detecting Clang CUDA
#    elif defined(__clang__) && defined(__CUDA__)
#        define EINSUMS_COMPUTE_CODE
#        if defined(__CUDA_ARCH__)
// clang compiling CUDA code, device mode.
#            define EINSUMS_COMPUTE_DEVICE_CODE
#        endif
// Detecting HIPCC
#    elif defined(__HIPCC__)
#        include <hip/hip_version.h>
#        define EINSUMS_HIP_VERSION HIP_VERSION
#        if defined(__clang__)
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wdeprecated-copy"
#            pragma clang diagnostic ignored "-Wunused-parameter"
#        endif
// Not like nvcc, the __device__ __host__ function decorators are not defined
// by the compiler
#        include <hip/hip_runtime_api.h>
#        if defined(__clang__)
#            pragma clang diagnostic pop
#        endif
#        define EINSUMS_COMPUTE_CODE
#        if defined(__HIP_DEVICE_COMPILE__)
// hipclang compiling CUDA/HIP code, device mode.
#            define EINSUMS_COMPUTE_DEVICE_CODE
#        endif
#    endif

#    if !defined(EINSUMS_COMPUTE_DEVICE_CODE)
#        define EINSUMS_COMPUTE_HOST_CODE
#    endif

#    if defined(EINSUMS_COMPUTE_CODE)
#        define EINSUMS_DEVICE   __device__
#        define EINSUMS_HOST     __host__
#        define EINSUMS_CONSTANT __constant__
#    else
#        define EINSUMS_DEVICE
#        define EINSUMS_HOST
#        define EINSUMS_CONSTANT
#    endif
#    define EINSUMS_HOST_DEVICE EINSUMS_HOST EINSUMS_DEVICE

#    if defined(__NVCC__)
#        define EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE #pragma hd_warning_disable
#    else
#        define EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
#    endif

#    if !defined(EINSUMS_CDECL)
#        define EINSUMS_CDECL
#    endif

#    if defined(EINSUMS_HAVE_SANITIZERS)
#        if defined(__has_feature)
#            if __has_feature(address_sanitizer)
#                define EINSUMS_HAVE_ADDRESS_SANITIZER
#            endif
#        elif defined(__SANITIZE_ADDRESS__) // MSVC defines this
#            define EINSUMS_HAVE_ADDRESS_SANITIZER
#        endif
#    endif

#endif
