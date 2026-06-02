//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Defines.hpp>

#ifdef __cplusplus
#    include <complex>
#endif

/**
 * @def EINSUMS_TRANSACTION_SAFE_DYN
 *
 * If transactional memory is supported by your compiler, expands to the @c transaction_safe_dynamic keyword.
 *
 * @versionadded{2.0.0}
 */
/**
 * @def EINSUMS_TRANSACTION_SAFE
 *
 * If transactional memory is supported by your compiler, expands to the @c transaction_safe keyword.
 *
 * @versionadded{2.0.0}
 */
/**
 * @def EINSUMS_ATOMIC_CANCEL
 *
 * If transactional memory is supported by your compiler, expands to the @c atomic_cancel keyword.
 *
 * @versionadded{2.0.0}
 */
/**
 * @def EINSUMS_ATOMIC_COMMIT
 *
 * If transactional memory is supported by your compiler, expands to the @c atomic_commit keyword.
 *
 * @versionadded{2.0.0}
 */
/**
 * @def EINSUMS_ATOMIC_NOEXCEPT
 *
 * If transactional memory is supported by your compiler, expands to the @c atomic_noexcept keyword.
 *
 * @versionadded{2.0.0}
 */
/**
 * @def EINSUMS_SYNCHRONIZED
 *
 * If transactional memory is supported by your compiler, expands to the @c synchronized keyword.
 *
 * @versionadded{2.0.0}
 */
#if defined(__cpp_transactional_memory) && __cpp_transactional_memory >= 201505L
#    define EINSUMS_TRANSACTION_SAFE_DYN transaction_safe_dynamic
#    define EINSUMS_TRANSACTION_SAFE     transaction_safe
#    define EINSUMS_ATOMIC_CANCEL        atomic_cancel
#    define EINSUMS_ATOMIC_COMMIT        atomic_commit
#    define EINSUMS_ATOMIC_NOEXCEPT      atomic_noexcept
#    define EINSUMS_SYNCHRONIZED         synchronized
#else
#    define EINSUMS_TRANSACTION_SAFE_DYN
#    define EINSUMS_TRANSACTION_SAFE
#    define EINSUMS_ATOMIC_CANCEL
#    define EINSUMS_ATOMIC_COMMIT
#    define EINSUMS_ATOMIC_NOEXCEPT
#    define EINSUMS_SYNCHRONIZED
#endif

/**
 * @def EINSUMS_OMP_PRAGMA
 *
 * Creates a pragma line for an OpenMP call. The text inside does not need the "omp".
 *
 * @param stuff The OpenMP directives to use.
 *
 * @versionadded{1.1.0}
 */

/**
 * @def EINSUMS_OMP_SIMD_PRAGMA
 *
 * Creates a pragma line for an OpenMP call that can be vectorized, if supported by the compiler. The text inside does not need the "omp".
 *
 * @param stuff The OpenMP directives to use.
 *
 * @versionadded{1.1.0}
 */

/**
 * @def EINSUMS_OMP_SIMD
 *
 * Indicates that the following block can be vectorized.
 *
 * @versionadded{1.1.0}
 */

/**
 * @def EINSUMS_PRAGMA
 *
 * Creates a pragma line. Mostly used within other macros.
 *
 * @param stuff The pragma line without quotes.
 *
 * @versionadded{1.1.0}
 */
#define EINSUMS_PRAGMA(stuff) _Pragma(#stuff)
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PRAGMA(stuff)      EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD_PRAGMA(stuff) EINSUMS_PRAGMA(omp stuff simd)
#    define EINSUMS_OMP_SIMD               _Pragma("omp simd")
#else
#    define EINSUMS_OMP_PRAGMA(stuff)      EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD_PRAGMA(stuff) EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD
#endif
/**
 * @def EINSUMS_OMP_PARALLEL_FOR_SIMD
 *
 * Tell the compiler that the following for-loop can be parallelized and vectorized using SIMD.
 *
 * @versionadded{1.1.0}
 */
#define EINSUMS_OMP_PARALLEL_FOR_SIMD EINSUMS_OMP_SIMD_PRAGMA(parallel for)

/**
 * @def EINSUMS_OMP_PARALLEL_FOR
 *
 * Tell the compiler that the following for-loop can be parallelized.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_PARALLEL_FOR EINSUMS_OMP_PRAGMA(parallel for)

/**
 * @def EINSUMS_OMP_PARALLEL
 *
 * Tell the compiler that the following block should be done in parallel.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_PARALLEL EINSUMS_OMP_PRAGMA(parallel)

/**
 * @def EINSUMS_OMP_TASK_FOR
 *
 * Tell the compiler that the following for-loop can be parallelized and should use tasks.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_TASK_FOR EINSUMS_OMP_PRAGMA(taskloop)

/**
 * @def EINSUMS_OMP_TASK_FOR_SIMD
 *
 * Tell the compiler that the following for-loop can be parallelized and vectorized using SIMD and should use tasks.
 *
 * @versionadded{1.1.0}
 */
#define EINSUMS_OMP_TASK_FOR_SIMD EINSUMS_OMP_SIMD_PRAGMA(taskloop)

/**
 * @def EINSUMS_OMP_TASK
 *
 * Start a new task within a parallel region.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_TASK EINSUMS_OMP_PRAGMA(task)

/**
 * @def EINSUMS_OMP_FOR_NOWAIT
 *
 * Tell the compiler that the following for-loop can be parallelized and it should not wait for threads to finish.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_FOR_NOWAIT EINSUMS_OMP_PRAGMA(for nowait)

/**
 * @def EINSUMS_OMP_CRITICAL
 *
 * Indicates that the following block should only be executed by one thread at a time.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_OMP_CRITICAL EINSUMS_OMP_PRAGMA(critical)

#if defined(__GNUC__) && defined(__cplusplus)

// gcc does not have reductions for complex values.
#    pragma omp declare reduction(+ : std::complex<float> : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) initializer(omp_priv = omp_orig)

#endif

/**
 * @def EINSUMS_ALWAYS_INLINE
 *
 * Tell the compiler to inline a function.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_ALWAYS_INLINE __attribute__((always_inline)) inline

// clang-format off
#if defined(_MSC_VER)
/**
 * @def EINSUMS_DISABLE_WARNING_PUSH
 *
 * Pushes the current warning ignore state so that ignore rules can be changed and then reverted.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_DISABLE_WARNING_PUSH           __pragma(warning(push))
/**
 * @def EINSUMS_DISABLE_WARNING_POP
 *
 * Ppp the current warning ignore state to revert the changes to the warning ignore rules.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_DISABLE_WARNING_POP            __pragma(warning(pop))
/**
 * @def EINSUMS_DISABLE_WARNING
 *
 * Disables a specific warning. The behavior may differ between compilers.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))

/**
 * @def EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
 *
 * Disable warnings that a function that normally returns C's <tt>_Complex float</tt> or <tt>_Complex double</tt> is externed as
 * a function that returns C++'s <tt>std::complex<float></tt> or <tt>std::complex<double></tt>.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
/**
 * @def EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
 *
 * Disable warnings of deprecated functions, classes, variables, etc.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#    define EINSUMS_DISABLE_WARNING_PUSH          EINSUMS_PRAGMA(GCC diagnostic push)
#    define EINSUMS_DISABLE_WARNING_POP           EINSUMS_PRAGMA(GCC diagnostic pop)
#    define EINSUMS_DISABLE_WARNING(warningName)  EINSUMS_PRAGMA(GCC diagnostic ignored #warningName)
#ifndef __clang__
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS EINSUMS_DISABLE_WARNING(-Wdeprecated-declarations)
#else
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE EINSUMS_DISABLE_WARNING(-Wreturn-type-c-linkage)
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS EINSUMS_DISABLE_WARNING(-Wdeprecated-declarations)// other warnings you want to deactivate...
#endif

#else
#    define EINSUMS_DISABLE_WARNING_PUSH
#    define EINSUMS_DISABLE_WARNING_POP
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...
#endif
// clang-format on

// clang-format off
#if defined(__GNUC__)

// macros to facilitate handling of compiler-specific issues
/// Returns the GCC version einsums is compiled with. Only set if compiled with GCC.
/// @versionadded{1.0.0}
#  define EINSUMS_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)

#  define EINSUMS_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS 1

#  undef EINSUMS_CLANG_VERSION
#  undef EINSUMS_INTEL_VERSION

#else

#  undef EINSUMS_GCC_VERSION

#endif

#if defined(__clang__)

/// Returns the Clang version einsums is compiled with. Only set if compiled with
/// Clang.
/// @versionadded{1.0.0}
#  define EINSUMS_CLANG_VERSION \
 (__clang_major__*10000 + __clang_minor__*100 + __clang_patchlevel__)

#  undef EINSUMS_INTEL_VERSION

#else

#  undef EINSUMS_CLANG_VERSION

#endif

#if defined(__INTEL_COMPILER)
/// Returns the Intel Compiler version einsums is compiled with. Only set if
/// compiled with the Intel Compiler.
/// @versionadded{1.0.0}
# define EINSUMS_INTEL_VERSION __INTEL_COMPILER
# if defined(_WIN32) || (_WIN64)
#  define EINSUMS_INTEL_WIN EINSUMS_INTEL_VERSION
// suppress a couple of benign warnings
   // template parameter "..." is not used in declaring the parameter types of
   // function template "..."
#  pragma warning disable 488
   // invalid redeclaration of nested class
#  pragma warning disable 1170
   // decorated name length exceeded, name was truncated
#  pragma warning disable 2586
# endif
#else

#  undef EINSUMS_INTEL_VERSION

#endif

#if defined(_MSC_VER)
/// This macro is set if the compilation is for Windows.
/// @versionadded{1.0.0}
#  define EINSUMS_WINDOWS
/// This macro is set if the compilation is with MSVC.
/// @versionadded{1.0.0}
#  define EINSUMS_MSVC _MSC_VER
#  define EINSUMS_MSVC_WARNING_PRAGMA
#  if defined(__NVCC__)
#    define EINSUMS_MSVC_NVCC
#  endif
#  define EINSUMS_CDECL __cdecl
#endif

#if defined(__MINGW32__)
#   define EINSUMS_WINDOWS
/// This macro is set if the compilation is with Mingw.
/// @versionadded{1.0.0}
#   define EINSUMS_MINGW
#endif

// Detecting CUDA compilation mode
// Detecting nvhpc
// The CUDA version will also be defined, so we end this block with #endif,
// not #elif
#if defined(__NVCOMPILER)
/**
 * @def EINSUMS_NVHPC_VERSION
 *
 * Gives the version of the NVidia HPC compiler used to compile Einsums.
 *
 * @versionadded{1.0.0}
 */
#  define EINSUMS_NVHPC_VERSION (__NVCOMPILER_MAJOR__ * 10000 + __NVCOMPILER_MINOR__ * 100 + __NVCOMPILER_PATCHLEVEL__)
#endif

// Detecting NVCC/CUDA
#if defined(__NVCC__) || defined(__CUDACC__)
// NVCC build version numbers can be high (without limit?) so we leave it out
// from the version definition
/**
 * @def EINSUMS_CUDA_VERSION
 *
 * The version of CUDA used to compile Einsums.
 *
 * @versionadded{1.0.0}
 */
#  define EINSUMS_CUDA_VERSION (__CUDACC_VER_MAJOR__*100 + __CUDACC_VER_MINOR__)
/**
 * @def EINSUMS_COMPUTE_CODE
 *
 * Only available when Einsums was compiled with GPU capabilities. If not present, then there are no GPU capabilities.
 *
 * @versionadded{1.0.0}
 */
#  define EINSUMS_COMPUTE_CODE
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0
     // nvcc compiling CUDA code, device mode.
/**
 * @def EINSUMS_COMPUTE_DEVICE_CODE
 *
 * Only available when Einsums was compiled with GPU capabilities and only during compilation for the graphics device.
 *
 * @versionadded{1.0.0}
 */
#    define EINSUMS_COMPUTE_DEVICE_CODE
#  endif
// Detecting Clang CUDA
#elif defined(__clang__) && defined(__CUDA__)
#  define EINSUMS_COMPUTE_CODE
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0
     // clang compiling CUDA code, device mode.
#    define EINSUMS_COMPUTE_DEVICE_CODE
#  endif
// Detecting HIPCC
#elif defined(__HIPCC__)
#  include <hip/hip_version.h>
/**
 * @def EINSUMS_HIP_VERSION
 *
 * The version of HIP used to compile Einsums.
 *
 * @versionadded{1.0.0}
 */
#  define EINSUMS_HIP_VERSION HIP_VERSION
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-copy"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#  endif
   // Not like nvcc, the __device__ __host__ function decorators are not defined
   // by the compiler
#  include <hip/hip_runtime_api.h>
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  endif
#  define EINSUMS_COMPUTE_CODE
#  if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ != 0
     // hipclang compiling CUDA/HIP code, device mode.
#    define EINSUMS_COMPUTE_DEVICE_CODE
#  endif
#endif

#if !defined(EINSUMS_COMPUTE_DEVICE_CODE)
/**
 * @def EINSUMS_COMPUTE_HOST_CODE
 *
 * Available whenever Einsums is not being compiled for the device. This means that it is defined when GPU capabilities
 * are turned off and when they are turned on during compilation for the host.
 *
 * @versionadded{1.0.0}
 */
#  define EINSUMS_COMPUTE_HOST_CODE
#endif

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @def EINSUMS_DEVICE
 *
 * When compiled with GPU capabilities, this expands to @c __device__ . Otherwise, it expands to nothing.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_DEVICE __device__
/**
 * @def EINSUMS_HOST
 *
 * When compiled with GPU capabilities, this expands to @c __host__ . Otherwise, it expands to nothing.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_HOST __host__
/**
 * @def EINSUMS_CONSTANT
 *
 * When compiled with GPU capabilities, this expands to @c __constant__ . Otherwise, it expands to nothing.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_CONSTANT __constant__
/**
 * @def EINSUMS_HOSTDEV
 *
 * When compiled with GPU capabilities, this expands to @c __host__ @c __device__ . Otherwise, it expands to nothing.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_HOSTDEV __host__ __device__
#else
#define EINSUMS_DEVICE
#define EINSUMS_HOST
#define EINSUMS_CONSTANT
#define EINSUMS_HOSTDEV
#endif

/**
 * @copydoc EINSUMS_HOSTDEV
 */
#define EINSUMS_HOST_DEVICE EINSUMS_HOST EINSUMS_DEVICE

#if defined(__NVCC__)
/**
 * @def EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
 *
 * Silences warnings about calling a host function from a device function, which is common when not using host/device specifiers.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE #pragma hd_warning_disable
#else
#define EINSUMS_NVCC_PRAGMA_HD_WARNING_DISABLE
#endif

#if !defined(EINSUMS_CDECL)
/**
 * @def EINSUMS_CDECL
 *
 * Mark a function as using the @c __cdecl calling convention.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_CDECL
#endif

// clang-format on
#if defined(EINSUMS_HAVE_SANITIZERS)
#    if defined(__has_feature)
#        if __has_feature(address_sanitizer)
/**
 * @def EINSUMS_HAVE_ADDRESS_SANITIZER
 *
 * Defined if Einsums was compiled with address sanitization (ASAN).
 *
 * @versionadded{1.0.0}
 */
#            define EINSUMS_HAVE_ADDRESS_SANITIZER
#            if defined(EINSUMS_GCC_VERSION) || defined(EINSUMS_CLANG_VERSION)
/**
 * @def EINSUMS_NO_SANITIZE_ADDRESS
 *
 * Marks a function to not be sanitized.
 *
 * @versionadded{1.0.0}
 */
#                define EINSUMS_NO_SANITIZE_ADDRESS __attribute__((no_sanitize("address")))
#            endif
#        endif
#        if __has_feature(thread_sanitizer)
/**
 * @def EINSUMS_HAVE_THREAD_SANITIZER
 *
 * Defined if Einsums was compiled with thread sanitization.
 *
 * @versionadded{1.0.0}
 */
#            define EINSUMS_HAVE_THREAD_SANITIZER
#        endif
#    elif defined(__SANITIZE_ADDRESS__) // MSVC defines this
#        define EINSUMS_HAVE_ADDRESS_SANITIZER
#    endif
#endif

#if !defined(EINSUMS_NO_SANITIZE_ADDRESS)
#    define EINSUMS_NO_SANITIZE_ADDRESS
#endif
