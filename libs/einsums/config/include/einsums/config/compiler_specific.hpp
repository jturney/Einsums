//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#    define EINSUMS_OMP_SIMD         _Pragma("omp simd")
#    define EINSUMS_OMP_PARALLEL     _Pragma("omp parallel")
#    define EINSUMS_OMP_TASK_FOR     _Pragma("omp taskloop simd")
#    define EINSUMS_OMP_TASK         _Pragma("omp task")
#    define EINSUMS_OMP_FOR_NOWAIT   _Pragma("omp for nowait")
#    define EINSUMS_OMP_CRITICAL     _Pragma("omp critical")
#else
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for default(none)")
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL   _Pragma("omp parallel default(none)")
#    define EINSUMS_OMP_TASK_FOR   _Pragma("omp taskloop")
#    define EINSUMS_OMP_TASK       _Pragma("omp task")
#    define EINSUMS_OMP_FOR_NOWAIT _Pragma("omp for nowait")
#    define EINSUMS_OMP_CRITICAL   _Pragma("omp critical")
#endif

#define EINSUMS_ALWAYS_INLINE __attribute__((always_inline)) inline

// clang-format off
#if defined(_MSC_VER)
#    define EINSUMS_DISABLE_WARNING_PUSH           __pragma(warning(push))
#    define EINSUMS_DISABLE_WARNING_POP            __pragma(warning(pop))
#    define EINSUMS_DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))

#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#    define EINSUMS_DO_PRAGMA(X)                 _Pragma(#X)
#    define EINSUMS_DISABLE_WARNING_PUSH         EINSUMS_DO_PRAGMA(GCC diagnostic push)
#    define EINSUMS_DISABLE_WARNING_POP          EINSUMS_DO_PRAGMA(GCC diagnostic pop)
#    define EINSUMS_DISABLE_WARNING(warningName) EINSUMS_DO_PRAGMA(GCC diagnostic ignored #warningName)

#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE EINSUMS_DISABLE_WARNING(-Wreturn-type-c-linkage)
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS EINSUMS_DISABLE_WARNING(-Wdeprecated-declarations)
// other warnings you want to deactivate...

#else
#    define EINSUMS_DISABLE_WARNING_PUSH
#    define EINSUMS_DISABLE_WARNING_POP
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...

#endif
// clang-format on